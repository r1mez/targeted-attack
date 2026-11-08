"""Evaluate final soft counterfactuals with separate proxy and target probabilities."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from statistics import mean, median

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch

from src.counterfactuals.soft_counterfactual import counterfactual_loss, generate_soft, load_evaluation_models
from src.counterfactuals.training import expand_target_tasks
from src.data.ba3motif_dataset import (
    CLASS_NAMES, load_graph_split, load_originals, select_smoke_data, select_validation,
)
from src.experiments.runtime import ROOT, ExperimentRun, TargetLedger, append_jsonl, write_json
from src.models.ba3motif_model_access import DenseProxy, ProbabilityOracle, load_frozen, state_hash


def statistics(rows):
    count = len(rows)
    if not count:
        return None
    hits = sum(row["target_prediction"] == row["target"] for row in rows)
    flips = sum(row["target_prediction"] != row["source"] for row in rows)
    return {
        "tasks": count,
        "targeted_hits": hits,
        "targeted_asr": hits / count,
        "any_flips": flips,
        "any_flip_rate": flips / count,
        "proxy_targeted_asr": sum(row["proxy_prediction"] == row["target"] for row in rows) / count,
        "soft_l1_mean": mean(row["soft_l1"] for row in rows),
        "soft_l1_median": median(row["soft_l1"] for row in rows),
        "soft_l1_mean_success": (
            mean(row["soft_l1"] for row in rows if row["target_prediction"] == row["target"])
            if any(row["target_prediction"] == row["target"] for row in rows) else None
        ),
    }


def graph_statistics(rows):
    """Keep two target attempts per source graph distinct from task-wise rates."""
    if not rows:
        return None
    by_graph = defaultdict(list)
    for row in rows:
        by_graph[row["source_id"]].append(row)
    flips = hits = 0
    for group in by_graph.values():
        source = group[0]["source"]
        expected_targets = set(range(len(CLASS_NAMES))) - {source}
        if (any(row["source"] != source for row in group)
                or len(group) != len(expected_targets)
                or {row["target"] for row in group} != expected_targets):
            raise RuntimeError("Each source graph must have exactly one attempt per other target class")
        flips += any(row["target_prediction"] != source for row in group)
        hits += any(row["target_prediction"] == row["target"] for row in group)
    count = len(by_graph)
    return {"graphs": count, "attempts_per_graph": len(CLASS_NAMES) - 1,
            "any_flip_graphs": flips, "any_flip_rate": flips / count,
            "any_targeted_hit_graphs": hits, "any_targeted_hit_rate": hits / count}


def evaluate(config, run):
    settings = config["stages"]["evaluate"]
    data, cache = load_originals(run.paths)
    ae, flow, mean_z, std_z, artifacts = load_evaluation_models(
        config, run.paths, training_ids=data["training"]["ids"],
    )
    ae.to(run.device)
    flow.to(run.device)
    oracle = ProbabilityOracle(run.paths["target"])
    proxy_model = load_frozen(run.paths["proxy"]).to(run.device)
    proxy = DenseProxy(proxy_model, data["training"]["x"].size(1))
    initial_proxy = state_hash(proxy_model)
    initial_ae, initial_flow = state_hash(ae), state_hash(flow)
    ledger = TargetLedger(oracle, cache, run.directory / "queries.jsonl",
                          run.query_limit, run.check_time)
    split = settings["split"]
    if split not in ("evaluation", "testing"):
        raise ValueError("An approved evaluation split must be evaluation or testing")
    if run.phase == "smoke":
        original = select_smoke_data(
            select_validation(data, run.paths, config),
            config["smoke"]["graphs_per_class"], settings["seed"],
        )
        actual_split = "evaluation_smoke"
    elif split == "testing":
        original = load_graph_split(run.paths["data"], "testing")
        probabilities = []
        for index, graph_id in enumerate(original["ids"].tolist()):
            values, _ = ledger.query(original["x"][index], original["adjacency"][index],
                                     role="original", source_id=graph_id)
            probabilities.append(values)
        original["probabilities"] = torch.tensor(probabilities, dtype=torch.float32)
        original["classes"] = original["probabilities"].argmax(-1)
        actual_split = "testing"
    else:
        original, actual_split = data["evaluation"], "evaluation"
    tasks = expand_target_tasks(original, settings["target_policy"])
    batch_size = settings["batch_size"]
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("Evaluation batch_size must be explicitly positive")
    write_json(run.directory / "samples.json", {
        "split": actual_split, "original_ids": original["ids"].tolist(),
        "tasks": [{"source_id": int(i), "source": int(s), "target": int(t)}
                  for i, s, t in zip(tasks["ids"], tasks["classes"], tasks["targets"])],
        "initialization": artifacts,
    })
    rows, generated_graphs = [], []
    for start in range(0, len(tasks["ids"]), batch_size):
        run.check_time()
        block = {name: value[start:start + batch_size].to(run.device)
                 for name, value in tasks.items()}
        with torch.no_grad():
            generated = generate_soft(
                ae, flow, mean_z, std_z, block["x"], block["adjacency"], block["ids"],
                block["classes"], block["targets"], ae_seed=config["fixed_auxiliary_seed"],
                steps=settings["steps"], multiplier=settings["multiplier"],
                check_time=run.check_time,
            )
            losses = counterfactual_loss(proxy, block["x"], generated["adjacency"],
                                         block["targets"], block["adjacency"])
            proxy_probabilities = losses["proxy_logits"].softmax(-1)
        generated_graphs.append(generated["adjacency"].detach().cpu())
        for index in range(len(block["ids"])):
            graph_id = int(block["ids"][index])
            target = int(block["targets"][index])
            source = int(block["classes"][index])
            values, _ = ledger.query(
                block["x"][index], generated["adjacency"][index].detach().cpu(),
                role="soft_counterfactual", source_id=graph_id, source=source, target=target,
            )
            proxy_values = proxy_probabilities[index].detach().cpu().tolist()
            row = {
                "source_id": graph_id, "source": source, "target": target,
                "true_label": int(block["labels"][index]),
                "original_probabilities": block["probabilities"][index].tolist(),
                "proxy_probabilities": proxy_values, "target_probabilities": values,
                "proxy_prediction": max(range(len(CLASS_NAMES)), key=proxy_values.__getitem__),
                "target_prediction": max(range(len(CLASS_NAMES)), key=values.__getitem__),
                "soft_l1": float(losses["per_graph_distance"][index]),
            }
            append_jsonl(run.directory / "outcomes.jsonl", row)
            rows.append(row)
        run.log({"event": "evaluation_progress", "completed_tasks": len(rows),
                 "total_tasks": len(tasks["ids"])})
    if len(rows) != len(original["ids"]) * (len(CLASS_NAMES) - 1):
        raise RuntimeError("Soft evaluation did not cover all explicitly requested tasks")
    for model, before in ((ae, initial_ae), (flow, initial_flow), (proxy_model, initial_proxy)):
        if (model.training or state_hash(model) != before
                or any(p.requires_grad or p.grad is not None for p in model.parameters())):
            raise RuntimeError("A frozen model changed or acquired gradients during evaluation")
    oracle.check_frozen()
    by_source, by_direction, by_label = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        by_source[CLASS_NAMES[row["source"]]].append(row)
        by_direction[f"{CLASS_NAMES[row['source']]}->{CLASS_NAMES[row['target']]}"].append(row)
        by_label[CLASS_NAMES[row["true_label"]]].append(row)
    torch.save({"adjacency": torch.cat(generated_graphs), "ids": tasks["ids"],
                "source": tasks["classes"], "targets": tasks["targets"],
                "x": tasks["x"], "original_adjacency": tasks["adjacency"],
                "output": "continuous symmetric soft adjacency; original loops preserved"},
               run.directory / "soft_graphs.pt")
    correct = [row for row in rows if row["source"] == row["true_label"]]
    incorrect = [row for row in rows if row["source"] != row["true_label"]]
    write_json(run.directory / "summary.json", {
        "evaluation_split": actual_split, "graphs": len(original["ids"]),
        "generation_device": str(run.device), "target_oracle_device": "cpu",
        "all": statistics(rows),
        "by_source": {key: statistics(by_source[key]) for key in CLASS_NAMES},
        "by_direction": {f"{source}->{target}": statistics(by_direction[f"{source}->{target}"])
                         for source in CLASS_NAMES for target in CLASS_NAMES if source != target},
        "by_true_label": {key: statistics(by_label[key]) for key in CLASS_NAMES},
        "per_graph": graph_statistics(rows),
        "per_graph_by_source": {key: graph_statistics(by_source[key]) for key in CLASS_NAMES},
        "original_accuracy": float((original["classes"] == original["labels"]).float().mean()),
        "original_correct_graphs": int((original["classes"] == original["labels"]).sum()),
        "correct_originals": statistics(correct) if correct else None,
        "incorrect_originals": statistics(incorrect) if incorrect else None,
        "proxy_forwards": proxy.forwards, "target_queries": ledger.record(),
        "distance": "sum_i<j |A_soft-A_original| per graph, no discretization",
        "L_boundary": 0,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "full"))
    parser.add_argument("--config", type=Path, default=ROOT / "src/configs/ba3motif_flow.json")
    args = parser.parse_args()
    args.config = (ROOT / args.config).resolve()
    config = json.loads(args.config.read_text())
    with ExperimentRun(config, args.config, args.phase, "evaluate") as run:
        evaluate(config, run)
    print(run.directory)


if __name__ == "__main__":
    main()
