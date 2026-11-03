"""Distill a same-architecture BA3Motif surrogate from target probabilities."""

import argparse
from collections import Counter
from hashlib import sha256
import itertools
import json
import math
import os
import platform
from pathlib import Path
import random
import shlex
import sys
import tempfile

import networkx as nx
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from src.data.ba3motif_dataset import CLASS_NAMES, DATA_SHA256, load_ba3motif
from src.models.ba3motif_gnn import BA3MotifLEConv


EDIT_BUDGETS = (1, 2, 4, 5)
MAX_ATTEMPTS = 500
TRAIN_COVERAGE_FLOOR = 0.95
SMOKE_TRAIN_PER_CLASS = 3
SMOKE_VALIDATION_PER_CLASS = 1
SMOKE_EPOCHS = 2
DEFAULT_RUN_DIR = Path("results/ba3motif-surrogate-distillation/distill-001")
DEFAULT_OUTPUT = Path(
    "src/params/gnns/ba3motif/ba3motif_leconv_surrogate.pt"
)


def undirected_edges(graph):
    return {
        (min(int(src), int(dst)), max(int(src), int(dst)))
        for src, dst in graph.edge_index.t().tolist()
        if int(src) != int(dst)
    }


def self_loops(graph):
    return [
        (int(src), int(dst))
        for src, dst in graph.edge_index.t().tolist()
        if int(src) == int(dst)
    ]


def is_symmetric(graph):
    directed = {(int(src), int(dst)) for src, dst in graph.edge_index.t().tolist()}
    return all((dst, src) in directed for src, dst in directed if src != dst)


def is_connected(graph):
    network = nx.Graph()
    network.add_nodes_from(range(graph.num_nodes))
    network.add_edges_from(undirected_edges(graph))
    return nx.is_connected(network)


def make_candidate(source, edges):
    directed = sorted(
        edge
        for src, dst in edges
        for edge in ((src, dst), (dst, src))
    )
    directed.extend(self_loops(source))
    if directed:
        edge_index = torch.tensor(directed, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    candidate = Data(x=source.x.clone(), edge_index=edge_index)
    if candidate.num_nodes != source.num_nodes or not torch.equal(candidate.x, source.x):
        raise RuntimeError("Candidate construction changed nodes or features")
    if self_loops(candidate) != self_loops(source):
        raise RuntimeError("Candidate construction changed source self-loops")
    if not is_symmetric(candidate):
        raise RuntimeError("Candidate edge list is not symmetric")
    return candidate


def create_edit(source, budget, rng):
    original = undirected_edges(source)
    node_pairs = list(itertools.combinations(range(source.num_nodes), 2))
    seen = set()
    disconnected = 0

    for attempt in range(1, MAX_ATTEMPTS + 1):
        flips = tuple(sorted(rng.sample(node_pairs, budget)))
        signature = tuple(sorted(original.symmetric_difference(flips)))
        if signature in seen:
            continue
        seen.add(signature)
        candidate_edges = set(signature)
        candidate = make_candidate(source, candidate_edges)
        additions = candidate_edges - original
        deletions = original - candidate_edges
        if len(additions) + len(deletions) != budget:
            raise RuntimeError("Candidate does not match its requested edit budget")
        if is_connected(candidate):
            return candidate, {
                "status": "complete",
                "attempts": attempt,
                "unique_candidates": len(seen),
                "disconnected_candidates": disconnected,
                "additions": len(additions),
                "deletions": len(deletions),
            }
        disconnected += 1

    return None, {
        "status": "attempt_limit",
        "attempts": MAX_ATTEMPTS,
        "unique_candidates": len(seen),
        "disconnected_candidates": disconnected,
    }


def build_query_records(dataset_splits, seed):
    rng = random.Random(seed)
    records = []
    generation = []
    source_graphs = {}

    for split in ("training", "evaluation"):
        graphs = sorted(dataset_splits[split], key=lambda item: int(item.idx))
        for graph_number, source in enumerate(graphs, start=1):
            label = int(source.y)
            sample_idx = int(source.idx)
            source_id = (split, sample_idx)
            source_graphs[source_id] = source
            if not is_symmetric(source):
                raise ValueError(f"Input graph {source_id} is not undirected/symmetric")

            records.append({
                "sample_id": f"{split}:{sample_idx}:budget-0",
                "split": split,
                "source_idx": sample_idx,
                "label": label,
                "budget": 0,
                "attempts": 0,
                "additions": 0,
                "deletions": 0,
                "graph": source,
                "teacher_probabilities": None,
            })

            for budget in EDIT_BUDGETS:
                candidate, stats = create_edit(source, budget, rng)
                generation.append({
                    "split": split,
                    "source_idx": sample_idx,
                    "label": label,
                    "budget": budget,
                    **stats,
                })
                if candidate is None:
                    continue
                records.append({
                    "sample_id": f"{split}:{sample_idx}:budget-{budget}",
                    "split": split,
                    "source_idx": sample_idx,
                    "label": label,
                    "budget": budget,
                    "attempts": stats["attempts"],
                    "additions": stats["additions"],
                    "deletions": stats["deletions"],
                    "graph": candidate,
                    "teacher_probabilities": None,
                })
            if graph_number % 100 == 0 or graph_number == len(graphs):
                print(
                    f"Prepared {split} edit queries for {graph_number}/{len(graphs)} graphs",
                    flush=True,
                )
    return records, generation, source_graphs


def generation_summary(generation, dataset_splits):
    output = {}
    for split in ("training", "evaluation"):
        output[split] = {}
        for label, class_name in enumerate(CLASS_NAMES):
            input_count = sum(int(graph.y) == label for graph in dataset_splits[split])
            output[split][class_name] = {}
            for budget in EDIT_BUDGETS:
                outcomes = [
                    item for item in generation
                    if item["split"] == split
                    and item["label"] == label
                    and item["budget"] == budget
                ]
                succeeded = sum(item["status"] == "complete" for item in outcomes)
                output[split][class_name][str(budget)] = {
                    "source_graphs": input_count,
                    "generated": succeeded,
                    "coverage": succeeded / input_count if input_count else 0.0,
                    "failures": dict(Counter(
                        item["status"] for item in outcomes
                        if item["status"] != "complete"
                    )),
                }
    return output


class ProbabilityOracle:
    def __init__(self, checkpoint_path):
        payload = torch.load(checkpoint_path, map_location="cpu")
        self.model_kwargs = payload["model_kwargs"]
        self.model = BA3MotifLEConv(**self.model_kwargs)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        self.cache = {}
        self.requests = 0
        self.forwards = 0

    @staticmethod
    def key(graph):
        features = graph.x.detach().cpu().contiguous().numpy().tobytes()
        return (
            tuple(sorted(undirected_edges(graph))),
            tuple(self_loops(graph)),
            features,
        )

    @torch.no_grad()
    def query(self, graph):
        self.requests += 1
        key = self.key(graph)
        if key in self.cache:
            return list(self.cache[key])
        batch = Batch.from_data_list([graph])
        probabilities = self.model(
            batch.x, batch.edge_index, batch.batch
        ).softmax(dim=-1)[0].cpu().tolist()
        if not all(math.isfinite(value) for value in probabilities):
            raise RuntimeError("Target GNN returned non-finite probabilities")
        if abs(sum(probabilities) - 1.0) > 1e-5:
            raise RuntimeError("Target GNN probabilities do not sum to one")
        self.cache[key] = tuple(probabilities)
        self.forwards += 1
        return probabilities


def public_record(record):
    return {key: value for key, value in record.items() if key != "graph"}


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_query(record, output):
    output.write(json.dumps(public_record(record)) + "\n")
    output.flush()


def select_smoke_sources(source_graphs, seed):
    rng = random.Random(seed)
    selected = set()
    for split, count in (
        ("training", SMOKE_TRAIN_PER_CLASS),
        ("evaluation", SMOKE_VALIDATION_PER_CLASS),
    ):
        for label in range(3):
            candidates = sorted(
                [
                    (key, graph) for key, graph in source_graphs.items()
                    if key[0] == split and int(graph.y) == label
                ],
                key=lambda item: item[0][1],
            )
            if len(candidates) < count:
                raise ValueError(f"Not enough {split} graphs for smoke test class {label}")
            selected.update(key for key, _ in rng.sample(candidates, count))
    return selected


def as_training_data(record):
    graph = record["graph"]
    return Data(
        x=graph.x,
        edge_index=graph.edge_index,
        y=torch.tensor([record["label"]], dtype=torch.long),
        edit_budget=torch.tensor([record["budget"]], dtype=torch.long),
        teacher_probabilities=torch.tensor(
            [record["teacher_probabilities"]], dtype=torch.float32
        ),
    )


def batch_kl(logits, teacher_probabilities):
    return F.kl_div(
        F.log_softmax(logits, dim=-1),
        teacher_probabilities,
        reduction="batchmean",
    )


def train_epoch(model, loader, optimizer):
    model.train()
    loss_sum = 0.0
    sample_count = 0
    for batch in loader:
        batch = batch.to("cpu")
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = batch_kl(logits, batch.teacher_probabilities)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite surrogate distillation loss")
        loss.backward()
        optimizer.step()
        count = int(batch.y.numel())
        loss_sum += float(loss.detach()) * count
        sample_count += count
    return loss_sum / max(1, sample_count)


@torch.no_grad()
def evaluate(model, records, batch_size):
    dataset = [as_training_data(record) for record in records]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    collected = []
    model.eval()
    for batch in loader:
        batch = batch.to("cpu")
        probabilities = model(
            batch.x, batch.edge_index, batch.batch
        ).softmax(dim=-1)
        teacher = batch.teacher_probabilities
        if not torch.isfinite(probabilities).all():
            raise RuntimeError("Surrogate returned non-finite probabilities")
        kl_values = F.kl_div(
            probabilities.clamp_min(1e-12).log(),
            teacher,
            reduction="none",
        ).sum(dim=-1)
        tv_values = 0.5 * (probabilities - teacher).abs().sum(dim=-1)
        agreements = probabilities.argmax(dim=-1) == teacher.argmax(dim=-1)
        for i in range(len(batch.y)):
            collected.append({
                "label": int(batch.y[i]),
                "teacher_class": int(teacher[i].argmax()),
                "budget": int(batch.edit_budget[i]),
                "kl": float(kl_values[i]),
                "total_variation": float(tv_values[i]),
                "argmax_agreement": int(agreements[i]),
            })

    def reduce(items):
        if not items:
            return {"count": 0, "mean_kl": None, "mean_total_variation": None,
                    "argmax_agreement": None}
        return {
            "count": len(items),
            "mean_kl": sum(item["kl"] for item in items) / len(items),
            "mean_total_variation": (
                sum(item["total_variation"] for item in items) / len(items)
            ),
            "argmax_agreement": (
                sum(item["argmax_agreement"] for item in items) / len(items)
            ),
        }

    result = {
        "overall": reduce(collected),
        "by_budget": {},
        "by_label": {},
        "by_teacher_class": {},
    }
    for budget in (0,) + EDIT_BUDGETS:
        result["by_budget"][str(budget)] = reduce(
            [item for item in collected if item["budget"] == budget]
        )
    for label, class_name in enumerate(CLASS_NAMES):
        result["by_label"][class_name] = {
            str(budget): reduce([
                item for item in collected
                if item["label"] == label and item["budget"] == budget
            ])
            for budget in (0,) + EDIT_BUDGETS
        }
        result["by_teacher_class"][class_name] = {
            str(budget): reduce([
                item for item in collected
                if item["teacher_class"] == label and item["budget"] == budget
            ])
            for budget in (0,) + EDIT_BUDGETS
        }
    return result


def initialize_model(model_kwargs, seed):
    torch.manual_seed(seed)
    return BA3MotifLEConv(**model_kwargs)


def smoke_test(records, model_kwargs, seed, batch_size, run_dir):
    training_records = [record for record in records if record["split"] == "training"]
    validation_records = [record for record in records if record["split"] == "evaluation"]
    if not training_records or not validation_records:
        raise RuntimeError("Smoke test needs training and validation query records")

    model = initialize_model(model_kwargs, seed)
    train_loader = DataLoader(
        [as_training_data(record) for record in training_records],
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = [train_epoch(model, train_loader, optimizer) for _ in range(SMOKE_EPOCHS)]
    metrics = evaluate(model, validation_records, batch_size)
    if not math.isfinite(metrics["overall"]["mean_kl"]):
        raise RuntimeError("Smoke validation KL is non-finite")

    model.eval()
    example = as_training_data(validation_records[0])
    batch = Batch.from_data_list([example])
    with torch.no_grad():
        before = model(batch.x, batch.edge_index, batch.batch).softmax(-1)
    checkpoint = {
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_kwargs": model_kwargs,
        "seed": seed,
        "smoke_epochs": SMOKE_EPOCHS,
    }
    smoke_path = run_dir / "smoke.pt"
    torch.save(checkpoint, smoke_path)
    reloaded = BA3MotifLEConv(**model_kwargs)
    reloaded.load_state_dict(torch.load(smoke_path, map_location="cpu")["model_state_dict"])
    reloaded.eval()
    with torch.no_grad():
        after = reloaded(batch.x, batch.edge_index, batch.batch).softmax(-1)
    if not torch.allclose(before, after, atol=1e-7, rtol=1e-6):
        raise RuntimeError("Smoke checkpoint reload changed surrogate probabilities")
    return {
        "status": "passed",
        "training_graphs": len(training_records),
        "validation_graphs": len(validation_records),
        "epochs": SMOKE_EPOCHS,
        "training_kl": losses,
        "validation": metrics,
        "checkpoint_reload_match": True,
    }


def save_query_tensor(records, run_dir):
    payload = []
    for record in records:
        payload.append({
            "sample_id": record["sample_id"],
            "split": record["split"],
            "source_idx": record["source_idx"],
            "label": record["label"],
            "budget": record["budget"],
            "attempts": record["attempts"],
            "additions": record["additions"],
            "deletions": record["deletions"],
            "x": record["graph"].x.cpu(),
            "edge_index": record["graph"].edge_index.cpu(),
            "teacher_probabilities": torch.tensor(
                record["teacher_probabilities"], dtype=torch.float32
            ),
        })
    torch.save(payload, run_dir / "teacher_queries.pt")


def run(args):
    repo = Path(__file__).resolve().parents[3]
    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output_path.resolve()
    run_dir = args.run_dir.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing target checkpoint: {checkpoint_path}")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Surrogate checkpoint already exists: {output_path}")
    run_dir.mkdir(parents=True, exist_ok=False)

    source_paths = (
        "src/data/ba3motif_dataset.py",
        "src/models/ba3motif_gnn.py",
        "src/scripts/surrogate_training/train_ba3motif_surrogate.py",
    )
    manifest = {
        "command": shlex.join([
            sys.executable,
            "-m",
            "src.scripts.surrogate_training.train_ba3motif_surrogate",
            *sys.argv[1:],
        ]),
        "target_checkpoint": str(checkpoint_path),
        "target_checkpoint_sha256": sha256(checkpoint_path.read_bytes()).hexdigest(),
        "output_checkpoint": str(output_path),
        "data_sha256": DATA_SHA256,
        "source_sha256": {
            path: sha256((repo / path).read_bytes()).hexdigest()
            for path in source_paths
        },
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "networkx": nx.__version__,
        "device": "cpu",
        "query_seed": args.query_seed,
        "smoke_seed": args.smoke_seed,
        "surrogate_seed": args.surrogate_seed,
        "edit_budgets": EDIT_BUDGETS,
        "max_attempts_per_graph_budget": MAX_ATTEMPTS,
        "train_coverage_floor": TRAIN_COVERAGE_FLOOR,
        "smoke_train_per_class": SMOKE_TRAIN_PER_CLASS,
        "smoke_validation_per_class": SMOKE_VALIDATION_PER_CLASS,
        "smoke_epochs": SMOKE_EPOCHS,
        "training_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "query_limit_unique_forwards": 4250,
        "self_loops": "preserved exactly from source graph",
        "connectivity": "every edited candidate must be connected",
        "distillation_loss": "KL(target_probability || surrogate_probability), temperature=1",
    }
    write_json(run_dir / "run.json", manifest)

    dataset_splits = load_ba3motif(
        repo / "src/data/ba3motif", splits=("training", "evaluation")
    )
    records, generation, source_graphs = build_query_records(
        dataset_splits, args.query_seed
    )
    coverage = generation_summary(generation, dataset_splits)
    write_json(run_dir / "generation.json", {
        "coverage": coverage,
        "candidate_generation": generation,
    })
    low_coverage = []
    for class_name, budgets in coverage["training"].items():
        for budget, stats in budgets.items():
            if stats["coverage"] < TRAIN_COVERAGE_FLOOR:
                low_coverage.append({
                    "class": class_name,
                    "budget": int(budget),
                    "coverage": stats["coverage"],
                })
    if low_coverage:
        write_json(run_dir / "summary.json", {
            "status": "stopped_low_training_query_coverage",
            "low_coverage": low_coverage,
            "coverage": coverage,
        })
        raise RuntimeError("Training edit-query coverage fell below 95%; stopping before target queries")

    smoke_sources = select_smoke_sources(source_graphs, args.smoke_seed)
    smoke_records = [
        record for record in records
        if (record["split"], record["source_idx"]) in smoke_sources
    ]
    target_records = [
        record for record in records
        if (record["split"], record["source_idx"]) not in smoke_sources
    ]
    oracle = ProbabilityOracle(checkpoint_path)
    query_path = run_dir / "queries.jsonl"
    with query_path.open("x") as query_output:
        try:
            for record in smoke_records:
                record["teacher_probabilities"] = oracle.query(record["graph"])
                if oracle.forwards > 4250:
                    raise RuntimeError("Unique target query limit exceeded")
                write_query(record, query_output)
            try:
                smoke_metrics = smoke_test(
                    smoke_records,
                    oracle.model_kwargs,
                    args.surrogate_seed,
                    args.batch_size,
                    run_dir,
                )
            except Exception as error:
                smoke_metrics = {"status": "failed", "error": repr(error)}
                write_json(run_dir / "smoke.json", smoke_metrics)
                raise
            write_json(run_dir / "smoke.json", smoke_metrics)

            for query_number, record in enumerate(target_records, start=1):
                record["teacher_probabilities"] = oracle.query(record["graph"])
                if oracle.forwards > 4250:
                    raise RuntimeError("Unique target query limit exceeded")
                write_query(record, query_output)
                if query_number % 500 == 0 or query_number == len(target_records):
                    print(
                        f"Queried target probabilities for {query_number}/{len(target_records)} remaining graphs",
                        flush=True,
                    )
        except Exception:
            write_json(run_dir / "query_state.json", {
                "requests": oracle.requests,
                "unique_forwards": oracle.forwards,
                "cache_hits": oracle.requests - oracle.forwards,
                "query_limit": 4250,
            })
            raise

    all_records = sorted(records, key=lambda record: (
        0 if record["split"] == "training" else 1,
        record["source_idx"],
        record["budget"],
    ))
    training_records = [record for record in all_records if record["split"] == "training"]
    validation_records = [record for record in all_records if record["split"] == "evaluation"]
    save_query_tensor(all_records, run_dir)
    if oracle.forwards > 4250:
        raise RuntimeError("Unique target query limit exceeded")

    model = initialize_model(oracle.model_kwargs, args.surrogate_seed)
    train_loader = DataLoader(
        [as_training_data(record) for record in training_records],
        batch_size=args.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_kl = float("inf")
    best_epoch = None
    training_log = run_dir / "training.jsonl"
    with training_log.open("x") as log:
        for epoch in range(1, args.epochs + 1):
            train_kl = train_epoch(model, train_loader, optimizer)
            validation_metrics = evaluate(model, validation_records, args.batch_size)
            validation_kl = validation_metrics["overall"]["mean_kl"]
            row = {
                "epoch": epoch,
                "training_kl": train_kl,
                "validation": validation_metrics,
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(
                    f"Epoch {epoch:03d}  train KL {train_kl:.6f}  "
                    f"validation KL {validation_kl:.6f}",
                    flush=True,
                )
            if validation_kl < best_kl:
                best_kl = validation_kl
                best_epoch = epoch
                checkpoint = {
                    "model_state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "model_kwargs": oracle.model_kwargs,
                    "class_names": CLASS_NAMES,
                    "data_sha256": DATA_SHA256,
                    "target_checkpoint_sha256": manifest["target_checkpoint_sha256"],
                    "seed": args.surrogate_seed,
                    "epoch": epoch,
                    "validation": validation_metrics,
                }
                temporary_path = run_dir / "best.tmp"
                torch.save(checkpoint, temporary_path)
                temporary_path.replace(run_dir / "best.pt")

    if best_epoch is None:
        raise RuntimeError("No finite validation-selected surrogate checkpoint was saved")
    best_payload = torch.load(run_dir / "best.pt", map_location="cpu")
    final_model = BA3MotifLEConv(**best_payload["model_kwargs"])
    final_model.load_state_dict(best_payload["model_state_dict"])
    final_metrics = evaluate(final_model, validation_records, args.batch_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(output_path.parent)) as temp_dir:
        staged = Path(temp_dir) / "surrogate.pt"
        torch.save(best_payload, staged)
        try:
            os.link(staged, output_path)
        except FileExistsError:
            raise FileExistsError(f"Surrogate checkpoint already exists: {output_path}")

    summary = {
        "status": "completed",
        "best_epoch": best_epoch,
        "coverage": coverage,
        "train_query_records": len(training_records),
        "validation_query_records": len(validation_records),
        "oracle_requests": oracle.requests,
        "unique_target_forwards": oracle.forwards,
        "oracle_cache_hits": oracle.requests - oracle.forwards,
        "query_limit": 4250,
        "query_limit_respected": oracle.forwards <= 4250,
        "smoke": smoke_metrics,
        "validation": final_metrics,
        "target_checkpoint": str(checkpoint_path),
        "target_checkpoint_sha256": manifest["target_checkpoint_sha256"],
        "surrogate_checkpoint": str(output_path),
        "surrogate_checkpoint_sha256": sha256(output_path.read_bytes()).hexdigest(),
    }
    if oracle.forwards > 4250:
        raise RuntimeError("Unique target query limit exceeded")
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "query_state.json", {
        "requests": oracle.requests,
        "unique_forwards": oracle.forwards,
        "cache_hits": oracle.requests - oracle.forwards,
        "query_limit": 4250,
    })
    print(json.dumps(summary, indent=2))


def main():
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=repo / "src/params/gnns/ba3motif/ba3motif_leconv.pt")
    parser.add_argument("--run-dir", type=Path, default=repo / DEFAULT_RUN_DIR)
    parser.add_argument("--output-path", type=Path, default=repo / DEFAULT_OUTPUT)
    parser.add_argument("--query-seed", type=int, default=42)
    parser.add_argument("--smoke-seed", type=int, default=42042)
    parser.add_argument("--surrogate-seed", type=int, default=142)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch size and learning rate must be positive")
    run(args)


if __name__ == "__main__":
    main()
