"""Generate continuous BA3Motif prototypes under a separately approved plan."""

import argparse
import fcntl
import json
import platform
from pathlib import Path
import shutil
import sys
from time import monotonic

import networkx as nx
import numpy as np
import torch
import torch_geometric

from src.counterfactuals.soft_boundary import (
    SeedGraph, StructureReference, boundary_stats, check_deadline, graph_hash,
    optimize_soft_boundary, proxy_probabilities,
)
from src.data.ba3motif_dataset import CLASS_NAMES, SPLITS, load_ba3motif
from src.experiments.runtime import (
    ROOT, EXPECTED_PYTHON, append_jsonl, digest, new_directory, read_jsonl,
    validate_approval, write_json,
)
from src.models.ba3motif_model_access import ProbabilityOracle, load_frozen, state_hash


CONFIG_PATH = ROOT / "src/configs/ba3motif_soft_boundary.json"
SOURCES = (
    "src/scripts/boundary_transfer/generate_soft.py",
    "src/counterfactuals/soft_boundary.py",
    "src/data/ba3motif_dataset.py",
    "src/models/ba3motif_gnn.py",
    "src/models/ba3motif_model_access.py",
    "src/experiments/runtime.py",
)


def validate_probability_vector(probabilities, description):
    values = np.asarray(probabilities, dtype=np.float64)
    if (values.shape != (3,) or not np.isfinite(values).all() or (values < 0).any()
            or abs(float(values.sum()) - 1) > 1e-5):
        raise RuntimeError(f"Expected three finite, normalized probabilities for {description}")
    return values.tolist()


def load_sources(config):
    """Read the audited existing cache without creating replacement datasets."""
    root = ROOT / config["dataset_root"]
    if digest(root / "raw/graphs.npz") != config["data_sha256"]:
        raise RuntimeError("BA3Motif raw data hash mismatch")
    for split in SPLITS:
        cache = root / "processed" / f"{split}.pt"
        if not cache.is_file():
            raise FileNotFoundError(f"Existing processed cache required: {cache}")
    training_hash = digest(root / "processed/training.pt")
    if training_hash != config["training_sha256"]:
        raise RuntimeError("Training cache hash mismatch")
    graphs = load_ba3motif(root, splits=("training",))["training"]
    sources = {graph.parent_id: graph for graph in map(SeedGraph.from_data, graphs)}
    if len(sources) != 700:
        raise RuntimeError(f"Expected 700 training graphs, found {len(sources)}")
    return sources


def counted_forwards(roots):
    return sum(
        row["event"] == "forward_started"
        for root in roots for path in (ROOT / root).glob("*/queries.jsonl")
        for row in read_jsonl(path)
    )


def output_root(config):
    """Keep all previous experiments read-only, including their smoke artifacts."""
    root = (ROOT / config["results_root"]).resolve()
    histories = [(ROOT / name).resolve() for name in config["historical_results"]]
    if len(histories) != len(set(histories)):
        raise ValueError("Historical results must not be counted twice")
    for historical in histories:
        if root == historical or root in historical.parents or historical in root.parents:
            raise ValueError("A new soft experiment needs an independent results root")
    return root


class QueryBudget:
    """Count target forwards against this run and the read-only historical ledger."""

    def __init__(self, config, path, oracle, deadline):
        self.config = config
        self.path = path
        self.oracle = oracle
        self.deadline = deadline
        self.previous = counted_forwards([config["results_root"]])
        self.historical = counted_forwards(config["historical_results"])
        self.used = 0

    def query(self, source, adjacency, role):
        check_deadline(self.deadline)
        if (self.previous + self.used >= self.config["max_new_target_forwards"]
                or self.historical + self.previous + self.used
                >= self.config["total_target_forward_limit"]):
            raise RuntimeError("Target forward budget exhausted")
        row = {"role": role, "source_id": source.parent_id,
               "graph_hash": graph_hash(source.features, adjacency),
               "input": "binary" if adjacency.dtype == torch.bool else "weighted"}
        self.used += 1
        append_jsonl(self.path, {**row, "event": "forward_started"})
        probabilities = validate_probability_vector(
            self.oracle.probabilities(source.features, adjacency), role,
        )
        append_jsonl(self.path, {**row, "event": "result", "probabilities": probabilities})
        return probabilities


@torch.no_grad()
def check_interfaces(source, proxy, budget, tolerance):
    dense_proxy = validate_probability_vector(
        proxy_probabilities(proxy, source, weighted=True), "weighted proxy interface",
    )
    sparse_proxy = validate_probability_vector(
        proxy_probabilities(proxy, source, weighted=False), "binary proxy interface",
    )
    hard_target = budget.query(source, source.adjacency, "smoke_interface")
    soft_target = budget.query(source, source.adjacency.float(), "smoke_interface")
    differences = {
        "proxy": max(abs(a - b) for a, b in zip(dense_proxy, sparse_proxy)),
        "target": max(abs(a - b) for a, b in zip(hard_target, soft_target)),
    }
    if not all(np.isfinite(d) and d <= tolerance for d in differences.values()):
        raise RuntimeError(f"Binary and weighted interfaces differ for {source.parent_id}: {differences}")
    return {"source_id": source.parent_id, "max_absolute_probability_difference": differences}


def summarize(rows, pairs):
    groups = []
    for pair in pairs:
        selected = [row for row in rows if row["pair"] == pair]
        soft = [row for row in selected if row["fractional_edges"] > 0]
        proxy_pass = sum(row["proxy_boundary"]["passed"] for row in soft)
        joint_pass = sum(row["joint_pass"] for row in soft)
        groups.append({
            "pair": [CLASS_NAMES[k] for k in pair], "planned_slots": len(selected),
            "soft_count": len(soft), "proxy_pass": proxy_pass,
            "target_pass": sum(row["target_boundary"]["passed"] for row in soft),
            "joint_pass": joint_pass,
            "target_pass_given_proxy": joint_pass / proxy_pass if proxy_pass else None,
        })
    return groups


def find_smoke(root, signature, smoke_ids):
    for path in sorted(root.glob("smoke-*/summary.json"), reverse=True):
        summary = json.loads(path.read_text())
        metadata = json.loads((path.parent / "run.json").read_text())
        if not summary.get("complete") or metadata["signature"] != signature:
            continue
        rows = read_jsonl(path.parent / "candidates.jsonl")
        if len(rows) == len(smoke_ids) and {row["id"] for row in rows} == smoke_ids:
            return path.parent, rows
    raise RuntimeError("Full requires a complete smoke with this approved plan, config, and code")


def execute(config, phase, directory, smoke_directory, smoke_rows, jobs):
    started = monotonic()
    deadline = started + config[f"{phase}_seconds"]
    sources = load_sources(config)
    for role in ("target", "surrogate"):
        if digest(ROOT / config[f"{role}_checkpoint"]) != config[f"{role}_sha256"]:
            raise RuntimeError(f"{role} checkpoint hash mismatch")

    proxy = load_frozen(ROOT / config["surrogate_checkpoint"])
    initial_proxy = state_hash(proxy)
    oracle = ProbabilityOracle(ROOT / config["target_checkpoint"])
    budget = QueryBudget(config, directory / "queries.jsonl", oracle, deadline)
    if budget.historical < config["minimum_historical_target_forwards"]:
        raise RuntimeError("Historical query records are incomplete")
    reference = StructureReference(list(sources.values()))
    write_json(directory / "reference_stats.json", reference.record())
    graph_dir = directory / "graphs"
    graph_dir.mkdir()
    observations = directory / "candidates.jsonl"
    interface_checks = []
    rows = []
    for row in smoke_rows:
        artifact = smoke_directory / row["graph_path"]
        if digest(artifact) != row["file_sha256"]:
            raise RuntimeError(f"Smoke artifact changed: {artifact}")
        shutil.copy2(artifact, directory / row["graph_path"])
        row = {**row, "reused_from": smoke_directory.name}
        append_jsonl(observations, row)
        rows.append(row)
    seen = {row["id"] for row in rows}
    required = len(jobs) - len(seen) + (2 * len(jobs) if phase == "smoke" else 0)
    if (budget.previous + required > config["max_new_target_forwards"]
            or budget.historical + budget.previous + required
            > config["total_target_forward_limit"]):
        raise RuntimeError("Insufficient target budget to complete this phase")

    if phase == "smoke":
        for _, _, source_id in jobs:
            interface_checks.append(
                check_interfaces(sources[source_id], proxy, budget, config["interface_tolerance"])
            )
        write_json(directory / "interface_checks.json", interface_checks)

    for identifier, pair, source_id in jobs:
        if identifier in seen:
            continue
        check_deadline(deadline)
        source = sources[source_id]
        trace = directory / f"{identifier}.jsonl"
        adjacency, embedding, selected = optimize_soft_boundary(
            source, proxy, reference, pair, config, deadline, lambda row: append_jsonl(trace, row),
        )
        if not torch.equal(adjacency.diag(), source.adjacency.diag().float()):
            raise RuntimeError("Candidate self-loops differ from its source")
        target_probs = budget.query(source, adjacency, identifier)
        target_boundary = boundary_stats(target_probs, pair, config["probability_gap"],
                                         config["third_class_margin"])
        artifact = graph_dir / f"{identifier}.pt"
        torch.save({"adjacency": adjacency, "x": source.features, "proxy_embedding": embedding,
                    "source_id": source_id, "pair": pair}, artifact)
        row = {
            **selected, "id": identifier, "pair": pair, "source_id": source_id,
            "target_probs": target_probs, "target_boundary": target_boundary,
            "joint_pass": (selected["fractional_edges"] > 0 and selected["proxy_boundary"]["passed"]
                           and target_boundary["passed"]),
            "probability_tv": sum(abs(a - b) for a, b in zip(selected["proxy_probs"], target_probs)) / 2,
            "graph_path": str(artifact.relative_to(directory)), "file_sha256": digest(artifact),
        }
        append_jsonl(observations, row)
        rows.append(row)
        print(f"{identifier}: proxy={selected['proxy_probs']} target={target_probs}", flush=True)

    oracle.check_frozen()
    if state_hash(proxy) != initial_proxy or any(p.grad is not None for p in proxy.parameters()):
        raise RuntimeError("Surrogate model did not remain frozen")
    check_deadline(deadline)
    write_json(directory / "summary.json", {
        "complete": True, "phase": phase, "groups": summarize(rows, config["pairs"]),
        "interface_checks": interface_checks,
        "reused_from": str(smoke_directory) if smoke_directory else None,
        "target_forwards_this_run": budget.used,
        "target_forwards_this_experiment": budget.previous + budget.used,
        "target_forwards_cumulative": budget.historical + budget.previous + budget.used,
        "seconds": monotonic() - started,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "full"))
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError(f"Use {EXPECTED_PYTHON}")
    config_path = (ROOT / args.config).resolve()
    config = json.loads(config_path.read_text())
    # Historical approval documents the old implementation only. This gate
    # precedes model/data loading and every result-directory mutation.
    plan = validate_approval(config, phase=args.phase)
    root = output_root(config)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(config["seed"])
    signature = {
        "entrypoint": "src.scripts.boundary_transfer.generate_soft",
        "config": digest(config_path), "plan": digest(plan),
        "code": {path: digest(ROOT / path) for path in SOURCES},
    }
    jobs = [(f"p{i}-r{repeat}", pair, source_id)
            for i, pair in enumerate(config["pairs"])
            for repeat, source_id in enumerate(config["parents"][i])]
    smoke_ids = {f"p{i}-r0" for i in range(len(config["pairs"]))}
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".approval.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == "full":
            smoke_directory, smoke_rows = find_smoke(root, signature, smoke_ids)
        else:
            smoke_directory, smoke_rows = None, []
            jobs = [job for job in jobs if job[0] in smoke_ids]
        directory = new_directory(root, args.phase)
        shutil.copy2(plan, directory / "approved_plan.md")
        write_json(directory / "run.json", {
            "phase": args.phase, "signature": signature, "config": config,
            "command": [sys.executable, "-m", "src.scripts.boundary_transfer.generate_soft",
                        args.phase, "--config", str(config_path)],
            "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "numpy": np.__version__, "networkx": nx.__version__, "pyg": torch_geometric.__version__,
        })
        try:
            execute(config, args.phase, directory, smoke_directory, smoke_rows, jobs)
        except BaseException as exc:
            write_json(directory / "failure.json", {"error": repr(exc)})
            raise
        print(f"{args.phase} complete: {directory}", flush=True)


if __name__ == "__main__":
    main()
