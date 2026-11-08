"""BA3Motif C5/House/K5 dataset with the myexplainer raw/processed pipeline."""

from hashlib import sha256
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset


DATA_SHA256 = "a79863418159ec67e11d2c0283323d2a68a4d70814c59aba38b3d7cc1533d3b0"
CLASS_NAMES = ("C5", "House", "K5")
SPLITS = ("training", "evaluation", "testing")


class BA3Motif(InMemoryDataset):
    """Audited three-class BA3Motif data with cached PyG split files."""

    splits = SPLITS
    num_classes = 3
    raw_sha256 = DATA_SHA256

    def __init__(self, root, mode="training", transform=None,
                 pre_transform=None, pre_filter=None):
        if mode not in self.splits:
            raise ValueError("Unknown BA3Motif split: {}".format(mode))

        raw_path = Path(root) / "raw" / "graphs.npz"
        if raw_path.exists():
            digest = sha256(raw_path.read_bytes()).hexdigest()
            if digest != self.raw_sha256:
                raise ValueError("BA3Motif raw NPZ differs from the audited version")

        self.mode = mode
        super().__init__(root, transform, pre_transform, pre_filter)
        split_index = self.splits.index(mode)
        try:
            self.data, self.slices = torch.load(
                self.processed_paths[split_index],
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:  # PyTorch < 2.0 does not accept weights_only.
            self.data, self.slices = torch.load(
                self.processed_paths[split_index], map_location="cpu"
            )

    @property
    def raw_file_names(self):
        return ["graphs.npz"]

    @property
    def processed_file_names(self):
        return ["{}.pt".format(split) for split in self.splits]

    def download(self):
        raise FileNotFoundError(
            "Missing {}. Place the audited BA3Motif NPZ there (SHA-256: {})."
            .format(Path(self.raw_dir) / "graphs.npz", self.raw_sha256)
        )

    def process(self):
        with np.load(Path(self.raw_dir) / "graphs.npz", allow_pickle=False) as source:
            adjacency = source["adjacency"]
            features = source["x"]
            labels = source["y"]
            split_ids = source["split"]
            parent_ids = source["parent_id"]

        if (
            adjacency.shape != (1000, 25, 25)
            or features.shape != (1000, 25, 10)
            or labels.shape != (1000,)
            or split_ids.shape != (1000,)
            or parent_ids.shape != (1000,)
        ):
            raise ValueError("Invalid BA3Motif NPZ field shapes")
        if not np.all((labels >= 0) & (labels < self.num_classes)):
            raise ValueError("Invalid BA3Motif class labels")
        if not np.all((split_ids >= 0) & (split_ids < len(self.splits))):
            raise ValueError("Invalid BA3Motif split identifiers")
        if not np.array_equal(adjacency, adjacency.transpose(0, 2, 1)):
            raise ValueError("BA3Motif adjacency matrices must be symmetric")
        if not np.array_equal(np.bincount(labels, minlength=3), [334, 334, 332]):
            raise ValueError("Unexpected BA3Motif class counts")
        if not np.array_equal(parent_ids, np.arange(1000)):
            raise ValueError("Parent IDs must preserve audited source order")
        if not np.array_equal(np.bincount(split_ids, minlength=3), [700, 150, 150]):
            raise ValueError("Unexpected fixed BA3Motif split sizes")

        groups = [[] for _ in self.splits]
        for graph_index in range(1000):
            edge_index = torch.from_numpy(
                np.stack(np.where(adjacency[graph_index] != 0)).astype(np.int64)
            )
            source, target = edge_index
            motif_edge_mask = (
                (source >= 20) & (target >= 20) & (source != target)
            )
            graph = Data(
                x=torch.from_numpy(features[graph_index].astype(np.float32, copy=True)),
                edge_index=edge_index,
                edge_attr=torch.ones((edge_index.size(1), 1), dtype=torch.float32),
                y=torch.tensor(int(labels[graph_index]), dtype=torch.long),
                motif_edge_mask=motif_edge_mask,
                idx=torch.tensor(int(parent_ids[graph_index]), dtype=torch.long),
                name="BA-3motif{}".format(graph_index),
            )
            if self.pre_filter is not None and not self.pre_filter(graph):
                continue
            if self.pre_transform is not None:
                graph = self.pre_transform(graph)
            groups[int(split_ids[graph_index])].append(graph)

        if any(not group for group in groups):
            raise ValueError("All BA3Motif splits must be nonempty")
        for group, path in zip(groups, self.processed_paths):
            torch.save(self.collate(group), path)


def load_ba3motif(root, splits=SPLITS):
    """Return graphs by fixed split, building processed caches on first use."""
    unknown_splits = set(splits) - set(SPLITS)
    if unknown_splits:
        raise ValueError("Unknown BA3Motif splits: {}".format(sorted(unknown_splits)))
    return {
        split: list(BA3Motif(root, mode=split))
        for split in splits
    }


def graph_key(features, adjacency):
    """Identify the exact query, including each tensor's dtype and shape.

    A float32 soft graph and a bool graph have different identities even when
    their values coincide. Zero-weight edges are retained in soft graph keys.
    """
    state = sha256()
    for tensor in (features, adjacency):
        tensor = tensor.detach().cpu().contiguous()
        state.update(json.dumps([str(tensor.dtype), list(tensor.shape)]).encode())
        state.update(b"\0")
        state.update(tensor.numpy().tobytes())
    return state.hexdigest()


def load_graph_split(path, split):
    """Read one audited split in parent-ID order, without assigning teacher classes.

    Return tensors ``ids[N]``, ``x[N,25,10]``, ``adjacency[N,25,25]`` and
    ``labels[N]``. The labels are ground truth, never classifier predictions.
    This does not build or alter the existing PyG processed caches.
    """
    if split not in SPLITS:
        raise ValueError(f"Unknown BA3Motif split: {split}")
    path = Path(path)
    if sha256(path.read_bytes()).hexdigest() != DATA_SHA256:
        raise RuntimeError("BA3Motif raw NPZ differs from the audited version")
    with np.load(path, allow_pickle=False) as raw:
        adjacency, x, labels = raw["adjacency"], raw["x"], raw["y"]
        split_ids, parent_ids = raw["split"], raw["parent_id"]
        if (adjacency.shape != (1000, 25, 25) or x.shape != (1000, 25, 10)
                or labels.shape != (1000,) or split_ids.shape != (1000,)
                or parent_ids.shape != (1000,)
                or not np.array_equal(parent_ids, np.arange(1000))
                or not np.array_equal(np.bincount(split_ids, minlength=3), [700, 150, 150])
                or not np.array_equal(np.bincount(labels, minlength=3), [334, 334, 332])
                or not np.isin(adjacency, [0, 1]).all()
                or not np.isfinite(x).all()
                or not np.array_equal(adjacency, adjacency.transpose(0, 2, 1))):
            raise RuntimeError("Invalid audited BA3Motif arrays or fixed splits")
        selected = np.flatnonzero(split_ids == SPLITS.index(split))
        selected = selected[np.argsort(parent_ids[selected], kind="stable")]
        expected = (700, 150, 150)[SPLITS.index(split)]
        if len(selected) != expected:
            raise RuntimeError(f"Unexpected number of {split} graphs")
        return {
            "ids": torch.from_numpy(parent_ids[selected].astype(np.int64, copy=True)),
            "x": torch.from_numpy(x[selected].astype(np.float32, copy=True)),
            "adjacency": torch.from_numpy(adjacency[selected].astype(bool, copy=True)),
            "labels": torch.from_numpy(labels[selected].astype(np.int64, copy=True)),
        }


def load_originals(paths):
    """Match train/evaluation originals to the recorded target probability cache.

    ``paths`` contains ``data``, ``cache`` and ``target`` file paths, normally
    verified against the experiment configuration before this call. The cache's
    sibling ``run.json`` must fingerprint the exact data and target checkpoint.
    Return ``(data, cache)``: data has ``training`` and ``evaluation`` splits,
    extending :func:`load_graph_split` with float32 ``probabilities[N,3]`` and
    int64 ``classes[N]`` from teacher argmax. The cache maps :func:`graph_key`
    to original probability lists and never treats true labels as predictions.
    """
    paths = {name: Path(paths[name]) for name in ("data", "cache", "target")}
    metadata_path = paths["cache"].with_name("run.json")
    metadata = json.loads(metadata_path.read_text())
    if (metadata.get("data_sha256") != sha256(paths["data"].read_bytes()).hexdigest()
            or metadata.get("target_checkpoint_sha256")
            != sha256(paths["target"].read_bytes()).hexdigest()):
        raise RuntimeError("Original probability cache provenance mismatch")

    originals = {}
    with paths["cache"].open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["budget"] != 0:
                continue
            key = row["split"], int(row["source_idx"])
            probabilities = row["teacher_probabilities"]
            if (key in originals or len(probabilities) != len(CLASS_NAMES)
                    or not all(isinstance(p, (int, float)) and not isinstance(p, bool)
                               and math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
                    or abs(sum(probabilities) - 1) > 1e-5):
                raise RuntimeError(f"Invalid original probability record for {key}")
            originals[key] = row

    data, cache, matched = {}, {}, set()
    for split in ("training", "evaluation"):
        selected = load_graph_split(paths["data"], split)
        probabilities = []
        for i, graph_id in enumerate(selected["ids"].tolist()):
            key = split, graph_id
            if key not in originals:
                raise RuntimeError(f"Missing cached original probability for {key}")
            row = originals[key]
            if row["label"] != int(selected["labels"][i]):
                raise RuntimeError(f"Cached true label mismatch for {key}")
            prob = list(row["teacher_probabilities"])
            cache_key = graph_key(selected["x"][i], selected["adjacency"][i])
            if cache_key in cache and cache[cache_key] != prob:
                raise RuntimeError(f"Conflicting cached probabilities for {key}")
            cache[cache_key] = prob
            probabilities.append(prob)
            matched.add(key)
        selected["probabilities"] = torch.tensor(probabilities, dtype=torch.float32)
        selected["classes"] = selected["probabilities"].argmax(dim=-1)
        data[split] = selected
    if matched != set(originals):
        raise RuntimeError("Original cache records do not match train/evaluation splits")
    return data, cache


def select_validation(data, paths, config):
    """Use the explicitly selected full split or historical 90 AE development IDs."""
    selection = config["data_selection"]
    policy = selection.get("validation_policy", "ae_selection")
    if policy == "full_split":
        return data["evaluation"]
    if policy != "ae_selection":
        raise ValueError(f"Unknown validation policy: {policy}")
    manifest = json.loads(Path(paths[selection["validation_manifest_input"]]).read_text())
    ids = manifest[selection["validation_ids_key"]]
    if (not isinstance(ids, list) or len(ids) != 90 or len(set(ids)) != 90
            or any(not isinstance(i, int) or isinstance(i, bool) for i in ids)
            or set(ids).intersection(data["training"]["ids"].tolist())):
        raise RuntimeError("Expected 90 distinct recorded validation IDs outside training")
    evaluation = data["evaluation"]
    positions = {graph_id: i for i, graph_id in enumerate(evaluation["ids"].tolist())}
    if not set(ids).issubset(positions):
        raise RuntimeError("Recorded validation IDs do not belong to the evaluation split")
    indices = torch.tensor([positions[graph_id] for graph_id in ids], dtype=torch.long)
    return {name: values[indices] for name, values in evaluation.items()}


def select_smoke_data(data, per_class, seed):
    """Select a deterministic small subset per existing teacher-predicted class.

    This does not query a model, assign classes, or filter by true labels. Every
    field keeps the same selected rows, ordered by class and seeded sampling.
    """
    if not isinstance(per_class, int) or isinstance(per_class, bool) or per_class <= 0:
        raise ValueError("Smoke graphs per class must be an explicit positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("Smoke selection requires an explicit integer seed")
    if "classes" not in data or "ids" not in data:
        raise ValueError("Smoke selection requires existing classes and graph IDs")
    classes, ids = data["classes"], data["ids"]
    if (classes.ndim != 1 or classes.dtype != torch.long or ids.shape != classes.shape
            or not ((classes >= 0) & (classes < len(CLASS_NAMES))).all()
            or any(not isinstance(values, torch.Tensor) or values.ndim == 0
                   or values.size(0) != classes.size(0) for values in data.values())):
        raise ValueError("Smoke data must contain aligned tensors and valid existing classes")
    selected = []
    for class_index in range(len(CLASS_NAMES)):
        candidates = [i for i, value in enumerate(classes.tolist()) if value == class_index]
        candidates.sort(key=lambda i: int(ids[i]))
        if len(candidates) < per_class:
            raise RuntimeError(f"Not enough smoke graphs in existing class {class_index}")
        random.Random(seed + class_index).shuffle(candidates)
        selected.extend(candidates[:per_class])
    indices = torch.tensor(selected, dtype=torch.long)
    return {name: values[indices] for name, values in data.items()}
