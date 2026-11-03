"""BA3Motif C5/House/K5 dataset with the myexplainer raw/processed pipeline."""

from hashlib import sha256
from pathlib import Path

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
