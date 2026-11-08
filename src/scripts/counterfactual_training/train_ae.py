"""Train the sole structural Graph AE B under a separately approved plan."""

import argparse
import json
import os
from pathlib import Path

# PyG imports can enumerate CUDA devices, so choose ordering before importing it.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from src.counterfactuals.training import train_autoencoder
from src.data.ba3motif_dataset import load_originals, select_smoke_data, select_validation
from src.experiments.runtime import ROOT, ExperimentRun, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "full"))
    parser.add_argument("--config", type=Path, default=ROOT / "src/configs/ba3motif_ae.json")
    args = parser.parse_args()
    args.config = (ROOT / args.config).resolve()
    config = json.loads(args.config.read_text())
    with ExperimentRun(config, args.config, args.phase, "ae") as run:
        data, _ = load_originals(run.paths)
        train, validation = data["training"], select_validation(data, run.paths, config)
        settings = config["stages"]["ae"]
        epochs = settings["epochs"]
        if args.phase == "smoke":
            count, epochs = config["smoke"]["graphs_per_class"], config["smoke"]["epochs"]
            train = select_smoke_data(train, count, settings["seed"])
            validation = select_smoke_data(validation, count, settings["seed"])
        if settings["selection_metric"] != "soft_bce":
            raise ValueError("The B-only AE stage selects by soft validation BCE")
        write_json(run.directory / "samples.json", {
            "training_ids": train["ids"].tolist(),
            "validation_ids": validation["ids"].tolist(),
            "selection_metric": "soft_bce",
        })
        train = {name: value.to(run.device) for name, value in train.items()}
        validation = {name: value.to(run.device) for name, value in validation.items()}
        _, metadata = train_autoencoder(
            train, validation, architecture=config["ae"], epochs=epochs,
            batch_size=settings["batch_size"], lr=settings["lr"], seed=settings["seed"],
            ae_seed=config["fixed_auxiliary_seed"], directory=run.directory,
            log=run.log, check_time=run.check_time,
            provenance={"inputs": config["inputs"], "seeds": config["seeds"],
                        "fixed_auxiliary_seed": config["fixed_auxiliary_seed"],
                        "run_signature": run.signature},
        )
        write_json(run.directory / "summary.json", {
            "model": "Graph AE B", "training_graphs": len(train["ids"]),
            "validation_graphs": len(validation["ids"]), **metadata,
            "training_device": str(run.device),
        })
    print(run.directory)


if __name__ == "__main__":
    main()
