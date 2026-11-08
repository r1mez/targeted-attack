"""Train a fresh flow jointly with CFM, soft target CE and soft graph distance."""

import argparse
import json
import os
from pathlib import Path

# PyG imports can enumerate CUDA devices, so choose ordering before importing it.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from src.counterfactuals.soft_counterfactual import load_ae
from src.counterfactuals.training import expand_target_tasks, train_flow
from src.data.ba3motif_dataset import load_originals, select_smoke_data, select_validation
from src.experiments.runtime import ROOT, ExperimentRun, write_json
from src.models.ba3motif_model_access import DenseProxy, load_frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "full"))
    parser.add_argument("--config", type=Path, default=ROOT / "src/configs/ba3motif_flow.json")
    args = parser.parse_args()
    args.config = (ROOT / args.config).resolve()
    config = json.loads(args.config.read_text())
    with ExperimentRun(config, args.config, args.phase, "train") as run:
        data, _ = load_originals(run.paths)
        ae, ae_metadata = load_ae(config, run.paths, training_ids=data["training"]["ids"])
        train, validation = data["training"], select_validation(data, run.paths, config)
        settings = config["stages"]["train"]
        epochs = settings["epochs"]
        if args.phase == "smoke":
            count, epochs = config["smoke"]["graphs_per_class"], config["smoke"]["epochs"]
            train = select_smoke_data(train, count, settings["seed"])
            validation = select_smoke_data(validation, count, settings["seed"])
        manifest = {"training_ids": train["ids"].tolist(),
                    "validation_ids": validation["ids"].tolist(),
                    "frozen_autoencoder": ae_metadata, "flow_initialization": "random"}
        for name, original in (("training_tasks", train), ("validation_tasks", validation)):
            tasks = expand_target_tasks(original, settings["target_policy"],
                                        num_classes=config["flow"]["classes"])
            manifest[name] = [
                {"source_id": int(i), "source": int(s), "target": int(t)}
                for i, s, t in zip(tasks["ids"], tasks["classes"], tasks["targets"])
            ]
        write_json(run.directory / "samples.json", manifest)
        train = {name: value.to(run.device) for name, value in train.items()}
        validation = {name: value.to(run.device) for name, value in validation.items()}
        ae = ae.to(run.device)
        proxy = DenseProxy(load_frozen(run.paths["proxy"]).to(run.device), train["x"].size(1))
        _, report = train_flow(
            ae, proxy, train, validation, architecture=config["flow"],
            target_policy=settings["target_policy"], epochs=epochs,
            batch_size=settings["batch_size"], lr=settings["lr"], seed=settings["seed"],
            ae_seed=config["fixed_auxiliary_seed"], std_floor=settings["std_floor"],
            steps=settings["steps"], multiplier=settings["multiplier"],
            lambda_cf=settings["lambda_cf"], lambda_dist=settings["lambda_dist"],
            directory=run.directory, log=run.log, check_time=run.check_time,
            provenance={"inputs": config["inputs"], "seeds": config["seeds"],
                        "fixed_auxiliary_seed": config["fixed_auxiliary_seed"],
                        "run_signature": run.signature},
        )
        write_json(run.directory / "summary.json", {
            "training_graphs": len(train["ids"]), "validation_graphs": len(validation["ids"]),
            "objective": "L_CFM + lambda_cf*L_cf + lambda_dist*L_dist",
            "L_boundary": 0, "trainable": "shared flow only",
            "training_device": str(run.device),
            "target_gnn_forwards_during_training": 0, **report,
        })
    print(run.directory)


if __name__ == "__main__":
    main()
