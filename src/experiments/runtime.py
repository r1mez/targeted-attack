"""Shared approval, provenance, resource limits and exact probability-query records."""

from datetime import date, datetime, timezone
import fcntl
from hashlib import sha256
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shlex
import shutil
import signal
import subprocess
import sys
import threading
from time import monotonic
import traceback

import numpy as np
import torch

from src.data.ba3motif_dataset import CLASS_NAMES, graph_key


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PYTHON = Path("/home/ll_yqs2/data/anaconda3/envs/myexplainer/bin/python")


def training_device(config):
    """Resolve an explicitly selected local GPU; never fall back to CPU."""
    name = config.get("device")
    if not isinstance(name, str) or not name:
        raise RuntimeError("训练配置必须显式指定本机 CUDA GPU，例如 device='cuda:1'")
    device = torch.device(name)
    if device.type != "cuda":
        raise RuntimeError("模型训练必须使用本机 GPU；禁止自动或默认使用 CPU")
    order = config.get("cuda_device_order", "PCI_BUS_ID")
    if order != "PCI_BUS_ID":
        raise RuntimeError("Local GPU indices must use CUDA_DEVICE_ORDER=PCI_BUS_ID")
    os.environ["CUDA_DEVICE_ORDER"] = order
    # Set before CUDA model operations for the pinned PyTorch deterministic mode.
    workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if workspace not in (":4096:8", ":16:8"):
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8")
    if not torch.cuda.is_available():
        raise RuntimeError("本机 CUDA 不可用；停止训练，不回退 CPU")
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index < 0 or index >= torch.cuda.device_count():
        raise RuntimeError(f"Requested CUDA device does not exist: {name}")
    device = torch.device("cuda", index)
    torch.cuda.set_device(device)
    expected = config.get("gpu_name")
    if expected and torch.cuda.get_device_name(device) != expected:
        raise RuntimeError(f"GPU mapping mismatch: {device} is not {expected}; set CUDA_DEVICE_ORDER=PCI_BUS_ID before Python starts")
    return device


def digest(path):
    state = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def append_jsonl(path, row):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def new_directory(root, prefix):
    """Atomically allocate a new numbered directory without replacing old output."""
    root = Path(root)
    if not prefix or Path(prefix).name != prefix or prefix in (".", ".."):
        raise ValueError("Run prefix must be a single nonempty path name")
    root.mkdir(parents=True, exist_ok=True)
    for index in range(1, 1000000):
        directory = root / f"{prefix}-{index:03d}"
        try:
            directory.mkdir()
        except FileExistsError:
            continue
        return directory
    raise RuntimeError("No available run directory")


def validate_approval(config, phase=None, stage=None):
    """Reject unapproved or incomplete scope before any experiment model is loaded."""
    if config.get("approval_status") != "已批准":
        raise RuntimeError("实验方案尚未批准；不得加载实验模型或运行实验")
    approved_date = config.get("approval_date")
    if not isinstance(approved_date, str):
        raise RuntimeError("Approved configuration requires an explicit approval_date")
    try:
        date.fromisoformat(approved_date)
    except ValueError as error:
        raise RuntimeError("approval_date must use YYYY-MM-DD") from error
    if phase is not None and phase not in config.get("approved_phases", []):
        raise RuntimeError(f"Experiment phase is not approved: {phase}")
    if stage is not None and stage not in config.get("approved_stages", []):
        raise RuntimeError(f"Experiment stage is not approved: {stage}")
    if not config.get("plan_path") or not config.get("plan_sha256"):
        raise RuntimeError("Approval requires plan_path and the approved plan_sha256")
    plan = ROOT / config["plan_path"]
    if not plan.is_file():
        raise FileNotFoundError(f"Approved plan not found: {plan}")
    if digest(plan) != config["plan_sha256"]:
        raise RuntimeError("Experiment plan differs from the approved fingerprint")
    return plan


def verified_inputs(config, stage=None):
    """Verify common inputs plus inputs needed by this operation only.

    A trained flow is an evaluation input, never an input to fresh flow training.
    """
    paths = {}
    sources = dict(config["inputs"])
    if stage is not None:
        sources.update(config.get("stages", {}).get(stage, {}).get("inputs", {}))
    for name, source in sources.items():
        if not source.get("path") or not source.get("sha256"):
            raise RuntimeError(f"Input path/fingerprint is not configured: {name}")
        path = ROOT / source["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Experiment input does not exist: {name}: {path}")
        if digest(path) != source["sha256"]:
            raise RuntimeError(f"Unexpected SHA-256 for experiment input: {name}")
        paths[name] = path
    return paths


def _probabilities(values):
    if (len(values) != len(CLASS_NAMES)
            or not all(isinstance(p, (int, float)) and not isinstance(p, bool)
                       and math.isfinite(p) and 0 <= p <= 1 for p in values)
            or abs(sum(values) - 1) > 1e-5):
        raise RuntimeError("Invalid target probability vector")
    return list(values)


class TargetLedger:
    """Cache exact hard/soft inputs and count a forward before starting a query.

    The supplied cache must belong to the same verified target checkpoint as the
    oracle. Failed target calls consume budget as attempted forwards. Soft inputs
    retain their dtype, shape and all edge weights in the cache identity.
    """

    def __init__(self, oracle, cache, path, limit, check_time):
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError("Target query limit must be a nonnegative integer")
        self.target = oracle
        self.cache = {key: _probabilities(value) for key, value in cache.items()}
        self.path = Path(path)
        self.limit = limit
        self.check_time = check_time
        self.started = 0
        self.completed = 0
        self.requests = 0
        self.cache_hits = 0

    def query(self, x, adjacency, **context):
        self.check_time()
        key = graph_key(x, adjacency)
        record = {**context, "graph_sha256": key,
                  "features_dtype": str(x.dtype), "features_shape": list(x.shape),
                  "adjacency_dtype": str(adjacency.dtype), "adjacency_shape": list(adjacency.shape)}
        self.requests += 1
        if key in self.cache:
            probabilities = list(self.cache[key])
            self.cache_hits += 1
            append_jsonl(self.path, {**record, "event": "cache_hit", "probabilities": probabilities})
            return probabilities, False
        if self.started >= self.limit:
            append_jsonl(self.path, {**record, "event": "budget_exhausted"})
            raise RuntimeError("Target forward budget exhausted")
        append_jsonl(self.path, {**record, "event": "forward_started"})
        self.started += 1
        try:
            probabilities = _probabilities(self.target.probabilities(x, adjacency))
        except BaseException as error:
            append_jsonl(self.path, {**record, "event": "forward_failed", "error": repr(error)})
            raise
        append_jsonl(self.path, {**record, "event": "forward_completed", "probabilities": probabilities})
        self.completed += 1
        self.cache[key] = probabilities
        return list(probabilities), True

    def record(self):
        return {"requests": self.requests, "new_target_forwards": self.started,
                "completed_target_forwards": self.completed, "cache_hits": self.cache_hits,
                "target_forward_limit": self.limit}


class ExperimentRun:
    """An approved run with independent output and a matching-smoke gate for full.

    Enter this context before loading models. ``config[phase]`` supplies explicit
    positive ``seconds`` and nonnegative ``queries``. A named stage selects its
    seed from ``config['stages'][stage]['seed']``, otherwise ``config['seed']`` is
    required. Callers retain the experiment implementation and write any result
    fields to ``summary.json``; exit adds execution status and provenance.
    """

    def __init__(self, config, config_path, phase, stage=None):
        self.config = config
        self.config_path = ROOT / config_path
        self.phase = phase
        self.stage = stage
        self.directory = None
        self._lock = None
        self._previous_signal = None
        self._previous_timer = None

    def __enter__(self):
        plan = validate_approval(self.config, self.phase, self.stage)
        if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
            raise RuntimeError(f"Use the project experiment interpreter {EXPECTED_PYTHON}")
        if json.loads(self.config_path.read_text()) != self.config:
            raise RuntimeError("Runtime config differs from its on-disk configuration")
        phase_config = {
            **self.config.get(self.phase, {}),
            **self.config.get("stages", {}).get(self.stage, {}).get(self.phase, {}),
        }
        seconds, queries = phase_config.get("seconds"), phase_config.get("queries")
        if (not isinstance(seconds, (int, float)) or isinstance(seconds, bool)
                or not math.isfinite(seconds) or seconds <= 0):
            raise RuntimeError(f"{self.phase}.seconds must be explicitly configured and positive")
        if not isinstance(queries, int) or isinstance(queries, bool) or queries < 0:
            raise RuntimeError(f"{self.phase}.queries must be explicitly configured and nonnegative")
        seed = self.config.get("seed")
        if self.stage is not None:
            stage_config = self.config.get("stages", {}).get(self.stage)
            if stage_config is None:
                raise RuntimeError(f"Experiment stage is not configured: {self.stage}")
            if stage_config.get("seed") is not None:
                seed = stage_config["seed"]
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2 ** 32:
            raise RuntimeError("An explicit seed in [0,2**32) is required for this stage")
        threads = self.config.get("threads")
        if not isinstance(threads, int) or isinstance(threads, bool) or threads <= 0:
            raise RuntimeError("threads must be explicitly configured and positive")
        self.device = training_device(self.config) if self.stage in ("train", "ae", "evaluate") else None
        self.paths = verified_inputs(self.config, self.stage)
        sources = {str(path.relative_to(ROOT)): digest(path)
                   for path in sorted((ROOT / "src").rglob("*.py"))}
        approval_fields = {"approval_status", "approval_date", "approved_phases", "approved_stages"}
        settings = {key: value for key, value in self.config.items() if key not in approval_fields}
        self.signature = {"stage": self.stage, "sources": sources,
                          "settings": sha256(json.dumps(settings, sort_keys=True,
                                                       allow_nan=False).encode()).hexdigest()}
        root = ROOT / self.config["results_root"]
        root.mkdir(parents=True, exist_ok=True)
        self._lock = (root / ".lock").open("a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            prefix = f"{self.stage}-{self.phase}" if self.stage else self.phase
            prior = 0
            for path in root.glob(f"{prefix}-*/queries.jsonl"):
                prior += sum(row.get("event") == "forward_started" for row in read_jsonl(path))
            self.query_limit = max(0, queries - prior)
            if self.phase == "full":
                smoke_prefix = f"{self.stage}-smoke" if self.stage else "smoke"
                complete = False
                for path in root.glob(f"{smoke_prefix}-*/summary.json"):
                    record = json.loads(path.read_text())
                    if record.get("status") == "completed" and record.get("signature") == self.signature:
                        complete = True
                        break
                if not complete:
                    raise RuntimeError("Full run requires a completed new smoke with matching stage, config and code")
            self.directory = new_directory(root, prefix)
            self.started = monotonic()
            self.deadline = self.started + seconds
            shutil.copy2(self.config_path, self.directory / "config.json")
            shutil.copy2(plan, self.directory / "plan.md")
            for source in sources:
                copied = self.directory / "source" / source
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / source, copied)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.set_num_threads(threads)
            torch.use_deterministic_algorithms(True)
            packages = {}
            for package in ("torch", "torch-geometric", "numpy", "networkx"):
                try:
                    packages[package] = metadata.version(package)
                except metadata.PackageNotFoundError:
                    packages[package] = None
            environment = {"python": sys.version, "executable": sys.executable,
                           "platform": platform.platform(), "packages": packages,
                           "torch_threads": torch.get_num_threads(),
                           "torch_interop_threads": torch.get_num_interop_threads(),
                           "deterministic_algorithms": True,
                           "cudnn_benchmark": torch.backends.cudnn.benchmark}
            if self.device is not None:
                properties = torch.cuda.get_device_properties(self.device)
                environment.update(training_device=str(self.device),
                                   gpu_name=properties.name,
                                   gpu_total_memory_bytes=properties.total_memory,
                                   cuda_runtime=torch.version.cuda,
                                   cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                                   cuda_device_order=os.environ["CUDA_DEVICE_ORDER"],
                                   cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"])
            git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                 capture_output=True, text=True, check=False)
            self.metadata = {"phase": self.phase, "stage": self.stage, "config": self.config,
                             "config_sha256": digest(self.config_path), "plan_sha256": digest(plan),
                             "signature": self.signature, "seed": seed, "seconds_limit": seconds,
                             "query_limit": self.query_limit, "prior_phase_target_forwards": prior,
                             "git_head": git.stdout.strip() if git.returncode == 0 else None,
                             "command": shlex.join([sys.executable, *sys.argv]),
                             "started_utc": datetime.now(timezone.utc).isoformat(),
                             "environment": environment}
            write_json(self.directory / "run.json", self.metadata)
            write_json(self.directory / "environment.json", environment)
            self.log({"event": "run_started"})
            if threading.current_thread() is threading.main_thread():
                self._previous_signal = signal.getsignal(signal.SIGALRM)
                self._previous_timer = signal.getitimer(signal.ITIMER_REAL)
                signal.signal(signal.SIGALRM, self._timeout)
                signal.setitimer(signal.ITIMER_REAL, max(self.deadline - monotonic(), 0.001))
            self.check_time()
            return self
        except BaseException as error:
            if self.directory is not None:
                self._finish(type(error), error, error.__traceback__)
            else:
                self._release()
            raise

    def check_time(self):
        if monotonic() >= self.deadline:
            raise TimeoutError(f"{self.stage or 'experiment'}/{self.phase} exceeded its time limit")

    def _timeout(self, _signum, _frame):
        raise TimeoutError(f"{self.stage or 'experiment'}/{self.phase} exceeded its time limit")

    def log(self, row):
        append_jsonl(self.directory / "events.jsonl", {**row, "elapsed_seconds": monotonic() - self.started})

    def __exit__(self, error_type, error, tb):
        expired = error is None and monotonic() >= self.deadline
        if expired:
            error = TimeoutError(f"{self.stage or 'experiment'}/{self.phase} exceeded its time limit")
            error_type = TimeoutError
        self._finish(error_type, error, tb)
        if expired:
            raise error
        return False

    def _finish(self, error_type, error, tb):
        try:
            if self._previous_signal is not None:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, self._previous_signal)
            elapsed = monotonic() - self.started
            status = "completed" if error is None else ("timeout" if isinstance(error, TimeoutError) else "failed")
            if error is not None:
                write_json(self.directory / "failure.json", {
                    "error_type": error_type.__name__, "message": str(error),
                    "traceback": "".join(traceback.format_exception(error_type, error, tb)),
                    "phase": self.phase, "stage": self.stage, "elapsed_seconds": elapsed})
            summary_path = self.directory / "summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            queries = self.directory / "queries.jsonl"
            forwards = sum(row.get("event") == "forward_started" for row in read_jsonl(queries)) if queries.exists() else 0
            summary.update(status=status, phase=self.phase, stage=self.stage, signature=self.signature,
                           elapsed_seconds=elapsed, new_target_forwards=forwards,
                           run_directory=str(self.directory))
            write_json(summary_path, summary)
            self.log({"event": "run_finished", "status": status})
        finally:
            self._release()

    def _release(self):
        if self._previous_timer is not None and self._previous_timer[0] > 0:
            remaining = max(self._previous_timer[0] - (monotonic() - self.started), 0.001)
            signal.setitimer(signal.ITIMER_REAL, remaining, self._previous_timer[1])
        self._previous_timer = None
        if self._lock is not None:
            self._lock.close()
            self._lock = None
