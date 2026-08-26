from __future__ import annotations

import gc
import json
import math
import os
import random
import time
import warnings
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from scipy.signal import spectrogram
from scipy.stats import t as student_t
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset


warnings.filterwarnings("ignore", category=UserWarning)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)
cv2.setNumThreads(1)


OUTPUT_DIR = Path("/kaggle/working/ds1_group_f_generalization_screen")
SEARCH_ROOTS = [Path("/kaggle/input"), Path("/kaggle/working")]

FS = 360
EXPECTED_SEGMENT_LENGTH = 175
EXPECTED_WINDOW_BEFORE = 65
EXPECTED_WINDOW_AFTER = 110

CLASS_NAMES = ["N", "S", "V", "F"]
LABEL_MAP = {name: index for index, name in enumerate(CLASS_NAMES)}
F_CLASS_INDEX = LABEL_MAP["F"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_WORKERS = 2
N_JOBS_PREPROCESSING = max(1, min(3, (os.cpu_count() or 2) - 1))

SEEDS = list(range(8200, 8205))
FIXED_EPOCHS = 15
MAX_RUNTIME_HOURS = 10.5
MIN_REMAINING_MINUTES_FOR_NEW_RUN = 12.0

SAVE_PREDICTIONS = True
SAVE_MODEL_CHECKPOINTS = False

STFT_CONFIG = {
    "method": "STFT",
    "image_size": [128, 128],
    "norm_type": "sqrt",
    "clip_pct": 5,
    "nperseg": 64,
    "window": "hamming",
    "noverlap_pct": 0.25,
    "nfft": 64,
}

ARCHITECTURE = {
    "candidate_id": "BILSTM2D_T7_BOTH",
    "model_id": "BILSTM2D",
    "n_parameters": 318_885,
    "params": {
        "rr_dim": 32,
        "head_dim": 128,
        "dropout": 0.4,
        "n_layers": 1,
        "input_projection": 128,
        "hidden_size": 128,
        "pooling": "attention",
        "learning_rate": 0.0023942104033463168,
        "weight_decay": 1.1605943819949519e-05,
    },
}

FOLDS = [
    {"fold_id": "HOLDOUT_208", "holdout_records": ["208"]},
    {"fold_id": "HOLDOUT_223", "holdout_records": ["223"]},
    {"fold_id": "HOLDOUT_205", "holdout_records": ["205"]},
    {
        "fold_id": "HOLDOUT_MINOR_F",
        "holdout_records": ["108", "109", "124", "201", "203", "215", "114"],
    },
]

EXPECTED_F_RECORD_COUNTS = {
    "108": 2,
    "109": 2,
    "114": 4,
    "124": 5,
    "201": 2,
    "203": 1,
    "205": 11,
    "208": 372,
    "215": 1,
    "223": 14,
}


def make_configs():
    return [
        {
            "config_id": "BASELINE_CE",
            "family": "baseline",
            "kind": "ce",
        },
        {
            "config_id": "FOCAL_A025_G3",
            "family": "focal",
            "kind": "focal",
            "alpha_power": 0.25,
            "gamma": 3.0,
        },
        {
            "config_id": "F_RECORD_BALANCED_CE_P025",
            "family": "record_balanced",
            "kind": "record_balanced_ce",
            "class_weight_power": 0.25,
        },
        {
            "config_id": "F_RECORD_BALANCED_FOCAL_A025_G3",
            "family": "record_balanced",
            "kind": "record_balanced_focal",
            "class_weight_power": 0.25,
            "gamma": 3.0,
        },
        {
            "config_id": "WGAN_F_RECORD_BALANCED_X5",
            "family": "wgan_f_record_balanced",
            "kind": "wgan_f_record_balanced",
            "wgan_factor": 5,
        },
        {
            "config_id": "WGAN_F_RECORD_BALANCED_X10",
            "family": "wgan_f_record_balanced",
            "kind": "wgan_f_record_balanced",
            "wgan_factor": 10,
        },
        {
            "config_id": "WGAN_F_RECORD_BALANCED_X20",
            "family": "wgan_f_record_balanced",
            "kind": "wgan_f_record_balanced",
            "wgan_factor": 20,
        },
    ]


CONFIGS = make_configs()
CONFIG_BY_ID = {config["config_id"]: config for config in CONFIGS}
MAX_WGAN_FACTOR = max(
    config.get("wgan_factor", 1) for config in CONFIGS
)
EXPECTED_RUNS = len(FOLDS) * len(CONFIGS) * len(SEEDS)
EXPECTED_RECORD_RUNS = len(EXPECTED_F_RECORD_COUNTS) * len(CONFIGS) * len(SEEDS)


WGAN_LATENT_DIM = 128
WGAN_BASE_CHANNELS = 128
WGAN_EPOCHS = 150
WGAN_BATCH_SIZE = 64
WGAN_CRITIC_STEPS = 5
WGAN_LAMBDA_GP = 10.0
WGAN_DRIFT = 1e-3
WGAN_LR = 1e-4
WGAN_CANDIDATE_MULTIPLIER = 2
WGAN_MAX_QUALITY_ATTEMPTS = 3
WGAN_SEED_BASE = 2_100_000
WGAN_MIN_SELECTED_ACCEPTABLE_FRACTION = 0.99
WGAN_MIN_DISTANCE_RATIO = 0.25
WGAN_MAX_DISTANCE_RATIO = 2.00
WGAN_MIN_DIVERSITY_RATIO = 0.25
WGAN_MAX_DIVERSITY_RATIO = 2.00
WGAN_POOL_PROTOCOL_VERSION = 2


def set_deterministic(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def named_counts(labels: np.ndarray) -> dict[str, int]:
    counts = Counter(np.asarray(labels).astype(int).tolist())
    return {CLASS_NAMES[index]: int(counts.get(index, 0)) for index in range(4)}


def iter_named_files(filename: str):
    seen = set()
    for search_root in SEARCH_ROOTS:
        if not search_root.exists():
            continue
        for root, _, files in os.walk(search_root):
            if filename not in files:
                continue
            path = Path(root) / filename
            resolved = str(path.resolve())
            if resolved not in seen:
                seen.add(resolved)
                yield path


def find_ds1_npz_paths() -> tuple[Path, Path]:
    candidate_dirs: dict[Path, set[str]] = {}
    for filename in ["mitbih_train.npz", "mitbih_val.npz"]:
        for path in iter_named_files(filename):
            candidate_dirs.setdefault(path.parent, set()).add(filename)
    complete = [
        directory
        for directory, files in candidate_dirs.items()
        if {"mitbih_train.npz", "mitbih_val.npz"}.issubset(files)
    ]
    if not complete:
        raise FileNotFoundError(
            "Nie znaleziono jednego katalogu z mitbih_train.npz i "
            "mitbih_val.npz. Podlacz datasetostrrfixed_65x110 jako Input."
        )

    valid = []
    for directory in complete:
        train_path = directory / "mitbih_train.npz"
        val_path = directory / "mitbih_val.npz"
        try:
            with np.load(train_path, allow_pickle=False) as train_data:
                n_train = len(train_data["Y"])
            with np.load(val_path, allow_pickle=False) as val_data:
                n_val = len(val_data["Y"])
            if (n_train, n_val) == (37_862, 13_142):
                valid.append((train_path, val_path))
        except Exception:
            continue
    if not valid:
        raise RuntimeError(
            "Znalezione NPZ nie maja oczekiwanych rozmiarow 37 862 i 13 142."
        )
    return valid[0]


def find_metadata_path() -> Path:
    candidates = []
    for path in iter_named_files("all_split_beat_metadata.csv"):
        try:
            frame = pd.read_csv(path, dtype={"record_id": str})
            required = {
                "split", "split_index", "record_id", "class_id", "class_name"
            }
            if required.issubset(frame.columns):
                n_ds1 = int(frame["split"].isin(["DS1_TRAIN", "DS1_VAL"]).sum())
                if n_ds1 == 51_004:
                    candidates.append(path)
        except Exception:
            continue
    if not candidates:
        raise FileNotFoundError(
            "Nie znaleziono poprawnego all_split_beat_metadata.csv z etapu 5A."
        )
    return candidates[0]


def labels_to_ids(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.integer):
        result = values.astype(np.int64)
    else:
        normalized = [
            value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
            for value in values
        ]
        result = np.asarray([LABEL_MAP[value] for value in normalized], dtype=np.int64)
    if not set(np.unique(result)).issubset({0, 1, 2, 3}):
        raise ValueError(f"Nieznane etykiety: {np.unique(result)}")
    return result


def standardize_segments(signals: np.ndarray) -> np.ndarray:
    signals = signals.astype(np.float32, copy=False)
    means = signals.mean(axis=1, keepdims=True)
    stds = np.maximum(signals.std(axis=1, keepdims=True), 1e-6)
    return ((signals - means) / stds).astype(np.float32, copy=False)


def load_split(path: Path, split_name: str):
    with np.load(path, allow_pickle=False) as data:
        required = {"X", "RR", "Y"}
        if not required.issubset(data.files):
            raise ValueError(f"{path}: brak tablic {sorted(required - set(data.files))}.")
        signals = data["X"]
        rr = data["RR"]
        labels = labels_to_ids(data["Y"])
        if signals.shape != (len(labels), EXPECTED_SEGMENT_LENGTH):
            raise ValueError(f"{split_name}: niezgodny X={signals.shape}.")
        if rr.shape != (len(labels), 4):
            raise ValueError(f"{split_name}: niezgodny RR={rr.shape}.")
        if "WINDOW_BEFORE" in data and int(data["WINDOW_BEFORE"]) != 65:
            raise ValueError(f"{split_name}: niezgodne WINDOW_BEFORE.")
        if "WINDOW_AFTER" in data and int(data["WINDOW_AFTER"]) != 110:
            raise ValueError(f"{split_name}: niezgodne WINDOW_AFTER.")
        signals = standardize_segments(signals)
        rr = rr.astype(np.float32, copy=False)
    if not np.isfinite(signals).all() or not np.isfinite(rr).all():
        raise ValueError(f"{split_name}: NaN/Inf w X lub RR.")
    return signals, rr, labels


def load_and_validate_ds1():
    train_path, val_path = find_ds1_npz_paths()
    metadata_path = find_metadata_path()
    train_signals, train_rr, train_labels = load_split(train_path, "DS1_TRAIN")
    val_signals, val_rr, val_labels = load_split(val_path, "DS1_VAL")

    metadata_all = pd.read_csv(metadata_path, dtype={"record_id": str})
    metadata_frames = []
    for split_name, labels in [
        ("DS1_TRAIN", train_labels),
        ("DS1_VAL", val_labels),
    ]:
        frame = metadata_all[metadata_all["split"] == split_name].copy()
        frame = frame.sort_values("split_index").reset_index(drop=True)
        if len(frame) != len(labels):
            raise RuntimeError(
                f"{split_name}: metadata={len(frame)}, NPZ={len(labels)}."
            )
        if not np.array_equal(frame["class_id"].to_numpy(int), labels):
            raise RuntimeError(
                f"{split_name}: kolejnosc klas metadata nie zgadza sie z NPZ."
            )
        metadata_frames.append(frame)

    signals = np.concatenate([train_signals, val_signals], axis=0)
    rr = np.concatenate([train_rr, val_rr], axis=0)
    labels = np.concatenate([train_labels, val_labels], axis=0)
    metadata = pd.concat(metadata_frames, ignore_index=True)
    metadata["full_ds1_index"] = np.arange(len(metadata), dtype=np.int64)
    record_ids = metadata["record_id"].astype(str).to_numpy()

    if len(labels) != 51_004:
        raise RuntimeError(f"Pelny DS1 ma {len(labels)} zamiast 51 004 uderzen.")
    if named_counts(labels) != {"N": 45_858, "S": 944, "V": 3_788, "F": 414}:
        raise RuntimeError(f"Niezgodne klasy DS1: {named_counts(labels)}")

    f_counts = (
        metadata[metadata["class_id"] == F_CLASS_INDEX]
        .groupby("record_id")
        .size()
        .astype(int)
        .to_dict()
    )
    if f_counts != EXPECTED_F_RECORD_COUNTS:
        raise RuntimeError(
            f"Niezgodny rozklad F po rekordach: {f_counts} != "
            f"{EXPECTED_F_RECORD_COUNTS}."
        )

    fold_records = [record for fold in FOLDS for record in fold["holdout_records"]]
    if len(fold_records) != len(set(fold_records)):
        raise RuntimeError("Rekord F wystepuje w wiecej niz jednym foldzie.")
    if set(fold_records) != set(EXPECTED_F_RECORD_COUNTS):
        raise RuntimeError("Foldy nie pokrywaja dokladnie wszystkich rekordow F w DS1.")

    source_info = {
        "train_npz": str(train_path),
        "val_npz": str(val_path),
        "metadata_csv": str(metadata_path),
    }
    return signals, rr, labels, record_ids, metadata, source_info


def stft_one_signal(signal: np.ndarray) -> np.ndarray:
    nperseg = STFT_CONFIG["nperseg"]
    noverlap = int(round(nperseg * STFT_CONFIG["noverlap_pct"]))
    _, _, coefficients = spectrogram(
        signal,
        fs=FS,
        window=STFT_CONFIG["window"],
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=STFT_CONFIG["nfft"],
        scaling="density",
        mode="psd",
    )
    image = np.sqrt(np.abs(coefficients) + 1e-8)
    height, width = STFT_CONFIG["image_size"]
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    clip_pct = STFT_CONFIG["clip_pct"]
    low, high = np.percentile(image, [clip_pct, 100 - clip_pct])
    image = np.clip(image, low, high)
    minimum, maximum = float(image.min()), float(image.max())
    image = (image - minimum) / (maximum - minimum + 1e-8)
    return np.rint(image * 255.0).astype(np.uint8)


def process_to_stft_images(signals: np.ndarray) -> np.ndarray:
    images = Parallel(n_jobs=N_JOBS_PREPROCESSING, backend="loky")(
        delayed(stft_one_signal)(signal) for signal in signals
    )
    return np.stack(images, axis=0)


class IndexedImageRRDataset(Dataset):


    def __init__(
        self,
        all_images: np.ndarray,
        all_rr: np.ndarray,
        all_labels: np.ndarray,
        indices: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ):
        self.images = torch.from_numpy(all_images)
        self.rr = torch.from_numpy(all_rr).float()
        self.labels = torch.from_numpy(all_labels).long()
        self.indices = torch.from_numpy(np.asarray(indices, dtype=np.int64))
        if sample_weights is None:
            sample_weights = np.ones(len(indices), dtype=np.float32)
        sample_weights = np.asarray(sample_weights, dtype=np.float32)
        if sample_weights.shape != (len(indices),):
            raise ValueError("Niezgodny ksztalt sample_weights.")
        self.sample_weights = torch.from_numpy(sample_weights)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, local_index):
        source_index = self.indices[local_index]
        image = self.images[source_index].unsqueeze(0).float().div(255.0)
        return (
            image,
            self.rr[source_index],
            self.labels[source_index],
            self.sample_weights[local_index],
        )


class SyntheticImageRRDataset(Dataset):
    def __init__(self, images: np.ndarray, rr: np.ndarray, labels: np.ndarray):
        self.images = torch.from_numpy(images)
        self.rr = torch.from_numpy(rr).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.images[index].unsqueeze(0).float().div(255.0),
            self.rr[index],
            self.labels[index],
            torch.tensor(1.0, dtype=torch.float32),
        )


class RRHead(nn.Module):
    def __init__(self, feature_dim, rr_dim, head_dim, dropout):
        super().__init__()
        self.rr_projector = nn.Sequential(nn.Linear(4, rr_dim), nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim + rr_dim, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 4),
        )

    def forward(self, features, rr):
        return self.classifier(
            torch.cat([features, self.rr_projector(rr)], dim=1)
        )


class TunableBiLSTM2D(nn.Module):
    def __init__(
        self,
        input_projection,
        hidden_size,
        n_layers,
        pooling,
        rr_dim,
        head_dim,
        dropout,
    ):
        super().__init__()
        self.pooling = pooling
        self.input_projection = nn.Sequential(
            nn.Linear(128, input_projection),
            nn.LayerNorm(input_projection),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=input_projection,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        feature_dim = hidden_size * 2
        self.attention = nn.Linear(feature_dim, 1) if pooling == "attention" else None
        self.head = RRHead(feature_dim, rr_dim, head_dim, dropout)

    def forward(self, x, rr):
        sequence = self.input_projection(x.squeeze(1).transpose(1, 2))
        sequence, _ = self.lstm(sequence)
        if self.pooling == "attention":
            weights = torch.softmax(self.attention(sequence), dim=1)
            features = torch.sum(sequence * weights, dim=1)
        else:
            features = sequence.mean(dim=1)
        return self.head(features, rr)


def build_model() -> nn.Module:
    params = ARCHITECTURE["params"]
    return TunableBiLSTM2D(
        input_projection=int(params["input_projection"]),
        hidden_size=int(params["hidden_size"]),
        n_layers=int(params["n_layers"]),
        pooling=str(params["pooling"]),
        rr_dim=int(params["rr_dim"]),
        head_dim=int(params["head_dim"]),
        dropout=float(params["dropout"]),
    )


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def class_weights(labels: np.ndarray, power: float) -> np.ndarray:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=4).astype(float)
    if np.any(counts == 0):
        raise ValueError(f"Brak klasy w treningu: {counts.tolist()}")
    weights = np.power(counts, -float(power))
    weights /= weights.mean()
    return weights.astype(np.float32)


def record_balanced_sample_weights(
    train_labels: np.ndarray,
    train_record_ids: np.ndarray,
    class_weight_power: float,
) -> tuple[np.ndarray, dict]:

    train_labels = np.asarray(train_labels, dtype=np.int64)
    train_record_ids = np.asarray(train_record_ids).astype(str)
    alpha = class_weights(train_labels, class_weight_power)
    weights = alpha[train_labels].astype(np.float64)

    f_mask = train_labels == F_CLASS_INDEX
    f_records, f_counts = np.unique(train_record_ids[f_mask], return_counts=True)
    if len(f_records) < 2:
        raise ValueError("Record-balanced loss wymaga co najmniej dwoch rekordow F.")
    total_f = int(f_mask.sum())
    count_by_record = dict(zip(f_records.tolist(), f_counts.astype(int).tolist()))
    for record_id, count in count_by_record.items():
        record_mask = f_mask & (train_record_ids == record_id)

        weights[record_mask] *= total_f / (len(f_records) * count)

    weights /= weights.mean()
    diagnostics = {
        "class_alpha": alpha.tolist(),
        "f_record_counts": count_by_record,
        "f_record_total_weight_after_normalization": {
            record_id: float(weights[f_mask & (train_record_ids == record_id)].sum())
            for record_id in f_records
        },
        "sample_weight_min": float(weights.min()),
        "sample_weight_max": float(weights.max()),
        "sample_weight_mean": float(weights.mean()),
    }
    return weights.astype(np.float32), diagnostics


def sample_weights_for_config(
    config: dict,
    train_labels: np.ndarray,
    train_record_ids: np.ndarray,
) -> tuple[np.ndarray, dict]:
    if config["kind"] in {"record_balanced_ce", "record_balanced_focal"}:
        return record_balanced_sample_weights(
            train_labels,
            train_record_ids,
            float(config["class_weight_power"]),
        )
    return np.ones(len(train_labels), dtype=np.float32), {
        "sample_weight_min": 1.0,
        "sample_weight_max": 1.0,
        "sample_weight_mean": 1.0,
    }


def loss_vector(logits, targets, config, focal_alpha):
    ce = F.cross_entropy(logits, targets, reduction="none")
    kind = config["kind"]
    if kind == "focal":
        probability_true = torch.exp(-ce)
        return (
            focal_alpha[targets]
            * (1.0 - probability_true).pow(float(config["gamma"]))
            * ce
        )
    if kind == "record_balanced_focal":
        probability_true = torch.exp(-ce)
        return (1.0 - probability_true).pow(float(config["gamma"])) * ce
    return ce


class WGANGenerator1D(nn.Module):
    def __init__(self, latent_dim=WGAN_LATENT_DIM, base_channels=WGAN_BASE_CHANNELS):
        super().__init__()
        self.base_channels = int(base_channels)
        self.project = nn.Linear(latent_dim, self.base_channels * 22)
        self.network = nn.Sequential(
            nn.ConvTranspose1d(
                self.base_channels, self.base_channels, 4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm1d(self.base_channels),
            nn.ReLU(),
            nn.ConvTranspose1d(
                self.base_channels, self.base_channels // 2, 4,
                stride=2, padding=1, bias=False,
            ),
            nn.BatchNorm1d(self.base_channels // 2),
            nn.ReLU(),
            nn.ConvTranspose1d(
                self.base_channels // 2, self.base_channels // 4, 4,
                stride=2, padding=1, bias=False,
            ),
            nn.BatchNorm1d(self.base_channels // 4),
            nn.ReLU(),
            nn.Conv1d(self.base_channels // 4, 1, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, latent):
        features = self.project(latent).view(latent.size(0), self.base_channels, 22)
        return self.network(features)[..., :EXPECTED_SEGMENT_LENGTH]


class WGANCritic1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.output = nn.Linear(256 * 11, 1)

    def forward(self, signal):
        signal = F.pad(signal, (0, 1))
        return self.output(self.network(signal).flatten(1)).view(-1)


def gradient_penalty(critic, real_signals, fake_signals):
    batch_size = real_signals.size(0)
    alpha = torch.rand(batch_size, 1, 1, device=DEVICE)
    interpolated = (
        alpha * real_signals + (1.0 - alpha) * fake_signals
    ).requires_grad_(True)
    scores = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()


def wgan_record_sampling_weights(record_ids: np.ndarray) -> np.ndarray:
    record_ids = np.asarray(record_ids).astype(str)
    records, counts = np.unique(record_ids, return_counts=True)
    count_map = dict(zip(records.tolist(), counts.astype(int).tolist()))
    weights = np.asarray([1.0 / count_map[record] for record in record_ids], dtype=np.float64)
    weights /= weights.sum()
    return weights


def train_record_balanced_wgan_f(
    real_signals: np.ndarray,
    real_record_ids: np.ndarray,
    fold_id: str,
    seed: int,
    attempt: int,
):
    wgan_seed = WGAN_SEED_BASE + int(seed) * 100 + int(attempt) * 10_000
    set_deterministic(wgan_seed)
    robust_scale = max(float(np.percentile(np.abs(real_signals), 99.5)), 1e-6)
    scaled = np.clip(real_signals / robust_scale, -1.0, 1.0).astype(np.float32)
    tensor_dataset = torch.from_numpy(scaled).unsqueeze(1)
    sampling_weights = torch.from_numpy(
        wgan_record_sampling_weights(real_record_ids)
    ).double()
    sampler = torch.utils.data.WeightedRandomSampler(
        sampling_weights,
        num_samples=len(tensor_dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(wgan_seed),
    )
    loader = DataLoader(
        tensor_dataset,
        batch_size=min(WGAN_BATCH_SIZE, len(tensor_dataset)),
        sampler=sampler,
        drop_last=False,
        num_workers=0,
    )

    generator = WGANGenerator1D().to(DEVICE)
    critic = WGANCritic1D().to(DEVICE)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=WGAN_LR, betas=(0.0, 0.9))
    optimizer_c = torch.optim.Adam(critic.parameters(), lr=WGAN_LR, betas=(0.0, 0.9))
    history = []


    for epoch in range(1, WGAN_EPOCHS + 1):
        critic_losses, generator_losses, penalties = [], [], []
        for real_batch in loader:
            real_batch = real_batch.to(DEVICE)
            batch_size = real_batch.size(0)
            for _ in range(WGAN_CRITIC_STEPS):
                latent = torch.randn(batch_size, WGAN_LATENT_DIM, device=DEVICE)
                fake_batch = generator(latent).detach()
                score_real = critic(real_batch)
                score_fake = critic(fake_batch)
                penalty = gradient_penalty(critic, real_batch, fake_batch)
                loss_c = (
                    score_fake.mean() - score_real.mean()
                    + WGAN_LAMBDA_GP * penalty
                    + WGAN_DRIFT * score_real.pow(2).mean()
                )
                optimizer_c.zero_grad(set_to_none=True)
                loss_c.backward()
                optimizer_c.step()
                critic_losses.append(float(loss_c.item()))
                penalties.append(float(penalty.item()))

            latent = torch.randn(batch_size, WGAN_LATENT_DIM, device=DEVICE)
            loss_g = -critic(generator(latent)).mean()
            optimizer_g.zero_grad(set_to_none=True)
            loss_g.backward()
            optimizer_g.step()
            generator_losses.append(float(loss_g.item()))

        row = {
            "epoch": epoch,
            "critic_loss": float(np.mean(critic_losses)),
            "generator_loss": float(np.mean(generator_losses)),
            "gradient_penalty": float(np.mean(penalties)),
        }
        history.append(row)


    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / f"wgan_training_{fold_id}_seed_{seed}_attempt_{attempt}.csv",
        index=False,
    )
    del critic, optimizer_g, optimizer_c, loader
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return generator, robust_scale, wgan_seed


def generate_wgan_candidates(generator, robust_scale, n_candidates, seed):
    set_deterministic(seed)
    generator.eval()
    batches = []
    with torch.no_grad():
        for start in range(0, n_candidates, 256):
            batch_size = min(256, n_candidates - start)
            latent = torch.randn(batch_size, WGAN_LATENT_DIM, device=DEVICE)
            batches.append(generator(latent).squeeze(1).cpu().numpy())
    signals = np.concatenate(batches, axis=0) * robust_scale
    return standardize_segments(signals.astype(np.float32, copy=False))


def select_wgan_candidates(
    real_signals: np.ndarray,
    real_rr: np.ndarray,
    real_record_ids: np.ndarray,
    candidates: np.ndarray,
    n_needed: int,
):
    if len(real_signals) < 3:
        raise ValueError("Za malo realnych probek F dla WGAN-GP.")
    real_nn_model = NearestNeighbors(n_neighbors=2, metric="euclidean", n_jobs=-1)
    real_nn_model.fit(real_signals)
    real_distances = real_nn_model.kneighbors(
        real_signals, return_distance=True
    )[0][:, 1]
    nearest_model = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
    nearest_model.fit(real_signals)
    candidate_distances, nearest_indices = nearest_model.kneighbors(
        candidates, return_distance=True
    )
    candidate_distances = candidate_distances[:, 0]
    nearest_indices = nearest_indices[:, 0]
    real_median = float(np.median(real_distances))
    lower_limit = float(max(1e-6, np.quantile(real_distances, 0.01) * 0.25))
    upper_limit = float(np.quantile(real_distances, 0.99) * 3.0)
    acceptable = (
        (candidate_distances >= lower_limit)
        & (candidate_distances <= upper_limit)
    )
    log_score = np.abs(
        np.log((candidate_distances + 1e-8) / (real_median + 1e-8))
    )
    score = log_score + np.where(acceptable, 0.0, 100.0)
    order = np.argsort(score)


    source_records_all = np.asarray(real_record_ids).astype(str)[nearest_indices]
    unique_records = sorted(np.unique(np.asarray(real_record_ids).astype(str)))
    base_quota, quota_remainder = divmod(int(n_needed), len(unique_records))
    selected_by_record = {}
    used = set()
    for record_position, record_id in enumerate(unique_records):
        quota = base_quota + int(record_position < quota_remainder)
        record_candidates = order[source_records_all[order] == record_id]
        chosen = record_candidates[:quota].tolist()
        selected_by_record[record_id] = chosen
        used.update(chosen)


    missing = int(n_needed) - sum(len(values) for values in selected_by_record.values())
    if missing > 0:
        fillers = [int(index) for index in order if int(index) not in used][:missing]
        for index in fillers:
            record_id = source_records_all[index]
            selected_by_record.setdefault(record_id, []).append(index)
            used.add(index)


    selected_indices = []
    round_position = 0
    while len(selected_indices) < int(n_needed):
        added = False
        for record_id in unique_records:
            candidates_for_record = selected_by_record.get(record_id, [])
            if round_position < len(candidates_for_record):
                selected_indices.append(candidates_for_record[round_position])
                added = True
        if not added:
            break
        round_position += 1
    selected_indices = np.asarray(selected_indices[:n_needed], dtype=np.int64)
    if len(selected_indices) != int(n_needed):
        raise RuntimeError(
            f"Nie udalo sie wybrac wymaganej puli WGAN: "
            f"{len(selected_indices)} != {n_needed}."
        )
    selected = candidates[selected_indices]
    selected_distances = candidate_distances[selected_indices]
    nearest = nearest_indices[selected_indices]
    selected_rr = real_rr[nearest].astype(np.float32, copy=True)
    selected_source_records = np.asarray(real_record_ids).astype(str)[nearest]

    generated_nn = NearestNeighbors(
        n_neighbors=2, metric="euclidean", n_jobs=-1
    ).fit(selected)
    generated_self_distances = generated_nn.kneighbors(
        selected, return_distance=True
    )[0][:, 1]
    generated_self_median = float(np.median(generated_self_distances))
    selected_median = float(np.median(selected_distances))
    distance_ratio = float(selected_median / max(real_median, 1e-8))
    diversity_ratio = float(generated_self_median / max(real_median, 1e-8))
    quality = {
        "n_real": int(len(real_signals)),
        "n_real_records": int(len(np.unique(real_record_ids))),
        "n_candidates": int(len(candidates)),
        "n_selected": int(len(selected)),
        "record_balanced_candidate_selection": True,
        "selected_acceptable_fraction": float(np.mean(acceptable[selected_indices])),
        "real_nn_median": real_median,
        "selected_nn_median": selected_median,
        "generated_self_nn_median": generated_self_median,
        "selected_to_real_nn_ratio": distance_ratio,
        "generated_self_to_real_nn_ratio": diversity_ratio,
        "memorization_warning": bool(
            distance_ratio < WGAN_MIN_DISTANCE_RATIO
        ),
        "mode_collapse_warning": bool(
            diversity_ratio < WGAN_MIN_DIVERSITY_RATIO
        ),
        "distribution_shift_warning": bool(
            distance_ratio > WGAN_MAX_DISTANCE_RATIO
        ),
        "excess_diversity_warning": bool(
            diversity_ratio > WGAN_MAX_DIVERSITY_RATIO
        ),
        "selected_source_record_counts": dict(
            Counter(selected_source_records.tolist())
        ),
    }
    quality["quality_gate_passed"] = bool(
        quality["selected_acceptable_fraction"] >= WGAN_MIN_SELECTED_ACCEPTABLE_FRACTION
        and WGAN_MIN_DISTANCE_RATIO <= distance_ratio <= WGAN_MAX_DISTANCE_RATIO
        and WGAN_MIN_DIVERSITY_RATIO <= diversity_ratio <= WGAN_MAX_DIVERSITY_RATIO
    )
    quality["quality_score_lower_is_better"] = float(
        abs(math.log(max(distance_ratio, 1e-8)))
        + abs(math.log(max(diversity_ratio, 1e-8)))
        + 10.0 * (1.0 - quality["selected_acceptable_fraction"])
    )
    return selected, selected_rr, selected_source_records, quality


def wgan_pool_filename(fold_id: str, seed: int) -> str:
    return f"wgan_f_record_balanced_pool_{fold_id}_seed_{seed}.npz"


def source_record_count_json(record_ids: np.ndarray) -> str:
    return json.dumps(dict(sorted(Counter(np.asarray(record_ids).astype(str)).items())))


def find_existing_wgan_pool(
    fold_id,
    seed,
    train_f_count,
    train_f_record_ids,
):
    filename = wgan_pool_filename(fold_id, seed)
    paths = []
    direct = OUTPUT_DIR / filename
    if direct.exists():
        paths.append(direct)
    for path in iter_named_files(filename):
        paths.append(path)
    expected_source_json = source_record_count_json(train_f_record_ids)

    for path in dict.fromkeys(paths):
        try:
            with np.load(path, allow_pickle=False) as data:
                required = {
                    "X", "RR", "Y", "SOURCE_RECORD_ID", "FOLD_ID", "SEED",
                    "MAX_FACTOR", "TRAIN_F_COUNT", "TRAIN_F_RECORD_COUNTS_JSON",
                    "QUALITY_JSON", "POOL_PROTOCOL_VERSION",
                }
                if not required.issubset(data.files):
                    continue
                if str(data["FOLD_ID"].item()) != fold_id:
                    continue
                if int(data["SEED"]) != int(seed):
                    continue
                if int(data["MAX_FACTOR"]) != MAX_WGAN_FACTOR:
                    continue
                if int(data["POOL_PROTOCOL_VERSION"]) != WGAN_POOL_PROTOCOL_VERSION:
                    continue
                if int(data["TRAIN_F_COUNT"]) != int(train_f_count):
                    continue
                if str(data["TRAIN_F_RECORD_COUNTS_JSON"].item()) != expected_source_json:
                    continue
                signals = data["X"].astype(np.float32)
                rr = data["RR"].astype(np.float32)
                labels = data["Y"].astype(np.int64)
                source_records = data["SOURCE_RECORD_ID"].astype(str)
                quality = json.loads(str(data["QUALITY_JSON"].item()))
                if signals.shape != ((MAX_WGAN_FACTOR - 1) * train_f_count, 175):
                    continue
                if rr.shape != (len(signals), 4):
                    continue
                if not np.all(labels == F_CLASS_INDEX):
                    continue

            return signals, rr, labels, source_records, quality
        except Exception:
            continue
    return None


def build_or_load_wgan_pool(
    train_signals,
    train_rr,
    train_labels,
    train_record_ids,
    fold_id,
    seed,
):
    f_mask = train_labels == F_CLASS_INDEX
    real_signals = train_signals[f_mask]
    real_rr = train_rr[f_mask]
    real_record_ids = train_record_ids[f_mask]
    existing = find_existing_wgan_pool(
        fold_id, seed, len(real_signals), real_record_ids
    )
    if existing is not None:
        return existing

    n_needed = (MAX_WGAN_FACTOR - 1) * len(real_signals)
    n_candidates = max(
        n_needed * WGAN_CANDIDATE_MULTIPLIER,
        n_needed + 512,
    )
    attempts = []
    for attempt in range(1, WGAN_MAX_QUALITY_ATTEMPTS + 1):
        generator, robust_scale, wgan_seed = train_record_balanced_wgan_f(
            real_signals, real_record_ids, fold_id, seed, attempt
        )
        candidates = generate_wgan_candidates(
            generator,
            robust_scale,
            n_candidates=n_candidates,
            seed=wgan_seed + 1000,
        )
        selected, selected_rr, source_records, quality = select_wgan_candidates(
            real_signals,
            real_rr,
            real_record_ids,
            candidates,
            n_needed,
        )
        quality.update({
            "fold_id": fold_id,
            "classifier_seed": int(seed),
            "wgan_seed": int(wgan_seed),
            "quality_attempt": int(attempt),
            "max_factor": int(MAX_WGAN_FACTOR),
            "record_balanced_sampler": True,
        })
        attempts.append((quality["quality_score_lower_is_better"], selected, selected_rr, source_records, quality))
        del generator, candidates
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        if quality["quality_gate_passed"]:
            break


    _, selected, selected_rr, source_records, quality = min(
        attempts, key=lambda item: item[0]
    )
    labels = np.full(len(selected), F_CLASS_INDEX, dtype=np.int64)
    output_path = OUTPUT_DIR / wgan_pool_filename(fold_id, seed)
    np.savez_compressed(
        output_path,
        X=selected.astype(np.float32),
        RR=selected_rr.astype(np.float32),
        Y=labels,
        SOURCE_RECORD_ID=np.asarray(source_records).astype(str),
        FOLD_ID=np.asarray(fold_id),
        SEED=np.asarray(seed, dtype=np.int64),
        MAX_FACTOR=np.asarray(MAX_WGAN_FACTOR, dtype=np.int64),
        POOL_PROTOCOL_VERSION=np.asarray(WGAN_POOL_PROTOCOL_VERSION, dtype=np.int64),
        TRAIN_F_COUNT=np.asarray(len(real_signals), dtype=np.int64),
        TRAIN_F_RECORD_COUNTS_JSON=np.asarray(source_record_count_json(real_record_ids)),
        QUALITY_JSON=np.asarray(json.dumps(quality, ensure_ascii=False)),
    )
    pd.DataFrame([quality]).to_csv(
        OUTPUT_DIR / f"wgan_quality_{fold_id}_seed_{seed}.csv", index=False
    )


    return selected, selected_rr, labels, source_records, quality


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def metrics_from_confusion(matrix: np.ndarray) -> dict:
    matrix = np.asarray(matrix, dtype=float)
    row = matrix.sum(axis=1)
    col = matrix.sum(axis=0)
    f1_values, precision_values, recall_values = [], [], []
    for index in range(4):
        tp = matrix[index, index]
        fp = col[index] - tp
        fn = row[index] - tp
        precision_values.append(safe_div(tp, tp + fp))
        recall_values.append(safe_div(tp, tp + fn))
        f1_values.append(safe_div(2.0 * tp, 2.0 * tp + fp + fn))
    f1_n, f1_s, f1_v, f1_f = f1_values
    rare_hmean = safe_div(2.0 * f1_s * f1_f, f1_s + f1_f)
    total = matrix.sum()
    metrics = {
        "Accuracy": safe_div(float(np.trace(matrix)), float(total)),
        "Macro_F1": float(np.mean(f1_values)),
        "Rare_Macro_SF": float((f1_s + f1_f) / 2.0),
        "Min_Rare_F1": float(min(f1_s, f1_f)),
        "Rare_HMean_SF": rare_hmean,
    }
    for index, class_name in enumerate(CLASS_NAMES):
        metrics[f"F1_{class_name}"] = float(f1_values[index])
        metrics[f"Precision_{class_name}"] = float(precision_values[index])
        metrics[f"Recall_{class_name}"] = float(recall_values[index])
        metrics[f"Support_{class_name}"] = int(row[index])
        metrics[f"Predicted_{class_name}"] = int(col[index])
    return metrics


def evaluate_model(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, rr, labels, _ in loader:
            images = images.to(DEVICE, non_blocking=True)
            rr = rr.to(DEVICE, non_blocking=True)
            logits = model(images, rr)
            y_pred.extend(logits.argmax(1).cpu().numpy())
            y_true.extend(labels.numpy())
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    return metrics_from_confusion(matrix), matrix, y_true, y_pred


def per_record_metric_rows(
    y_true,
    y_pred,
    val_record_ids,
    fold_id,
    config,
    seed,
):
    rows = []
    for record_id in sorted(np.unique(val_record_ids)):
        mask = val_record_ids == record_id
        matrix = confusion_matrix(y_true[mask], y_pred[mask], labels=[0, 1, 2, 3])
        row = {
            "fold_id": fold_id,
            "record_id": str(record_id),
            "config_id": config["config_id"],
            "family": config["family"],
            "seed": int(seed),
            "n_beats": int(mask.sum()),
            "confusion_json": json.dumps(matrix.astype(int).tolist()),
        }
        row.update(metrics_from_confusion(matrix))
        rows.append(row)
    return rows


def make_loader(dataset, shuffle, seed):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(int(seed)) if shuffle else None,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
    )


def train_one_run(
    fold_id,
    config,
    seed,
    train_dataset,
    val_dataset,
    original_train_labels,
    train_counts,
    val_counts,
    holdout_records,
    wgan_info,
    weight_info,
):
    set_deterministic(seed)
    train_loader = make_loader(train_dataset, shuffle=True, seed=seed)
    val_loader = make_loader(val_dataset, shuffle=False, seed=seed)
    model = build_model().to(DEVICE)
    n_parameters = count_parameters(model)
    if n_parameters != int(ARCHITECTURE["n_parameters"]):
        raise RuntimeError(
            f"Oczekiwano {ARCHITECTURE['n_parameters']} parametrow, "
            f"zbudowano {n_parameters}."
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(ARCHITECTURE["params"]["learning_rate"]),
        weight_decay=float(ARCHITECTURE["params"]["weight_decay"]),
    )
    focal_alpha = None
    if config["kind"] == "focal":
        focal_alpha = torch.tensor(
            class_weights(
                original_train_labels, float(config["alpha_power"])
            ),
            dtype=torch.float32,
            device=DEVICE,
        )
    history = []
    started = time.time()
    for epoch in range(1, FIXED_EPOCHS + 1):
        model.train()
        losses = []
        for images, rr, labels, sample_weights in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            rr = rr.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            sample_weights = sample_weights.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            losses_vector = loss_vector(model(images, rr), labels, config, focal_alpha)
            loss = (losses_vector * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        history.append({
            "epoch": epoch,
            "train_loss_mean": float(np.mean(losses)),
        })

    metrics, matrix, y_true, y_pred = evaluate_model(model, val_loader)
    run_name = f"{fold_id}_{config['config_id']}_seed_{seed}"
    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / f"training_history_{run_name}.csv", index=False
    )
    if SAVE_PREDICTIONS:
        np.savez_compressed(
            OUTPUT_DIR / f"val_predictions_{run_name}.npz",
            Y_TRUE=y_true,
            Y_PRED=y_pred,
            FOLD_ID=np.asarray(fold_id),
            CONFIG_ID=np.asarray(config["config_id"]),
            SEED=np.asarray(seed, dtype=np.int64),
            HOLDOUT_RECORDS=np.asarray(holdout_records),
        )
    if SAVE_MODEL_CHECKPOINTS:
        torch.save(
            {
                "protocol": "DS1 grouped F-record screening",
                "fold_id": fold_id,
                "config": config,
                "seed": int(seed),
                "architecture": ARCHITECTURE,
                "fixed_epochs": FIXED_EPOCHS,
                "model_state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
            },
            OUTPUT_DIR / f"model_{run_name}.pt",
        )

    result = {
        "fold_id": fold_id,
        "holdout_records_json": json.dumps(holdout_records),
        "config_id": config["config_id"],
        "family": config["family"],
        "config_json": json.dumps(config, sort_keys=True),
        "seed": int(seed),
        "n_parameters": int(n_parameters),
        "fixed_epochs": int(FIXED_EPOCHS),
        "n_train_samples": int(len(train_dataset)),
        "n_val_samples": int(len(y_true)),
        "original_train_counts_json": json.dumps(train_counts),
        "val_counts_json": json.dumps(val_counts),
        "train_loss_final": float(history[-1]["train_loss_mean"]),
        "time_s": float(time.time() - started),
        "confusion_json": json.dumps(matrix.astype(int).tolist()),
        "weight_info_json": json.dumps(weight_info, sort_keys=True),
        "wgan_info_json": json.dumps(wgan_info, sort_keys=True),
    }
    result.update(metrics)

    del model, optimizer, train_loader, val_loader
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return result, y_true, y_pred


def find_best_csv(filename: str, required_columns: set[str]) -> pd.DataFrame:
    paths = []
    direct = OUTPUT_DIR / filename
    if direct.exists():
        paths.append(direct)
    for path in iter_named_files(filename):
        paths.append(path)
    candidates = []
    for path in dict.fromkeys(paths):
        try:
            frame = pd.read_csv(path, dtype={"record_id": str})
            if required_columns.issubset(frame.columns):
                candidates.append((len(frame), path, frame))
        except Exception:
            continue
    if not candidates:
        return pd.DataFrame()
    _, _, frame = max(candidates, key=lambda item: item[0])

    return frame


def t_confidence_interval(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        value = float(values[0]) if len(values) else float("nan")
        return value, value
    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    margin = float(student_t.ppf((1.0 + confidence) / 2.0, len(values) - 1) * sem)
    return mean - margin, mean + margin


SUMMARY_METRICS = [
    "RecordMacro_F1_F",
    "RecordMacro_Recall_F",
    "RecordMacro_Precision_F",
    "RecordMedian_F1_F",
    "RecordWorst_F1_F",
    "OOF_Accuracy",
    "OOF_Macro_F1",
    "OOF_Rare_Macro_SF",
    "OOF_Min_Rare_F1",
    "OOF_F1_N",
    "OOF_F1_S",
    "OOF_F1_V",
    "OOF_F1_F",
]


def make_oof_seed_summary(runs: pd.DataFrame, record_runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (config_id, seed), group in runs.groupby(["config_id", "seed"]):
        if set(group["fold_id"]) != {fold["fold_id"] for fold in FOLDS}:
            continue
        matrices = np.stack([
            np.asarray(json.loads(value), dtype=int)
            for value in group["confusion_json"]
        ])
        oof_matrix = matrices.sum(axis=0)
        pooled = metrics_from_confusion(oof_matrix)
        record_group = record_runs[
            (record_runs["config_id"] == config_id)
            & (record_runs["seed"].astype(int) == int(seed))
        ]
        expected_records = set(EXPECTED_F_RECORD_COUNTS)
        if set(record_group["record_id"].astype(str)) != expected_records:
            continue
        f1_f = record_group["F1_F"].to_numpy(float)
        row = {
            "config_id": config_id,
            "family": CONFIG_BY_ID[config_id]["family"],
            "seed": int(seed),
            "n_folds": int(len(group)),
            "n_validation_records": int(len(record_group)),
            "RecordMacro_F1_F": float(np.mean(f1_f)),
            "RecordMacro_Recall_F": float(np.mean(record_group["Recall_F"])),
            "RecordMacro_Precision_F": float(np.mean(record_group["Precision_F"])),
            "RecordMedian_F1_F": float(np.median(f1_f)),
            "RecordWorst_F1_F": float(np.min(f1_f)),
            "oof_confusion_json": json.dumps(oof_matrix.astype(int).tolist()),
        }
        wgan_infos = [json.loads(value) for value in group["wgan_info_json"]]
        if CONFIG_BY_ID[config_id]["kind"] == "wgan_f_record_balanced":
            qa_passes = [bool(info["quality_gate_passed"]) for info in wgan_infos]
            row["WGAN_QA_fold_pass_rate"] = float(np.mean(qa_passes))
            row["WGAN_QA_all_folds_passed"] = bool(all(qa_passes))
        else:
            row["WGAN_QA_fold_pass_rate"] = 1.0
            row["WGAN_QA_all_folds_passed"] = True
        for metric_name, value in pooled.items():
            if metric_name.startswith("Support_") or metric_name.startswith("Predicted_"):
                continue
            row[f"OOF_{metric_name}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def make_config_summary(oof_seed_summary: pd.DataFrame) -> pd.DataFrame:
    if oof_seed_summary.empty:
        return pd.DataFrame()
    rows = []
    for config_id, group in oof_seed_summary.groupby("config_id"):
        is_wgan = CONFIG_BY_ID[config_id]["kind"] == "wgan_f_record_balanced"
        qa_all_seeds = bool(group["WGAN_QA_all_folds_passed"].astype(bool).all())
        row = {
            "config_id": config_id,
            "family": CONFIG_BY_ID[config_id]["family"],
            "n_seeds": int(group["seed"].nunique()),
            "config_json": json.dumps(CONFIG_BY_ID[config_id], sort_keys=True),
            "WGAN_QA_fold_pass_rate": float(group["WGAN_QA_fold_pass_rate"].mean()),
            "WGAN_QA_all_seeds_passed": qa_all_seeds,
            "eligible_for_confirmation": bool((not is_wgan) or qa_all_seeds),
        }
        for metric in SUMMARY_METRICS:
            values = group[metric].to_numpy(float)
            low, high = t_confidence_interval(values)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary = summary.sort_values(
        [
            "eligible_for_confirmation",
            "RecordMacro_F1_F_mean",
            "RecordMacro_Recall_F_mean",
            "OOF_Macro_F1_mean",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    summary.insert(0, "screen_rank", np.arange(1, len(summary) + 1))
    return summary


def make_vs_baseline(oof_seed_summary: pd.DataFrame) -> pd.DataFrame:
    if oof_seed_summary.empty:
        return pd.DataFrame()
    baseline = oof_seed_summary[
        oof_seed_summary["config_id"] == "BASELINE_CE"
    ].set_index("seed")
    rows = []
    for config_id, group in oof_seed_summary.groupby("config_id"):
        if config_id == "BASELINE_CE":
            continue
        candidate = group.set_index("seed")
        common = sorted(set(candidate.index) & set(baseline.index))
        if not common:
            continue
        for metric in [
            "RecordMacro_F1_F",
            "RecordMacro_Recall_F",
            "OOF_Macro_F1",
            "OOF_F1_F",
        ]:
            differences = (
                candidate.loc[common, metric].to_numpy(float)
                - baseline.loc[common, metric].to_numpy(float)
            )
            low, high = t_confidence_interval(differences)
            rows.append({
                "candidate_config_id": config_id,
                "baseline_config_id": "BASELINE_CE",
                "metric": metric,
                "n_paired_seeds": len(common),
                "candidate_mean": float(candidate.loc[common, metric].mean()),
                "baseline_mean": float(baseline.loc[common, metric].mean()),
                "mean_difference_candidate_minus_baseline": float(np.mean(differences)),
                "mean_difference_ci95_low": low,
                "mean_difference_ci95_high": high,
                "wins_candidate": int(np.sum(differences > 0)),
                "ties": int(np.sum(differences == 0)),
                "wins_baseline": int(np.sum(differences < 0)),
            })
    return pd.DataFrame(rows)


def save_current_outputs(runs, record_runs, source_info, elapsed_hours):
    runs = runs.drop_duplicates(
        ["fold_id", "config_id", "seed"], keep="last"
    ).sort_values(["fold_id", "config_id", "seed"]).reset_index(drop=True)
    record_runs = record_runs.drop_duplicates(
        ["fold_id", "record_id", "config_id", "seed"], keep="last"
    ).sort_values(["fold_id", "record_id", "config_id", "seed"]).reset_index(drop=True)
    runs.to_csv(OUTPUT_DIR / "group_f_screen_runs.csv", index=False)
    record_runs.to_csv(OUTPUT_DIR / "group_f_screen_per_record_runs.csv", index=False)

    oof = make_oof_seed_summary(runs, record_runs)
    oof.to_csv(OUTPUT_DIR / "group_f_screen_oof_seed_summary.csv", index=False)
    summary = make_config_summary(oof)
    summary.to_csv(OUTPUT_DIR / "group_f_screen_config_summary.csv", index=False)
    comparisons = make_vs_baseline(oof)
    comparisons.to_csv(OUTPUT_DIR / "group_f_screen_vs_baseline.csv", index=False)

    complete = bool(
        len(runs) == EXPECTED_RUNS
        and len(record_runs) == EXPECTED_RECORD_RUNS
        and len(oof) == len(CONFIGS) * len(SEEDS)
    )
    manifest = {
        "protocol": "DS1 grouped F-record generalization screening",
        "screening_stage": True,
        "ds1_only": True,
        "ds2_npz_loaded": False,
        "ds2_predictions_loaded": False,
        "ds2_used_for_selection": False,
        "raw_window": {
            "before": EXPECTED_WINDOW_BEFORE,
            "after": EXPECTED_WINDOW_AFTER,
            "length": EXPECTED_SEGMENT_LENGTH,
        },
        "stft_config": STFT_CONFIG,
        "architecture": ARCHITECTURE,
        "folds": FOLDS,
        "f_record_counts_ds1": EXPECTED_F_RECORD_COUNTS,
        "seeds": SEEDS,
        "fixed_epochs": FIXED_EPOCHS,
        "early_stopping": False,
        "configs": CONFIGS,
        "primary_metric": "RecordMacro_F1_F",
        "selection_rule": (
            "eligible_for_confirmation first; then max mean RecordMacro_F1_F; "
            "tie-break mean RecordMacro_Recall_F; then mean OOF_Macro_F1. "
            "WGAN is eligible only if every fold-seed pool passes train-only QA."
        ),
        "screening_warning": (
            "This ranking selects candidates for a separate DS1 confirmation. "
            "It is not a new final DS2 result."
        ),
        "wgan": {
            "class": "F only",
            "record_balanced_real_sampler": True,
            "record_balanced_candidate_selection": True,
            "nested_pool_order": "round-robin across nearest training F records",
            "max_factor": MAX_WGAN_FACTOR,
            "pool_protocol_version": WGAN_POOL_PROTOCOL_VERSION,
            "nested_factors": [5, 10, 20],
            "epochs": WGAN_EPOCHS,
            "quality_attempts": WGAN_MAX_QUALITY_ATTEMPTS,
            "quality_gate_is_diagnostic": True,
            "failed_quality_policy": (
                "continue with best train-only quality attempt, flag as failed"
            ),
            "rr_assignment": "nearest real F beat from training records",
        },
        "expected_runs": EXPECTED_RUNS,
        "completed_runs": int(len(runs)),
        "expected_per_record_runs": EXPECTED_RECORD_RUNS,
        "completed_per_record_runs": int(len(record_runs)),
        "complete": complete,
        "source_info": source_info,
        "elapsed_hours_this_session": float(elapsed_hours),
    }
    (OUTPUT_DIR / "group_f_screen_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return runs, record_runs, oof, summary, comparisons, complete


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("Ten eksperyment wymaga GPU w Kaggle.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    session_start = time.time()
    deadline = session_start + MAX_RUNTIME_HOURS * 3600.0

    model_check = build_model()
    actual_parameters = count_parameters(model_check)
    del model_check
    if actual_parameters != ARCHITECTURE["n_parameters"]:
        raise RuntimeError(
            f"Niezgodna architektura: {actual_parameters} parametrow zamiast "
            f"{ARCHITECTURE['n_parameters']}."
        )

    signals, rr, labels, record_ids, _, source_info = load_and_validate_ds1()


    all_images = process_to_stft_images(signals)


    runs = find_best_csv(
        "group_f_screen_runs.csv",
        {"fold_id", "config_id", "seed", "confusion_json", "wgan_info_json"},
    )
    record_runs = find_best_csv(
        "group_f_screen_per_record_runs.csv",
        {"fold_id", "record_id", "config_id", "seed", "F1_F"},
    )
    if not runs.empty:
        valid_fold_ids = {fold["fold_id"] for fold in FOLDS}
        valid_config_ids = set(CONFIG_BY_ID)
        if not set(runs["fold_id"]).issubset(valid_fold_ids):
            raise RuntimeError("Plik wznowienia zawiera nieznany fold.")
        if not set(runs["config_id"]).issubset(valid_config_ids):
            raise RuntimeError("Plik wznowienia zawiera nieznana konfiguracje.")
        if not set(runs["seed"].astype(int)).issubset(set(SEEDS)):
            raise RuntimeError("Plik wznowienia zawiera nieznany seed.")
        runs = runs.drop_duplicates(["fold_id", "config_id", "seed"], keep="last")
    if not record_runs.empty:
        record_runs["record_id"] = record_runs["record_id"].astype(str)
        record_runs = record_runs.drop_duplicates(
            ["fold_id", "record_id", "config_id", "seed"], keep="last"
        )

    completed = set()
    if not runs.empty and not record_runs.empty:
        for row in runs[["fold_id", "config_id", "seed"]].itertuples(index=False):
            holdouts = next(
                fold["holdout_records"] for fold in FOLDS if fold["fold_id"] == row.fold_id
            )
            matching_records = record_runs[
                (record_runs["fold_id"] == row.fold_id)
                & (record_runs["config_id"] == row.config_id)
                & (record_runs["seed"].astype(int) == int(row.seed))
            ]
            if set(matching_records["record_id"].astype(str)) == set(holdouts):
                completed.add((row.fold_id, row.config_id, int(row.seed)))

    stopped_by_time = False
    for fold in FOLDS:
        fold_id = fold["fold_id"]
        holdout_records = fold["holdout_records"]
        val_mask = np.isin(record_ids, holdout_records)
        train_mask = ~val_mask
        train_indices = np.flatnonzero(train_mask)
        val_indices = np.flatnonzero(val_mask)
        train_labels = labels[train_indices]
        train_record_ids = record_ids[train_indices]
        val_labels = labels[val_indices]
        val_record_ids = record_ids[val_indices]
        train_counts = named_counts(train_labels)
        val_counts = named_counts(val_labels)


        for seed in SEEDS:
            pending_configs = [
                config for config in CONFIGS
                if (fold_id, config["config_id"], int(seed)) not in completed
            ]
            if not pending_configs:
                continue

            wgan_cache = None
            wgan_image_dataset = None
            for config in pending_configs:
                remaining_minutes = (deadline - time.time()) / 60.0
                if remaining_minutes < MIN_REMAINING_MINUTES_FOR_NEW_RUN:
                    stopped_by_time = True
                    break

                sample_weights, weight_info = sample_weights_for_config(
                    config, train_labels, train_record_ids
                )
                base_train_dataset = IndexedImageRRDataset(
                    all_images,
                    rr,
                    labels,
                    train_indices,
                    sample_weights,
                )
                val_dataset = IndexedImageRRDataset(
                    all_images,
                    rr,
                    labels,
                    val_indices,
                )
                train_dataset = base_train_dataset
                wgan_info = {
                    "used": False,
                    "quality_gate_passed": None,
                }

                if config["kind"] == "wgan_f_record_balanced":
                    if wgan_cache is None:
                        wgan_cache = build_or_load_wgan_pool(
                            signals[train_indices],
                            rr[train_indices],
                            train_labels,
                            train_record_ids,
                            fold_id,
                            seed,
                        )
                    pool_signals, pool_rr, pool_labels, source_records, quality = wgan_cache
                    if wgan_image_dataset is None:


                        pool_images = process_to_stft_images(pool_signals)
                        wgan_image_dataset = SyntheticImageRRDataset(
                            pool_images, pool_rr, pool_labels
                        )
                    n_needed = (
                        int(config["wgan_factor"]) - 1
                    ) * int(np.sum(train_labels == F_CLASS_INDEX))
                    if n_needed > len(wgan_image_dataset):
                        raise RuntimeError("Pula WGAN jest za mala dla konfiguracji.")
                    extra_dataset = Subset(
                        wgan_image_dataset, np.arange(n_needed, dtype=np.int64)
                    )
                    train_dataset = ConcatDataset([base_train_dataset, extra_dataset])
                    wgan_info = {
                        "used": True,
                        "factor": int(config["wgan_factor"]),
                        "n_synthetic_f": int(n_needed),
                        "max_pool_size": int(len(pool_labels)),
                        "quality_gate_passed": bool(quality["quality_gate_passed"]),
                        "quality_attempt": int(quality["quality_attempt"]),
                        "wgan_seed": int(quality["wgan_seed"]),
                        "selected_to_real_nn_ratio": float(
                            quality["selected_to_real_nn_ratio"]
                        ),
                        "generated_self_to_real_nn_ratio": float(
                            quality["generated_self_to_real_nn_ratio"]
                        ),
                        "synthetic_source_record_counts": dict(
                            Counter(source_records[:n_needed].tolist())
                        ),
                    }


                result, y_true, y_pred = train_one_run(
                    fold_id=fold_id,
                    config=config,
                    seed=seed,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    original_train_labels=train_labels,
                    train_counts=train_counts,
                    val_counts=val_counts,
                    holdout_records=holdout_records,
                    wgan_info=wgan_info,
                    weight_info=weight_info,
                )
                new_record_rows = per_record_metric_rows(
                    y_true,
                    y_pred,
                    val_record_ids,
                    fold_id,
                    config,
                    seed,
                )
                runs = pd.concat([runs, pd.DataFrame([result])], ignore_index=True)
                record_runs = pd.concat(
                    [record_runs, pd.DataFrame(new_record_rows)], ignore_index=True
                )
                elapsed_hours = (time.time() - session_start) / 3600.0
                runs, record_runs, _, _, _, _ = save_current_outputs(
                    runs, record_runs, source_info, elapsed_hours
                )
                completed.add((fold_id, config["config_id"], int(seed)))


                del train_dataset, base_train_dataset, val_dataset
                gc.collect()
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

            del wgan_cache, wgan_image_dataset
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            if stopped_by_time:
                break
        if stopped_by_time:
            break

    elapsed_hours = (time.time() - session_start) / 3600.0
    save_current_outputs(
        runs, record_runs, source_info, elapsed_hours
    )


if __name__ == "__main__":
    main()

