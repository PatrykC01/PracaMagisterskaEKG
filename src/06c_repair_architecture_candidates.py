import gc
import json
import os
import random
import shutil
import time
import warnings
from collections import Counter

import cv2
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from scipy.signal import spectrogram
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.INFO)
torch.backends.cuda.matmul.allow_tf32 = True
cv2.setNumThreads(1)

OUTPUT_DIR = "/kaggle/working/architecture_candidate_repair_65x110"

RESUME_INPUT_DIR = "/kaggle/input"
PREVIOUS_LIVE_OUTPUT_DIRS = [
    "/kaggle/working/architecture_sweep_1d_2d_stft_65x110",
    "/kaggle/working/architecture_sweep_hybrid_extension_65x110",
]

DATA_DIR_CANDIDATES = [
    "/kaggle/working/datasetostrrfixed_65x110",
    "/kaggle/input/datasetostrrfixed-65x110",
    "/kaggle/input/datasetostrrfixed_65x110",
    "/kaggle/input/datasets/patrykc01/datasetostrrfixed-65x110",
    "/kaggle/input/datasets/patrykc01/datasetostrrfixed_65x110",
]

ALL_MODELS = [
    "CNN1D",
    "BILSTM1D",
    "TRANSFORMER1D",
    "CNN2D",
    "BILSTM2D",
    "VIT2D",
    "CNN_BILSTM1D",
    "CNN_BILSTM2D",
]

MODELS_TO_RUN = ["CNN2D"]

N_UNIQUE_CONFIGS_TARGET = 20
MAX_TOTAL_TRIALS_CNN2D = 200
MAX_RUNTIME_HOURS = 3.0
REPAIR_SAMPLER_SEED = 20260812

MIN_EXPECTED_EXISTING_CNN2D_UNIQUE = 16

FS = 360
EXPECTED_SEGMENT_LENGTH = 175
EXPECTED_WINDOW_BEFORE = 65
EXPECTED_WINDOW_AFTER = 110

CLASS_NAMES = ["N", "S", "V", "F"]
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_SUBSET_SEED = 20260810
TRAIN_CLASS_LIMITS = {"N": 2000, "S": None, "V": 2000, "F": None}

SWEEP_SEEDS = [3000, 3001]

BATCH_SIZE = 64
N_EPOCHS = 25
PATIENCE = 7
MIN_DELTA = 1e-4
NUM_WORKERS = 2
N_JOBS_PREPROCESSING = max(1, min(3, (os.cpu_count() or 2) - 1))

MIN_PARAMETERS = 250_000
MAX_PARAMETERS = 550_000

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

MODEL_DOMAIN = {
    "CNN1D": "RAW_1D",
    "BILSTM1D": "RAW_1D",
    "TRANSFORMER1D": "RAW_1D",
    "CNN2D": "STFT_2D",
    "BILSTM2D": "STFT_2D",
    "VIT2D": "STFT_2D",
    "CNN_BILSTM1D": "RAW_1D",
    "CNN_BILSTM2D": "STFT_2D",
}

METRIC_NAMES = [
    "Macro_F1",
    "Rare_Macro_SF",
    "Min_Rare_F1",
    "F1_N",
    "F1_S",
    "F1_V",
    "F1_F",
    "Precision_S",
    "Precision_F",
    "Recall_S",
    "Recall_F",
]


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def named_counts(labels):
    counts = Counter(np.asarray(labels).tolist())
    return {CLASS_NAMES[idx]: int(counts.get(idx, 0)) for idx in range(4)}


def resolve_data_paths():
    for candidate in DATA_DIR_CANDIDATES:
        train_path = os.path.join(candidate, "mitbih_train.npz")
        val_path = os.path.join(candidate, "mitbih_val.npz")
        if os.path.exists(train_path) and os.path.exists(val_path):
            return train_path, val_path

        backslash_train = candidate + "\\mitbih_train.npz"
        backslash_val = candidate + "\\mitbih_val.npz"
        if os.path.exists(backslash_train) and os.path.exists(backslash_val):
            return backslash_train, backslash_val

    kaggle_input = "/kaggle/input"
    if os.path.exists(kaggle_input):
        for root, _, files in os.walk(kaggle_input):
            if "mitbih_train.npz" in files and "mitbih_val.npz" in files:
                return (
                    os.path.join(root, "mitbih_train.npz"),
                    os.path.join(root, "mitbih_val.npz"),
                )

    raise FileNotFoundError(
        "Nie znaleziono mitbih_train.npz i mitbih_val.npz. "
        "Uzupelnij DATA_DIR_CANDIDATES."
    )


def validate_npz_metadata(data, split_name):
    required = {"X", "RR", "Y"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"{split_name}: brakuje tablic {sorted(missing)}.")

    signals, rr, labels = data["X"], data["RR"], data["Y"]
    if signals.ndim != 2 or signals.shape[1] != EXPECTED_SEGMENT_LENGTH:
        raise ValueError(
            f"{split_name}: oczekiwano X[:, {EXPECTED_SEGMENT_LENGTH}], "
            f"otrzymano {signals.shape}."
        )
    if rr.ndim != 2 or rr.shape[1] != 4:
        raise ValueError(f"{split_name}: oczekiwano RR[:, 4], otrzymano {rr.shape}.")
    if not (len(signals) == len(rr) == len(labels)):
        raise ValueError(f"{split_name}: rozne liczby X, RR i Y.")
    if "WINDOW_BEFORE" in data and int(data["WINDOW_BEFORE"]) != EXPECTED_WINDOW_BEFORE:
        raise ValueError(f"{split_name}: niepoprawne WINDOW_BEFORE.")
    if "WINDOW_AFTER" in data and int(data["WINDOW_AFTER"]) != EXPECTED_WINDOW_AFTER:
        raise ValueError(f"{split_name}: niepoprawne WINDOW_AFTER.")
    if "FS" in data and int(data["FS"]) != FS:
        raise ValueError(f"{split_name}: niepoprawne FS.")
    if not np.isfinite(signals).all() or not np.isfinite(rr).all():
        raise ValueError(f"{split_name}: X lub RR zawiera NaN/Inf.")


def encode_labels(labels_raw):
    labels_raw = np.asarray(labels_raw).astype(str)
    unknown = sorted(set(labels_raw) - set(CLASS_NAMES) - {"Q"})
    if unknown:
        raise ValueError(f"Nieznane etykiety: {unknown}")
    mask = labels_raw != "Q"
    labels = np.asarray([LABEL_MAP[label] for label in labels_raw[mask]], dtype=np.int64)
    return mask, labels


def standardize_segments(signals):
    signals = signals.astype(np.float32, copy=False)
    means = signals.mean(axis=1, keepdims=True)
    stds = np.maximum(signals.std(axis=1, keepdims=True), 1e-6)
    return ((signals - means) / stds).astype(np.float32, copy=False)


def deterministic_train_subset(signals, rr, labels_raw):
    mask, labels = encode_labels(labels_raw)
    signals = signals[mask].astype(np.float32, copy=False)
    rr = rr[mask].astype(np.float32, copy=False)
    rng = np.random.default_rng(TRAIN_SUBSET_SEED)
    selected = []

    for class_name in CLASS_NAMES:
        class_idx = LABEL_MAP[class_name]
        indices = np.flatnonzero(labels == class_idx)
        limit = TRAIN_CLASS_LIMITS[class_name]
        if limit is not None and len(indices) > limit:
            indices = rng.permutation(indices)[:limit]
        selected.append(np.sort(indices))

    selected = np.concatenate(selected)
    return (
        standardize_segments(signals[selected]),
        rr[selected],
        labels[selected],
    )


def full_validation(signals, rr, labels_raw):
    mask, labels = encode_labels(labels_raw)
    return (
        standardize_segments(signals[mask]),
        rr[mask].astype(np.float32, copy=False),
        labels,
    )


def load_data(train_path, val_path):
    with np.load(train_path, allow_pickle=False) as train_data:
        validate_npz_metadata(train_data, "DS1 TRAIN")
        train_signals, train_rr, train_labels = deterministic_train_subset(
            train_data["X"], train_data["RR"], train_data["Y"]
        )
    with np.load(val_path, allow_pickle=False) as val_data:
        validate_npz_metadata(val_data, "DS1 VAL")
        val_signals, val_rr, val_labels = full_validation(
            val_data["X"], val_data["RR"], val_data["Y"]
        )

    return train_signals, train_rr, train_labels, val_signals, val_rr, val_labels


def stft_one_signal(signal):
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


def process_to_stft_images(signals):
    images = Parallel(n_jobs=N_JOBS_PREPROCESSING, backend="loky")(
        delayed(stft_one_signal)(signal) for signal in signals
    )
    return np.stack(images, axis=0)


class RawRRDataset(Dataset):
    def __init__(self, signals, rr, labels):
        self.signals = torch.from_numpy(signals).unsqueeze(1)
        self.rr = torch.from_numpy(rr).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.signals[index], self.rr[index], self.labels[index]


class ImageRRDataset(Dataset):
    def __init__(self, images, rr, labels):
        self.images = torch.from_numpy(images).unsqueeze(1)
        self.rr = torch.from_numpy(rr).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index].float().div_(255.0), self.rr[index], self.labels[index]


def activation_layer(name):
    return nn.GELU() if name == "gelu" else nn.ReLU()


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


class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size, activation):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
            activation_layer(activation),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.activation = activation_layer(activation)

    def forward(self, x):
        return self.activation(x + self.block(x))


class ResidualBlock2D(nn.Module):
    def __init__(self, channels, kernel_size, activation):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(channels),
            activation_layer(activation),
            nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = activation_layer(activation)

    def forward(self, x):
        return self.activation(x + self.block(x))


class TunableCNN1D(nn.Module):
    def __init__(
        self, base_channels, blocks_per_stage, kernel_size, pool_bins,
        activation, rr_dim, head_dim, dropout,
    ):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.stem = nn.Sequential(
            nn.Conv1d(1, c1, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(c1), activation_layer(activation),
        )
        self.stage1 = self._stage(c1, c1, blocks_per_stage, kernel_size, activation)
        self.stage2 = self._stage(c1, c2, blocks_per_stage, kernel_size, activation)
        self.stage3 = self._stage(c2, c3, blocks_per_stage, kernel_size, activation)
        self.pool = nn.AdaptiveAvgPool1d(pool_bins)
        self.head = RRHead(c3 * pool_bins, rr_dim, head_dim, dropout)

    @staticmethod
    def _stage(in_channels, out_channels, blocks, kernel_size, activation):
        layers = []
        if in_channels != out_channels:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
                activation_layer(activation),
            ])
        for _ in range(blocks):
            layers.append(ResidualBlock1D(out_channels, kernel_size, activation))
        layers.append(nn.MaxPool1d(2))
        return nn.Sequential(*layers)

    def forward(self, x, rr):
        features = self.pool(self.stage3(self.stage2(self.stage1(self.stem(x)))))
        return self.head(torch.flatten(features, 1), rr)


class TunableBiLSTM1D(nn.Module):
    def __init__(
        self, hidden_size, n_layers, bidirectional, lstm_dropout,
        rr_dim, head_dim, dropout,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout if n_layers > 1 else 0.0,
        )
        feature_dim = hidden_size * (2 if bidirectional else 1)
        self.head = RRHead(feature_dim, rr_dim, head_dim, dropout)

    def forward(self, x, rr):
        _, (hidden, _) = self.lstm(x.transpose(1, 2))
        if self.bidirectional:
            features = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            features = hidden[-1]
        return self.head(features, rr)


class TunableCNNBiLSTM1D(nn.Module):

    def __init__(
        self, base_channels, blocks_per_stage, kernel_size, activation,
        hidden_size, n_layers, pooling, lstm_dropout,
        rr_dim, head_dim, dropout,
    ):
        super().__init__()
        c1, c2 = base_channels, base_channels * 2
        self.pooling = pooling

        self.stem = nn.Sequential(
            nn.Conv1d(1, c1, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(c1),
            activation_layer(activation),
        )
        self.stage1 = nn.Sequential(
            *[
                ResidualBlock1D(c1, kernel_size, activation)
                for _ in range(blocks_per_stage)
            ],
            nn.MaxPool1d(2),
        )
        self.transition = nn.Sequential(
            nn.Conv1d(c1, c2, 1, bias=False),
            nn.BatchNorm1d(c2),
            activation_layer(activation),
        )
        self.stage2 = nn.Sequential(
            *[
                ResidualBlock1D(c2, kernel_size, activation)
                for _ in range(blocks_per_stage)
            ],
            nn.MaxPool1d(2),
        )

        self.lstm = nn.LSTM(
            input_size=c2,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if n_layers > 1 else 0.0,
        )
        feature_dim = hidden_size * 2
        self.attention = nn.Linear(feature_dim, 1) if pooling == "attention" else None
        self.head = RRHead(feature_dim, rr_dim, head_dim, dropout)

    def forward(self, x, rr):
        features = self.stage2(self.transition(self.stage1(self.stem(x))))
        sequence = features.transpose(1, 2)
        sequence, _ = self.lstm(sequence)
        if self.pooling == "attention":
            weights = torch.softmax(self.attention(sequence), dim=1)
            features = torch.sum(sequence * weights, dim=1)
        else:
            features = sequence.mean(dim=1)
        return self.head(features, rr)


class TunableTransformer1D(nn.Module):
    def __init__(
        self, d_model, n_heads, n_layers, ff_multiplier,
        transformer_dropout, rr_dim, head_dim, dropout,
    ):
        super().__init__()
        self.projection = nn.Linear(1, d_model)
        self.position = nn.Parameter(torch.zeros(1, EXPECTED_SEGMENT_LENGTH, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_multiplier,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = RRHead(d_model, rr_dim, head_dim, dropout)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, x, rr):
        sequence = self.projection(x.transpose(1, 2)) + self.position
        features = self.norm(self.encoder(sequence)).mean(dim=1)
        return self.head(features, rr)


class TunableCNN2D(nn.Module):
    def __init__(
        self, base_channels, blocks_per_stage, kernel_size, pool_bins,
        activation, rr_dim, head_dim, dropout,
    ):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.stem = nn.Sequential(
            nn.Conv2d(1, c1, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c1), activation_layer(activation), nn.MaxPool2d(2),
        )
        self.stage1 = self._stage(c1, c1, blocks_per_stage, kernel_size, activation, False)
        self.stage2 = self._stage(c1, c2, blocks_per_stage, kernel_size, activation, True)
        self.stage3 = self._stage(c2, c3, blocks_per_stage, kernel_size, activation, True)
        self.pool = nn.AdaptiveAvgPool2d((pool_bins, pool_bins))
        self.head = RRHead(c3 * pool_bins * pool_bins, rr_dim, head_dim, dropout)

    @staticmethod
    def _stage(in_channels, out_channels, blocks, kernel_size, activation, use_pool):
        layers = []
        if in_channels != out_channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                activation_layer(activation),
            ])
        for _ in range(blocks):
            layers.append(ResidualBlock2D(out_channels, kernel_size, activation))
        if use_pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def forward(self, x, rr):
        features = self.pool(self.stage3(self.stage2(self.stage1(self.stem(x)))))
        return self.head(torch.flatten(features, 1), rr)


class TunableBiLSTM2D(nn.Module):
    def __init__(
        self, input_projection, hidden_size, n_layers, bidirectional,
        pooling, lstm_dropout, rr_dim, head_dim, dropout,
    ):
        super().__init__()
        self.bidirectional = bidirectional
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
            bidirectional=bidirectional,
            dropout=lstm_dropout if n_layers > 1 else 0.0,
        )
        feature_dim = hidden_size * (2 if bidirectional else 1)
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


class TunableCNNBiLSTM2D(nn.Module):

    def __init__(
        self, base_channels, blocks_per_stage, kernel_size, activation,
        hidden_size, n_layers, pooling, lstm_dropout,
        rr_dim, head_dim, dropout,
    ):
        super().__init__()
        c1, c2 = base_channels, base_channels * 2
        self.pooling = pooling

        self.stem = nn.Sequential(
            nn.Conv2d(1, c1, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            activation_layer(activation),
            nn.MaxPool2d(2),
        )
        self.stage1 = nn.Sequential(
            *[
                ResidualBlock2D(c1, kernel_size, activation)
                for _ in range(blocks_per_stage)
            ],
            nn.MaxPool2d(2),
        )
        self.transition = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            activation_layer(activation),
        )
        self.stage2 = nn.Sequential(
            *[
                ResidualBlock2D(c2, kernel_size, activation)
                for _ in range(blocks_per_stage)
            ],
            nn.MaxPool2d(kernel_size=(2, 1)),
        )

        self.lstm = nn.LSTM(
            input_size=c2,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if n_layers > 1 else 0.0,
        )
        feature_dim = hidden_size * 2
        self.attention = nn.Linear(feature_dim, 1) if pooling == "attention" else None
        self.head = RRHead(feature_dim, rr_dim, head_dim, dropout)

    def forward(self, x, rr):
        features = self.stage2(self.transition(self.stage1(self.stem(x))))

        sequence = features.mean(dim=2).transpose(1, 2)
        sequence, _ = self.lstm(sequence)
        if self.pooling == "attention":
            weights = torch.softmax(self.attention(sequence), dim=1)
            features = torch.sum(sequence * weights, dim=1)
        else:
            features = sequence.mean(dim=1)
        return self.head(features, rr)


class TunableViT2D(nn.Module):
    def __init__(
        self, patch_size, d_model, n_heads, n_layers, ff_multiplier,
        transformer_dropout, rr_dim, head_dim, dropout,
    ):
        super().__init__()
        self.patch_size = patch_size
        n_patches = (128 // patch_size) ** 2
        self.patch_embedding = nn.Linear(patch_size * patch_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_multiplier,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = RRHead(d_model, rr_dim, head_dim, dropout)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, x, rr):
        batch, _, height, width = x.shape
        patch = self.patch_size
        patches = (
            x.unfold(2, patch, patch)
            .unfold(3, patch, patch)
            .contiguous()
            .view(batch, -1, patch * patch)
        )
        sequence = self.patch_embedding(patches)
        cls = self.cls_token.expand(batch, -1, -1)
        sequence = torch.cat([cls, sequence], dim=1) + self.position
        features = self.norm(self.encoder(sequence))[:, 0]
        return self.head(features, rr)


def count_parameters(model):
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def suggest_common(trial):
    return {
        "rr_dim": trial.suggest_categorical("rr_dim", [16, 32]),
        "head_dim": trial.suggest_categorical("head_dim", [64, 128]),
        "dropout": trial.suggest_float("dropout", 0.2, 0.6, step=0.1),
    }


def suggest_model(trial, model_id):
    common = suggest_common(trial)

    if model_id == "CNN1D":
        model = TunableCNN1D(
            base_channels=trial.suggest_categorical("base_channels", [28, 32, 36, 40]),
            blocks_per_stage=trial.suggest_int("blocks_per_stage", 1, 2),
            kernel_size=trial.suggest_categorical("kernel_size", [5, 7, 11]),
            pool_bins=trial.suggest_categorical("pool_bins", [2, 4, 6]),
            activation=trial.suggest_categorical("activation", ["relu", "gelu"]),
            **common,
        )

    elif model_id == "BILSTM1D":
        n_layers = trial.suggest_int("n_layers", 1, 3)
        model = TunableBiLSTM1D(
            hidden_size=trial.suggest_categorical("hidden_size", [96, 104, 112, 120, 128]),
            n_layers=n_layers,
            bidirectional=True,
            lstm_dropout=(
                trial.suggest_float("lstm_dropout", 0.0, 0.4, step=0.1)
                if n_layers > 1 else 0.0
            ),
            **common,
        )

    elif model_id == "CNN_BILSTM1D":
        n_layers = trial.suggest_int("n_layers", 1, 2)
        model = TunableCNNBiLSTM1D(
            base_channels=trial.suggest_categorical(
                "base_channels", [24, 28, 32, 36]
            ),
            blocks_per_stage=trial.suggest_int("blocks_per_stage", 1, 2),
            kernel_size=trial.suggest_categorical("kernel_size", [5, 7, 11]),
            activation=trial.suggest_categorical("activation", ["relu", "gelu"]),
            hidden_size=trial.suggest_categorical(
                "hidden_size", [64, 80, 96, 112]
            ),
            n_layers=n_layers,
            pooling=trial.suggest_categorical("pooling", ["mean", "attention"]),
            lstm_dropout=(
                trial.suggest_float("lstm_dropout", 0.0, 0.4, step=0.1)
                if n_layers > 1 else 0.0
            ),
            **common,
        )

    elif model_id == "TRANSFORMER1D":
        pair = trial.suggest_categorical(
            "model_head_pair", ["64x4", "80x4", "96x4", "96x8", "112x4", "112x8"]
        )
        d_model, n_heads = [int(value) for value in pair.split("x")]
        model = TunableTransformer1D(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=trial.suggest_int("n_layers", 2, 5),
            ff_multiplier=trial.suggest_categorical("ff_multiplier", [2, 4]),
            transformer_dropout=trial.suggest_float(
                "transformer_dropout", 0.0, 0.4, step=0.1
            ),
            **common,
        )

    elif model_id == "CNN2D":
        model = TunableCNN2D(
            base_channels=trial.suggest_categorical("base_channels", [22, 26, 30, 34]),
            blocks_per_stage=trial.suggest_int("blocks_per_stage", 1, 2),
            kernel_size=trial.suggest_categorical("kernel_size", [3, 5]),
            pool_bins=trial.suggest_categorical("pool_bins", [2, 3, 4]),
            activation=trial.suggest_categorical("activation", ["relu", "gelu"]),
            **common,
        )

    elif model_id == "BILSTM2D":
        n_layers = trial.suggest_int("n_layers", 1, 3)
        model = TunableBiLSTM2D(
            input_projection=trial.suggest_categorical(
                "input_projection", [64, 96, 128]
            ),
            hidden_size=trial.suggest_categorical("hidden_size", [80, 96, 112, 128]),
            n_layers=n_layers,
            bidirectional=True,
            pooling=trial.suggest_categorical("pooling", ["mean", "attention"]),
            lstm_dropout=(
                trial.suggest_float("lstm_dropout", 0.0, 0.4, step=0.1)
                if n_layers > 1 else 0.0
            ),
            **common,
        )

    elif model_id == "CNN_BILSTM2D":
        n_layers = trial.suggest_int("n_layers", 1, 2)
        model = TunableCNNBiLSTM2D(
            base_channels=trial.suggest_categorical(
                "base_channels", [12, 16, 20, 24]
            ),
            blocks_per_stage=trial.suggest_int("blocks_per_stage", 1, 2),
            kernel_size=trial.suggest_categorical("kernel_size", [3, 5]),
            activation=trial.suggest_categorical("activation", ["relu", "gelu"]),
            hidden_size=trial.suggest_categorical(
                "hidden_size", [64, 80, 96, 112]
            ),
            n_layers=n_layers,
            pooling=trial.suggest_categorical("pooling", ["mean", "attention"]),
            lstm_dropout=(
                trial.suggest_float("lstm_dropout", 0.0, 0.4, step=0.1)
                if n_layers > 1 else 0.0
            ),
            **common,
        )

    elif model_id == "VIT2D":
        pair = trial.suggest_categorical(
            "model_head_pair", ["64x4", "80x4", "96x4", "96x8", "112x4", "112x8"]
        )
        d_model, n_heads = [int(value) for value in pair.split("x")]
        model = TunableViT2D(

            patch_size=trial.suggest_categorical("patch_size", [16, 32]),
            d_model=d_model,
            n_heads=n_heads,
            n_layers=trial.suggest_int("n_layers", 2, 5),
            ff_multiplier=trial.suggest_categorical("ff_multiplier", [2, 4]),
            transformer_dropout=trial.suggest_float(
                "transformer_dropout", 0.0, 0.4, step=0.1
            ),
            **common,
        )
    else:
        raise ValueError(f"Nieznany model: {model_id}")

    return model


def evaluate_model(model, loader):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for inputs, rr, batch_labels in loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            rr = rr.to(DEVICE, non_blocking=True)
            predictions.extend(model(inputs, rr).argmax(1).cpu().numpy())
            labels.extend(batch_labels.numpy())

    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    f1 = f1_score(labels, predictions, labels=[0, 1, 2, 3], average=None, zero_division=0)
    precision = precision_score(
        labels, predictions, labels=[0, 1, 2, 3], average=None, zero_division=0
    )
    recall = recall_score(
        labels, predictions, labels=[0, 1, 2, 3], average=None, zero_division=0
    )
    return {
        "Macro_F1": float(np.mean(f1)),
        "Rare_Macro_SF": float((f1[1] + f1[3]) / 2.0),
        "Min_Rare_F1": float(min(f1[1], f1[3])),
        "F1_N": float(f1[0]),
        "F1_S": float(f1[1]),
        "F1_V": float(f1[2]),
        "F1_F": float(f1[3]),
        "Precision_S": float(precision[1]),
        "Precision_F": float(precision[3]),
        "Recall_S": float(recall[1]),
        "Recall_F": float(recall[3]),
    }


def train_one_seed(trial, model_id, train_dataset, val_dataset, seed):
    set_deterministic(seed)
    generator = torch.Generator().manual_seed(seed)
    pin_memory = DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    model = suggest_model(trial, model_id)
    n_parameters = count_parameters(model)
    trial.set_user_attr("n_parameters", n_parameters)
    if not MIN_PARAMETERS <= n_parameters <= MAX_PARAMETERS:
        del model, train_loader, val_loader
        gc.collect()
        raise optuna.TrialPruned(
            f"Parameter budget: {n_parameters} not in "
            f"[{MIN_PARAMETERS}, {MAX_PARAMETERS}]"
        )

    model = model.to(DEVICE)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_metrics = None
    best_macro = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        for inputs, rr, labels in train_loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            rr = rr.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs, rr), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        metrics = evaluate_model(model, val_loader)
        if metrics["Macro_F1"] > best_macro + MIN_DELTA:
            best_macro = metrics["Macro_F1"]
            best_metrics = dict(metrics)
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            break

    best_metrics["best_epoch"] = int(best_epoch)
    best_metrics["n_parameters"] = n_parameters

    del model, optimizer, train_loader, val_loader
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return best_metrics


def make_objective(model_id, train_dataset, val_dataset):
    def objective(trial):
        started = time.time()
        seed_metrics = []
        try:
            for seed in SWEEP_SEEDS:
                seed_metrics.append(
                    train_one_seed(trial, model_id, train_dataset, val_dataset, seed)
                )
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                raise optuna.TrialPruned("CUDA out of memory") from error
            raise

        for metric in METRIC_NAMES:
            values = np.asarray([row[metric] for row in seed_metrics], dtype=float)
            trial.set_user_attr(f"{metric}_mean", float(values.mean()))
            trial.set_user_attr(
                f"{metric}_std",
                float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            )

        trial.set_user_attr(
            "best_epoch_mean",
            float(np.mean([row["best_epoch"] for row in seed_metrics])),
        )
        trial.set_user_attr("time_s", float(time.time() - started))
        trial.set_user_attr("model_id", model_id)
        trial.set_user_attr("domain", MODEL_DOMAIN[model_id])

        rare_macro = trial.user_attrs["Rare_Macro_SF_mean"]
        macro_f1 = trial.user_attrs["Macro_F1_mean"]

        return rare_macro, macro_f1

    return objective


def safe_json_value(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def trial_payload(trial, model_id, role):
    return {
        "model_id": model_id,
        "domain": MODEL_DOMAIN[model_id],
        "trial_number": int(trial.number),
        "role": role,
        "Rare_Macro_SF_mean": float(trial.values[0]),
        "Macro_F1_mean": float(trial.values[1]),
        "Min_Rare_F1_mean": float(trial.user_attrs["Min_Rare_F1_mean"]),
        "F1_N": float(trial.user_attrs["F1_N_mean"]),
        "F1_S": float(trial.user_attrs["F1_S_mean"]),
        "F1_V": float(trial.user_attrs["F1_V_mean"]),
        "F1_F": float(trial.user_attrs["F1_F_mean"]),
        "Precision_S": float(trial.user_attrs["Precision_S_mean"]),
        "Precision_F": float(trial.user_attrs["Precision_F_mean"]),
        "Recall_S": float(trial.user_attrs["Recall_S_mean"]),
        "Recall_F": float(trial.user_attrs["Recall_F_mean"]),
        "best_epoch_mean": float(trial.user_attrs["best_epoch_mean"]),
        "n_parameters": int(trial.user_attrs["n_parameters"]),
        "params": {key: safe_json_value(value) for key, value in trial.params.items()},
        "raw_window": {"before": 65, "after": 110, "length": 175},
        "stft_config": STFT_CONFIG if MODEL_DOMAIN[model_id] == "STFT_2D" else None,
        "screen_seeds": SWEEP_SEEDS,
        "requires_confirmation": True,
        "ds2_used": False,
    }


def parameter_signature(trial):
    normalized = {
        key: safe_json_value(value) for key, value in sorted(trial.params.items())
    }
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def unique_complete_trials(study):

    seen = set()
    unique_trials = []
    duplicate_numbers = []
    complete_trials = sorted(
        [
            trial for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
            and trial.values is not None
        ],
        key=lambda trial: trial.number,
    )
    for trial in complete_trials:
        signature = parameter_signature(trial)
        if signature in seen:
            duplicate_numbers.append(int(trial.number))
            continue
        seen.add(signature)
        unique_trials.append(trial)
    return unique_trials, duplicate_numbers


def pareto_from_unique_trials(trials):
    pareto = []
    for trial in trials:
        dominated = any(
            other.number != trial.number
            and other.values[0] >= trial.values[0]
            and other.values[1] >= trial.values[1]
            and (
                other.values[0] > trial.values[0]
                or other.values[1] > trial.values[1]
            )
            for other in trials
        )
        if not dominated:
            pareto.append(trial)
    return sorted(pareto, key=lambda trial: trial.number)


def save_corrected_study_outputs(study, model_id):
    suffix = model_id.lower()
    study.trials_dataframe(
        attrs=("number", "values", "params", "user_attrs", "state")
    ).to_csv(os.path.join(OUTPUT_DIR, f"trials_{suffix}.csv"), index=False)

    unique_trials, duplicate_numbers = unique_complete_trials(study)
    if not unique_trials:
        raise RuntimeError(f"{model_id}: brak unikalnych triali COMPLETE.")

    pareto_trials = pareto_from_unique_trials(unique_trials)
    rare_trial = max(
        unique_trials,
        key=lambda trial: (trial.values[0], trial.values[1]),
    )
    balanced_trial = max(
        unique_trials,
        key=lambda trial: (
            trial.user_attrs.get("Min_Rare_F1_mean", -np.inf),
            trial.values[0],
            trial.values[1],
        ),
    )

    with open(
        os.path.join(OUTPUT_DIR, f"pareto_{suffix}.json"),
        "w", encoding="utf-8",
    ) as file:
        json.dump(
            [trial_payload(trial, model_id, "pareto_unique") for trial in pareto_trials],
            file, ensure_ascii=False, indent=2,
        )

    with open(
        os.path.join(OUTPUT_DIR, f"candidate_rare_priority_{suffix}.json"),
        "w", encoding="utf-8",
    ) as file:
        json.dump(
            trial_payload(rare_trial, model_id, "rare_priority"),
            file, ensure_ascii=False, indent=2,
        )

    with open(
        os.path.join(OUTPUT_DIR, f"candidate_balanced_minrare_{suffix}.json"),
        "w", encoding="utf-8",
    ) as file:
        json.dump(
            trial_payload(balanced_trial, model_id, "balanced_minrare_all_unique"),
            file, ensure_ascii=False, indent=2,
        )

    with open(
        os.path.join(OUTPUT_DIR, f"excluded_duplicates_{suffix}.json"),
        "w", encoding="utf-8",
    ) as file:
        json.dump(duplicate_numbers, file, ensure_ascii=False, indent=2)

    return {
        "model_id": model_id,
        "domain": MODEL_DOMAIN[model_id],
        "n_trials_total": len(study.trials),
        "n_complete_total": sum(
            trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
        ),
        "n_unique_complete": len(unique_trials),
        "excluded_duplicate_trial_numbers": duplicate_numbers,
        "pareto_trial_numbers": [int(trial.number) for trial in pareto_trials],
        "rare_priority_trial": int(rare_trial.number),
        "rare_priority_rare_macro": float(rare_trial.values[0]),
        "rare_priority_macro_f1": float(rare_trial.values[1]),
        "rare_priority_minrare": float(rare_trial.user_attrs["Min_Rare_F1_mean"]),
        "balanced_minrare_trial": int(balanced_trial.number),
        "balanced_minrare": float(balanced_trial.user_attrs["Min_Rare_F1_mean"]),
        "balanced_rare_macro": float(balanced_trial.values[0]),
        "balanced_macro_f1": float(balanced_trial.values[1]),
    }


def find_all_files(root_dir, filename):
    matches = []
    if root_dir is None or not os.path.exists(root_dir):
        return matches
    direct = os.path.join(root_dir, filename)
    if os.path.isfile(direct):
        matches.append(direct)
    for root, _, files in os.walk(root_dir):
        if filename in files:
            path = os.path.join(root, filename)
            if path not in matches:
                matches.append(path)
    return matches


def load_study_from_path(db_path, model_id):
    suffix = model_id.lower()
    return optuna.load_study(
        study_name=f"{suffix}_65x110_multi_v1",
        storage=f"sqlite:///{db_path}",
    )


def database_score(db_path, model_id):
    study = load_study_from_path(db_path, model_id)
    unique_trials, _ = unique_complete_trials(study)
    n_complete = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    return len(unique_trials), n_complete, len(study.trials)


def restore_best_databases():

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    search_roots = [OUTPUT_DIR, *PREVIOUS_LIVE_OUTPUT_DIRS, RESUME_INPUT_DIR]
    selected = {}

    for model_id in ALL_MODELS:
        filename = f"study_{model_id.lower()}_multi_v1.db"
        destination = os.path.join(OUTPUT_DIR, filename)
        candidates = []
        for root_dir in search_roots:
            candidates.extend(find_all_files(root_dir, filename))
        candidates = list(dict.fromkeys(candidates))

        valid = []
        for candidate in candidates:
            try:
                valid.append((database_score(candidate, model_id), candidate))
            except Exception:
                pass

        if not valid:
            raise FileNotFoundError(
                f"Nie znaleziono zgodnej bazy {filename}. Podlacz wyniki "
                "pierwotnego sweepu i rozszerzenia hybrydowego jako Kaggle Inputs."
            )

        score, source = max(valid, key=lambda item: item[0])
        if os.path.abspath(source) != os.path.abspath(destination):
            shutil.copy2(source, destination)
        selected[model_id] = {
            "source": source,
            "destination": destination,
            "score_before_repair": {
                "n_unique_complete": int(score[0]),
                "n_complete_total": int(score[1]),
                "n_trials_total": int(score[2]),
            },
        }

    return selected


def validate_required_cnn2d_database():
    db_path = os.path.join(OUTPUT_DIR, "study_cnn2d_multi_v1.db")
    study = load_study_from_path(db_path, "CNN2D")
    unique_trials, _ = unique_complete_trials(study)
    if len(unique_trials) < MIN_EXPECTED_EXISTING_CNN2D_UNIQUE:
        raise RuntimeError(
            f"Baza CNN2D ma tylko {len(unique_trials)} unikalnych COMPLETE; "
            f"oczekiwano co najmniej {MIN_EXPECTED_EXISTING_CNN2D_UNIQUE}."
        )


def save_manifest(studies_info, selected_sources, elapsed_hours):
    payload = {
        "protocol": "CNN2D unique-config repair and corrected candidate reselection",
        "raw_window": {"before": 65, "after": 110, "length": 175},
        "stft_config": STFT_CONFIG,
        "rr_features_used": True,
        "train_subset_seed": TRAIN_SUBSET_SEED,
        "train_class_limits": TRAIN_CLASS_LIMITS,
        "validation": "full DS1 VAL",
        "models_reselected": ALL_MODELS,
        "models_retrained": MODELS_TO_RUN,
        "model_domain": MODEL_DOMAIN,
        "unique_config_target_cnn2d": N_UNIQUE_CONFIGS_TARGET,
        "max_total_trials_cnn2d": MAX_TOTAL_TRIALS_CNN2D,
        "repair_sampler_seed": REPAIR_SAMPLER_SEED,
        "screen_seeds": SWEEP_SEEDS,
        "parameter_budget": [MIN_PARAMETERS, MAX_PARAMETERS],
        "n_epochs": N_EPOCHS,
        "patience": PATIENCE,
        "early_stopping_metric": "Macro F1",
        "optuna_directions": ["maximize Rare_Macro_SF", "maximize Macro_F1"],
        "database_selection": selected_sources,
        "studies_after_repair_and_reselection": studies_info,
        "candidate_rules": {
            "rare_priority": "max Rare_Macro_SF, tie-break Macro_F1",
            "balanced_minrare": (
                "max Min_Rare_F1 among all unique COMPLETE, "
                "tie-break Rare_Macro_SF then Macro_F1"
            ),
            "duplicate_policy": "keep earliest COMPLETE per exact params signature",
        },
        "elapsed_hours_this_session": elapsed_hours,
        "requires_confirmation": True,
        "confirmation_recommended_seeds": list(range(4000, 4010)),
        "ds2_used": False,
    }
    with open(
        os.path.join(OUTPUT_DIR, "architecture_candidate_repair_manifest.json"),
        "w", encoding="utf-8",
    ) as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    pd.DataFrame(list(studies_info.values())).to_csv(
        os.path.join(OUTPUT_DIR, "architecture_candidate_selection_summary.csv"),
        index=False,
    )


def main():
    invalid = set(ALL_MODELS) - set(MODEL_DOMAIN)
    if invalid:
        raise ValueError(f"Nieznane modele: {sorted(invalid)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    selected_sources = restore_best_databases()
    validate_required_cnn2d_database()

    session_start = time.time()
    deadline = session_start + MAX_RUNTIME_HOURS * 3600
    cnn2d_path = os.path.join(OUTPUT_DIR, "study_cnn2d_multi_v1.db")
    cnn2d_study = optuna.create_study(
        study_name="cnn2d_65x110_multi_v1",
        storage=f"sqlite:///{cnn2d_path}",
        sampler=optuna.samplers.NSGAIISampler(seed=REPAIR_SAMPLER_SEED),
        directions=["maximize", "maximize"],
        load_if_exists=True,
    )
    unique_trials, _ = unique_complete_trials(cnn2d_study)

    if len(unique_trials) < N_UNIQUE_CONFIGS_TARGET:
        train_path, val_path = resolve_data_paths()

        (
            train_signals, train_rr, train_labels,
            val_signals, val_rr, val_labels,
        ) = load_data(train_path, val_path)

        train_images = process_to_stft_images(train_signals)
        val_images = process_to_stft_images(val_signals)

        train_dataset = ImageRRDataset(train_images, train_rr, train_labels)
        val_dataset = ImageRRDataset(val_images, val_rr, val_labels)

        while (
            len(unique_trials) < N_UNIQUE_CONFIGS_TARGET
            and len(cnn2d_study.trials) < MAX_TOTAL_TRIALS_CNN2D
            and time.time() < deadline
        ):
            cnn2d_study.optimize(
                make_objective("CNN2D", train_dataset, val_dataset),
                n_trials=1,
                timeout=max(1, int(deadline - time.time())),
                gc_after_trial=True,
                show_progress_bar=False,
            )
            unique_trials, _ = unique_complete_trials(cnn2d_study)

    unique_trials, _ = unique_complete_trials(cnn2d_study)

    studies_info = {}
    for model_id in ALL_MODELS:
        db_path = os.path.join(OUTPUT_DIR, f"study_{model_id.lower()}_multi_v1.db")
        study = load_study_from_path(db_path, model_id)
        studies_info[model_id] = save_corrected_study_outputs(study, model_id)

    elapsed_hours = (time.time() - session_start) / 3600.0
    save_manifest(studies_info, selected_sources, elapsed_hours)


if __name__ == "__main__":
    main()
