import gc
import itertools
import json
import os
import random
import time
import warnings
from collections import Counter

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from scipy.signal import spectrogram
from scipy.stats import t as student_t
from scipy.stats import wilcoxon
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=UserWarning)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)
cv2.setNumThreads(1)

OUTPUT_DIR = "/kaggle/working/architecture_confirmation_11_candidates_65x110"

CANDIDATE_SEARCH_ROOT = "/kaggle/input"
RESUME_SEARCH_ROOT = "/kaggle/input"

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

EXPECTED_UNIQUE_CANDIDATES = 11
CONFIRMATION_SEEDS = list(range(4000, 4010))
MAX_RUNTIME_HOURS = 10.5

FS = 360
EXPECTED_SEGMENT_LENGTH = 175
EXPECTED_WINDOW_BEFORE = 65
EXPECTED_WINDOW_AFTER = 110

CLASS_NAMES = ["N", "S", "V", "F"]
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_SUBSET_SEED = 20260810
TRAIN_CLASS_LIMITS = {"N": 2000, "S": None, "V": 2000, "F": None}

BATCH_SIZE = 64
N_EPOCHS = 25
PATIENCE = 7
MIN_DELTA = 1e-4
NUM_WORKERS = 2
N_JOBS_PREPROCESSING = max(1, min(3, (os.cpu_count() or 2) - 1))

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

CONFIRM_METRICS = [
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


def candidate_signature(model_id, params):
    return model_id + "|" + json.dumps(params, sort_keys=True, separators=(",", ":"))


def find_candidate_bundle():
    manifest_paths = []
    if os.path.exists(CANDIDATE_SEARCH_ROOT):
        for root, _, files in os.walk(CANDIDATE_SEARCH_ROOT):
            for filename in files:
                if filename.startswith("architecture_candidate_repair_manifest") and filename.endswith(".json"):
                    manifest_paths.append(os.path.join(root, filename))

    valid = []
    for manifest_path in manifest_paths:
        try:
            with open(manifest_path, "r", encoding="utf-8") as file:
                manifest = json.load(file)
            if manifest.get("protocol") != "CNN2D unique-config repair and corrected candidate reselection":
                continue
            bundle_dir = os.path.dirname(manifest_path)
            candidate_files = [
                os.path.join(bundle_dir, filename)
                for filename in os.listdir(bundle_dir)
                if (
                    filename.startswith("candidate_rare_priority_")
                    or filename.startswith("candidate_balanced_minrare_")
                ) and filename.endswith(".json")
            ]
            payloads = []
            for path in candidate_files:
                with open(path, "r", encoding="utf-8") as file:
                    payload = json.load(file)
                if payload.get("role") in {
                    "rare_priority", "balanced_minrare_all_unique"
                }:
                    payloads.append((path, payload))
            unique_count = len({
                candidate_signature(payload["model_id"], payload["params"])
                for _, payload in payloads
            })
            valid.append((unique_count, len(payloads), manifest_path, payloads, manifest))
        except Exception:
            pass

    if not valid:
        raise FileNotFoundError(
            "Nie znaleziono poprawnego architecture_candidate_repair_manifest.json "
            "z plikami candidate_*.json. Podlacz wynik etapu naprawy jako Input."
        )

    unique_count, _, manifest_path, payloads, manifest = max(
        valid, key=lambda item: (item[0], item[1])
    )
    if unique_count != EXPECTED_UNIQUE_CANDIDATES:
        raise RuntimeError(
            f"Pakiet {manifest_path} zawiera {unique_count} unikalnych kandydatow; "
            f"oczekiwano {EXPECTED_UNIQUE_CANDIDATES}."
        )

    return manifest_path, payloads, manifest


def consolidate_candidates(payloads):
    grouped = {}
    model_role_coverage = {model_id: set() for model_id in ALL_MODELS}

    for source_path, payload in payloads:
        model_id = payload["model_id"]
        if model_id not in ALL_MODELS:
            raise ValueError(f"Nieznany model w {source_path}: {model_id}")
        role = payload["role"]
        model_role_coverage[model_id].add(role)

        if payload.get("raw_window") != {
            "before": EXPECTED_WINDOW_BEFORE,
            "after": EXPECTED_WINDOW_AFTER,
            "length": EXPECTED_SEGMENT_LENGTH,
        }:
            raise ValueError(f"{source_path}: niezgodne okno RAW.")
        if MODEL_DOMAIN[model_id] == "STFT_2D" and payload.get("stft_config") != STFT_CONFIG:
            raise ValueError(f"{source_path}: niezgodna konfiguracja STFT.")

        signature = candidate_signature(model_id, payload["params"])
        if signature not in grouped:
            grouped[signature] = {
                "model_id": model_id,
                "domain": MODEL_DOMAIN[model_id],
                "trial_number": int(payload["trial_number"]),
                "roles": set(),
                "params": dict(payload["params"]),
                "n_parameters": int(payload["n_parameters"]),
                "source_files": [],
                "screen_metrics": {},
            }
        row = grouped[signature]
        if row["trial_number"] != int(payload["trial_number"]):
            raise ValueError(f"Ta sama konfiguracja ma rozne numery triali: {source_path}")
        if row["n_parameters"] != int(payload["n_parameters"]):
            raise ValueError(f"Ta sama konfiguracja ma rozna liczbe parametrow: {source_path}")
        row["roles"].add(role)
        row["source_files"].append(source_path)
        row["screen_metrics"][role] = {
            "Rare_Macro_SF_mean": float(payload["Rare_Macro_SF_mean"]),
            "Macro_F1_mean": float(payload["Macro_F1_mean"]),
            "Min_Rare_F1_mean": float(payload["Min_Rare_F1_mean"]),
        }

    missing_roles = {
        model_id: sorted({"rare_priority", "balanced_minrare_all_unique"} - roles)
        for model_id, roles in model_role_coverage.items()
        if roles != {"rare_priority", "balanced_minrare_all_unique"}
    }
    if missing_roles:
        raise RuntimeError(f"Niepelne role kandydatow: {missing_roles}")

    candidates = []
    model_order = {model_id: idx for idx, model_id in enumerate(ALL_MODELS)}
    for row in grouped.values():
        roles = sorted(row["roles"])
        if len(roles) == 2:
            role_tag = "BOTH"
        elif roles == ["rare_priority"]:
            role_tag = "RARE"
        else:
            role_tag = "BALANCED"
        row["roles"] = roles
        row["candidate_id"] = (
            f"{row['model_id']}_T{row['trial_number']}_{role_tag}"
        )
        candidates.append(row)

    candidates.sort(
        key=lambda row: (
            0 if row["domain"] == "RAW_1D" else 1,
            model_order[row["model_id"]],
            row["trial_number"],
        )
    )
    if len(candidates) != EXPECTED_UNIQUE_CANDIDATES:
        raise RuntimeError(
            f"Po deduplikacji uzyskano {len(candidates)} kandydatow; "
            f"oczekiwano {EXPECTED_UNIQUE_CANDIDATES}."
        )
    if len({row["candidate_id"] for row in candidates}) != len(candidates):
        raise RuntimeError("Identyfikatory kandydatow nie sa unikalne.")
    return candidates


def build_candidate_model(model_id, params):
    params = dict(params)
    params.pop("learning_rate")
    params.pop("weight_decay")
    common = {
        "rr_dim": int(params.pop("rr_dim")),
        "head_dim": int(params.pop("head_dim")),
        "dropout": float(params.pop("dropout")),
    }

    if model_id == "CNN1D":
        model = TunableCNN1D(
            base_channels=int(params.pop("base_channels")),
            blocks_per_stage=int(params.pop("blocks_per_stage")),
            kernel_size=int(params.pop("kernel_size")),
            pool_bins=int(params.pop("pool_bins")),
            activation=params.pop("activation"),
            **common,
        )
    elif model_id == "BILSTM1D":
        n_layers = int(params.pop("n_layers"))
        model = TunableBiLSTM1D(
            hidden_size=int(params.pop("hidden_size")),
            n_layers=n_layers,
            bidirectional=True,
            lstm_dropout=float(params.pop("lstm_dropout", 0.0)),
            **common,
        )
    elif model_id == "TRANSFORMER1D":
        pair = params.pop("model_head_pair")
        d_model, n_heads = [int(value) for value in pair.split("x")]
        model = TunableTransformer1D(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=int(params.pop("n_layers")),
            ff_multiplier=int(params.pop("ff_multiplier")),
            transformer_dropout=float(params.pop("transformer_dropout")),
            **common,
        )
    elif model_id == "CNN2D":
        model = TunableCNN2D(
            base_channels=int(params.pop("base_channels")),
            blocks_per_stage=int(params.pop("blocks_per_stage")),
            kernel_size=int(params.pop("kernel_size")),
            pool_bins=int(params.pop("pool_bins")),
            activation=params.pop("activation"),
            **common,
        )
    elif model_id == "BILSTM2D":
        n_layers = int(params.pop("n_layers"))
        model = TunableBiLSTM2D(
            input_projection=int(params.pop("input_projection")),
            hidden_size=int(params.pop("hidden_size")),
            n_layers=n_layers,
            bidirectional=True,
            pooling=params.pop("pooling"),
            lstm_dropout=float(params.pop("lstm_dropout", 0.0)),
            **common,
        )
    elif model_id == "VIT2D":
        pair = params.pop("model_head_pair")
        d_model, n_heads = [int(value) for value in pair.split("x")]
        model = TunableViT2D(
            patch_size=int(params.pop("patch_size")),
            d_model=d_model,
            n_heads=n_heads,
            n_layers=int(params.pop("n_layers")),
            ff_multiplier=int(params.pop("ff_multiplier")),
            transformer_dropout=float(params.pop("transformer_dropout")),
            **common,
        )
    elif model_id == "CNN_BILSTM1D":
        n_layers = int(params.pop("n_layers"))
        model = TunableCNNBiLSTM1D(
            base_channels=int(params.pop("base_channels")),
            blocks_per_stage=int(params.pop("blocks_per_stage")),
            kernel_size=int(params.pop("kernel_size")),
            activation=params.pop("activation"),
            hidden_size=int(params.pop("hidden_size")),
            n_layers=n_layers,
            pooling=params.pop("pooling"),
            lstm_dropout=float(params.pop("lstm_dropout", 0.0)),
            **common,
        )
    elif model_id == "CNN_BILSTM2D":
        n_layers = int(params.pop("n_layers"))
        model = TunableCNNBiLSTM2D(
            base_channels=int(params.pop("base_channels")),
            blocks_per_stage=int(params.pop("blocks_per_stage")),
            kernel_size=int(params.pop("kernel_size")),
            activation=params.pop("activation"),
            hidden_size=int(params.pop("hidden_size")),
            n_layers=n_layers,
            pooling=params.pop("pooling"),
            lstm_dropout=float(params.pop("lstm_dropout", 0.0)),
            **common,
        )
    else:
        raise ValueError(f"Nieznany model: {model_id}")

    if params:
        raise ValueError(f"Niewykorzystane parametry {model_id}: {params}")
    return model


def evaluate_confirmation_model(model, loader):
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
    metrics = {
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
    matrix = confusion_matrix(labels, predictions, labels=[0, 1, 2, 3])
    return metrics, matrix


def train_confirmation_seed(candidate, train_dataset, val_dataset, seed):
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

    model = build_candidate_model(candidate["model_id"], candidate["params"])
    n_parameters = count_parameters(model)
    if n_parameters != candidate["n_parameters"]:
        raise RuntimeError(
            f"{candidate['candidate_id']}: oczekiwano {candidate['n_parameters']} "
            f"parametrow, zbudowano {n_parameters}."
        )
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(candidate["params"]["learning_rate"]),
        weight_decay=float(candidate["params"]["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()

    best_metrics = None
    best_matrix = None
    best_macro = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    started = time.time()

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

        metrics, matrix = evaluate_confirmation_model(model, val_loader)
        if metrics["Macro_F1"] > best_macro + MIN_DELTA:
            best_macro = metrics["Macro_F1"]
            best_metrics = dict(metrics)
            best_matrix = matrix.copy()
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            break

    result = {
        "candidate_id": candidate["candidate_id"],
        "model_id": candidate["model_id"],
        "domain": candidate["domain"],
        "trial_number": candidate["trial_number"],
        "roles": ";".join(candidate["roles"]),
        "seed": int(seed),
        "n_parameters": int(n_parameters),
        "best_epoch": int(best_epoch),
        "time_s": float(time.time() - started),
        "confusion_json": json.dumps(best_matrix.astype(int).tolist()),
    }
    result.update(best_metrics)

    del model, optimizer, train_loader, val_loader
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return result


def find_best_progress_file():
    filename = "architecture_confirmation_runs.csv"
    candidates = []
    direct = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(direct):
        candidates.append(direct)
    if os.path.exists(RESUME_SEARCH_ROOT):
        for root, _, files in os.walk(RESUME_SEARCH_ROOT):
            if filename in files:
                candidates.append(os.path.join(root, filename))
    valid = []
    for path in dict.fromkeys(candidates):
        try:
            frame = pd.read_csv(path)
            required = {"candidate_id", "seed", "Rare_Macro_SF", "Macro_F1"}
            if required.issubset(frame.columns):
                valid.append((len(frame), path, frame))
        except Exception:
            pass
    if not valid:
        return pd.DataFrame()
    _, path, frame = max(valid, key=lambda item: item[0])

    return frame


def t_confidence_interval(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean
    sem = float(values.std(ddof=1) / np.sqrt(len(values)))
    margin = float(student_t.ppf((1 + confidence) / 2, len(values) - 1) * sem)
    return mean - margin, mean + margin


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def make_summary(runs, candidates):
    candidate_map = {row["candidate_id"]: row for row in candidates}
    rows = []
    for candidate_id, group in runs.groupby("candidate_id"):
        candidate = candidate_map[candidate_id]
        row = {
            "candidate_id": candidate_id,
            "model_id": candidate["model_id"],
            "domain": candidate["domain"],
            "trial_number": candidate["trial_number"],
            "roles": ";".join(candidate["roles"]),
            "n_parameters": candidate["n_parameters"],
            "n_seeds": int(group["seed"].nunique()),
            "best_epoch_mean": float(group["best_epoch"].mean()),
            "time_s_mean": float(group["time_s"].mean()),
        }
        for metric in CONFIRM_METRICS:
            values = group[metric].astype(float).to_numpy()
            low, high = t_confidence_interval(values)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        ["Rare_Macro_SF_mean", "Min_Rare_F1_mean", "Macro_F1_mean"],
        ascending=False,
    ).reset_index(drop=True)
    summary.insert(0, "rank_rare_priority", np.arange(1, len(summary) + 1))
    return summary


def make_pairwise(runs, candidates):
    candidate_map = {row["candidate_id"]: row for row in candidates}
    rows = []
    for domain in ["RAW_1D", "STFT_2D"]:
        ids = [row["candidate_id"] for row in candidates if row["domain"] == domain]
        for candidate_a, candidate_b in itertools.combinations(ids, 2):
            frame_a = runs[runs["candidate_id"] == candidate_a].set_index("seed")
            frame_b = runs[runs["candidate_id"] == candidate_b].set_index("seed")
            common_seeds = sorted(set(frame_a.index) & set(frame_b.index))
            for metric in ["Rare_Macro_SF", "Min_Rare_F1", "Macro_F1"]:
                a = frame_a.loc[common_seeds, metric].astype(float).to_numpy()
                b = frame_b.loc[common_seeds, metric].astype(float).to_numpy()
                differences = a - b
                if len(differences) == 0 or np.allclose(differences, 0.0):
                    p_value = 1.0
                else:
                    try:
                        p_value = float(
                            wilcoxon(
                                differences,
                                zero_method="wilcox",
                                alternative="two-sided",
                                method="auto",
                            ).pvalue
                        )
                    except ValueError:
                        p_value = 1.0
                rows.append({
                    "domain": domain,
                    "metric": metric,
                    "candidate_a": candidate_a,
                    "model_a": candidate_map[candidate_a]["model_id"],
                    "candidate_b": candidate_b,
                    "model_b": candidate_map[candidate_b]["model_id"],
                    "n_paired_seeds": len(common_seeds),
                    "mean_a": float(a.mean()) if len(a) else np.nan,
                    "mean_b": float(b.mean()) if len(b) else np.nan,
                    "mean_difference_a_minus_b": float(differences.mean()) if len(a) else np.nan,
                    "median_difference_a_minus_b": float(np.median(differences)) if len(a) else np.nan,
                    "wins_a": int(np.sum(differences > 0)),
                    "ties": int(np.sum(np.isclose(differences, 0.0))),
                    "wins_b": int(np.sum(differences < 0)),
                    "wilcoxon_p_raw": p_value,
                })
    pairwise = pd.DataFrame(rows)
    if not pairwise.empty:
        pairwise["wilcoxon_p_holm"] = np.nan
        for (_, _), indices in pairwise.groupby(["domain", "metric"]).groups.items():
            indices = list(indices)
            pairwise.loc[indices, "wilcoxon_p_holm"] = holm_adjust(
                pairwise.loc[indices, "wilcoxon_p_raw"].to_numpy()
            )
        pairwise["significant_holm_0_05"] = pairwise["wilcoxon_p_holm"] < 0.05
    return pairwise


def make_pareto(summary):
    rows = []
    for _, candidate in summary.iterrows():
        dominated = False
        for _, other in summary.iterrows():
            if other["candidate_id"] == candidate["candidate_id"]:
                continue
            if (
                other["Rare_Macro_SF_mean"] >= candidate["Rare_Macro_SF_mean"]
                and other["Macro_F1_mean"] >= candidate["Macro_F1_mean"]
                and (
                    other["Rare_Macro_SF_mean"] > candidate["Rare_Macro_SF_mean"]
                    or other["Macro_F1_mean"] > candidate["Macro_F1_mean"]
                )
            ):
                dominated = True
                break
        if not dominated:
            rows.append(candidate.to_dict())
    return pd.DataFrame(rows)


def save_mean_confusions(runs):
    for candidate_id, group in runs.groupby("candidate_id"):
        matrices = np.stack([
            np.asarray(json.loads(value), dtype=float)
            for value in group["confusion_json"]
        ])
        mean_matrix = matrices.mean(axis=0)
        pd.DataFrame(
            mean_matrix,
            index=[f"true_{name}" for name in CLASS_NAMES],
            columns=[f"pred_{name}" for name in CLASS_NAMES],
        ).to_csv(os.path.join(OUTPUT_DIR, f"mean_confusion_{candidate_id}.csv"))


def save_confirmation_outputs(runs, candidates, candidate_manifest_path, candidate_manifest, elapsed_hours):
    runs = runs.sort_values(["candidate_id", "seed"]).reset_index(drop=True)
    runs.to_csv(os.path.join(OUTPUT_DIR, "architecture_confirmation_runs.csv"), index=False)
    summary = make_summary(runs, candidates)
    summary.to_csv(os.path.join(OUTPUT_DIR, "architecture_confirmation_summary.csv"), index=False)
    pairwise = make_pairwise(runs, candidates)
    pairwise.to_csv(os.path.join(OUTPUT_DIR, "architecture_confirmation_pairwise.csv"), index=False)
    pareto = make_pareto(summary)
    pareto.to_csv(os.path.join(OUTPUT_DIR, "architecture_confirmation_pareto.csv"), index=False)
    save_mean_confusions(runs)

    expected_runs = len(candidates) * len(CONFIRMATION_SEEDS)
    payload = {
        "protocol": "paired architecture confirmation on DS1",
        "candidate_manifest_source": candidate_manifest_path,
        "candidate_repair_protocol": candidate_manifest.get("protocol"),
        "raw_window": {"before": 65, "after": 110, "length": 175},
        "stft_config": STFT_CONFIG,
        "rr_features_used": True,
        "train_subset_seed": TRAIN_SUBSET_SEED,
        "train_class_limits": TRAIN_CLASS_LIMITS,
        "validation": "full DS1 VAL",
        "confirmation_seeds": CONFIRMATION_SEEDS,
        "n_candidates": len(candidates),
        "expected_runs": expected_runs,
        "completed_runs": int(len(runs)),
        "complete": bool(len(runs) == expected_runs),
        "n_epochs": N_EPOCHS,
        "patience": PATIENCE,
        "early_stopping_metric": "Macro F1",
        "loss": "unweighted CrossEntropyLoss",
        "augmentation": False,
        "oversampling": False,
        "determinism": {
            "cudnn_deterministic": True,
            "tf32": False,
            "torch_deterministic_algorithms_warn_only": True,
        },
        "candidate_selection_after_confirmation": {
            "primary": "Rare_Macro_SF mean",
            "safeguard": "Min_Rare_F1 mean",
            "secondary": "Macro_F1 mean",
        },
        "pairwise_tests": "Wilcoxon paired by seed, Holm correction within domain and metric",
        "candidates": [
            {
                "candidate_id": row["candidate_id"],
                "model_id": row["model_id"],
                "domain": row["domain"],
                "trial_number": row["trial_number"],
                "roles": row["roles"],
                "n_parameters": row["n_parameters"],
                "params": row["params"],
            }
            for row in candidates
        ],
        "elapsed_hours_this_session": elapsed_hours,
        "requires_ds2": False,
        "ds2_used": False,
    }
    with open(
        os.path.join(OUTPUT_DIR, "architecture_confirmation_manifest.json"),
        "w", encoding="utf-8",
    ) as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return summary


def main():
    invalid = set(ALL_MODELS) - set(MODEL_DOMAIN)
    if invalid:
        raise ValueError(f"Nieznane modele: {sorted(invalid)}")
    if DEVICE.type != "cuda":
        raise RuntimeError("Confirmation wymaga akceleratora GPU w Kaggle.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session_start = time.time()
    deadline = session_start + MAX_RUNTIME_HOURS * 3600
    candidate_manifest_path, candidate_payloads, candidate_manifest = find_candidate_bundle()
    candidates = consolidate_candidates(candidate_payloads)

    for row in candidates:
        model = build_candidate_model(row["model_id"], row["params"])
        n_parameters = count_parameters(model)
        del model
        if n_parameters != row["n_parameters"]:
            raise RuntimeError(
                f"{row['candidate_id']}: zapisano {row['n_parameters']}, "
                f"zbudowano {n_parameters} parametrow."
            )

    train_path, val_path = resolve_data_paths()

    (
        train_signals, train_rr, train_labels,
        val_signals, val_rr, val_labels,
    ) = load_data(train_path, val_path)
    raw_train_dataset = RawRRDataset(train_signals, train_rr, train_labels)
    raw_val_dataset = RawRRDataset(val_signals, val_rr, val_labels)

    train_images = process_to_stft_images(train_signals)
    val_images = process_to_stft_images(val_signals)

    image_train_dataset = ImageRRDataset(train_images, train_rr, train_labels)
    image_val_dataset = ImageRRDataset(val_images, val_rr, val_labels)

    runs = find_best_progress_file()
    valid_candidate_ids = {row["candidate_id"] for row in candidates}
    if not runs.empty:
        unknown_candidates = set(runs["candidate_id"]) - valid_candidate_ids
        unknown_seeds = set(runs["seed"].astype(int)) - set(CONFIRMATION_SEEDS)
        if unknown_candidates or unknown_seeds:
            raise RuntimeError(
                f"Niezgodny plik wznowienia: candidates={unknown_candidates}, "
                f"seeds={unknown_seeds}"
            )
        runs = runs.drop_duplicates(["candidate_id", "seed"], keep="last")

    completed_keys = set()
    if not runs.empty:
        completed_keys = {
            (str(row.candidate_id), int(row.seed))
            for row in runs[["candidate_id", "seed"]].itertuples(index=False)
        }

    stopped_by_time = False
    for candidate in candidates:
        if candidate["domain"] == "RAW_1D":
            train_dataset, val_dataset = raw_train_dataset, raw_val_dataset
        else:
            train_dataset, val_dataset = image_train_dataset, image_val_dataset

        for seed in CONFIRMATION_SEEDS:
            key = (candidate["candidate_id"], int(seed))
            if key in completed_keys:
                continue
            if time.time() >= deadline:
                stopped_by_time = True
                break

            result = train_confirmation_seed(candidate, train_dataset, val_dataset, seed)

            runs = pd.concat([runs, pd.DataFrame([result])], ignore_index=True)
            runs = runs.drop_duplicates(["candidate_id", "seed"], keep="last")
            runs.to_csv(
                os.path.join(OUTPUT_DIR, "architecture_confirmation_runs.csv"),
                index=False,
            )
            completed_keys.add(key)
        if stopped_by_time:
            break

    elapsed_hours = (time.time() - session_start) / 3600.0
    save_confirmation_outputs(
        runs, candidates, candidate_manifest_path, candidate_manifest, elapsed_hours
    )


if __name__ == "__main__":
    main()
