import gc
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
import torch.nn.functional as F
from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed
from scipy.signal import spectrogram
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)

warnings.filterwarnings("ignore", category=UserWarning)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)
cv2.setNumThreads(1)

OUTPUT_DIR = "/kaggle/working/balancing_screen_bilstm_raw_stft_65x110"
ARCHITECTURE_SEARCH_ROOT = "/kaggle/input"
RESUME_SEARCH_ROOT = "/kaggle/input"

DATA_DIR_CANDIDATES = [
    "/kaggle/working/datasetostrrfixed_65x110",
    "/kaggle/input/datasetostrrfixed-65x110",
    "/kaggle/input/datasetostrrfixed_65x110",
    "/kaggle/input/datasets/patrykc01/datasetostrrfixed-65x110",
    "/kaggle/input/datasets/patrykc01/datasetostrrfixed_65x110",
]

REPRESENTATIONS_TO_RUN = ["STFT_2D", "RAW_1D"]
SCREEN_SEEDS = [5000, 5001]
MAX_RUNTIME_HOURS = 10.25

FS = 360
EXPECTED_SEGMENT_LENGTH = 175
EXPECTED_WINDOW_BEFORE = 65
EXPECTED_WINDOW_AFTER = 110

CLASS_NAMES = ["N", "S", "V", "F"]
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
RARE_CLASS_INDICES = (1, 3)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
N_EPOCHS = 35
PATIENCE = 10
MIN_DELTA = 1e-4
NUM_WORKERS = 2
N_JOBS_PREPROCESSING = max(1, min(3, (os.cpu_count() or 2) - 1))

AUGMENT_DATA_SEED = 20260811
SMOTE_DATA_SEED = 20260812
WGAN_DATA_SEED = 20260813

WGAN_LATENT_DIM = 128
WGAN_BASE_CHANNELS = 128
WGAN_EPOCHS = 150
WGAN_BATCH_SIZE = 64
WGAN_CRITIC_STEPS = 5
WGAN_LAMBDA_GP = 10.0
WGAN_DRIFT = 1e-3
WGAN_LR = 1e-4
WGAN_MAX_FACTOR = 5
WGAN_CANDIDATE_MULTIPLIER = 3

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

EXPECTED_ARCHITECTURES = {
    "RAW_1D": {
        "candidate_id": "BILSTM1D_T24_BOTH",
        "model_id": "BILSTM1D",
        "n_parameters": 328_228,
    },
    "STFT_2D": {
        "candidate_id": "BILSTM2D_T7_BOTH",
        "model_id": "BILSTM2D",
        "n_parameters": 318_885,
    },
}


def make_method_configs():
    configs = [{
        "config_id": "BASELINE_CE",
        "family": "baseline",
        "kind": "baseline",
    }]

    for power in [0.25, 0.50, 0.75, 1.00]:
        configs.append({
            "config_id": f"CLASS_WEIGHT_P{int(power * 100):03d}",
            "family": "class_weight",
            "kind": "class_weight",
            "weight_power": power,
        })

    for alpha_power in [0.25, 0.50]:
        for gamma in [1.0, 2.0, 3.0]:
            configs.append({
                "config_id": (
                    f"FOCAL_A{int(alpha_power * 100):03d}_G{int(gamma)}"
                ),
                "family": "focal",
                "kind": "focal",
                "alpha_power": alpha_power,
                "gamma": gamma,
            })

    for power in [0.50, 0.75, 1.00]:
        configs.append({
            "config_id": f"SAMPLER_P{int(power * 100):03d}",
            "family": "sampler",
            "kind": "sampler",
            "sampler_power": power,
        })

    for multiplier in [1, 3]:
        configs.append({
            "config_id": f"AUGMENT_X{multiplier}",
            "family": "augmentation",
            "kind": "augmentation",
            "augment_multiplier": multiplier,
        })

    for factor in [3, 5]:
        configs.append({
            "config_id": f"SMOTE_X{factor}",
            "family": "smote",
            "kind": "smote",
            "smote_factor": factor,
        })

    for factor in [3, 5]:
        configs.append({
            "config_id": f"WGAN_GP_X{factor}",
            "family": "wgan_gp",
            "kind": "wgan_gp",
            "wgan_factor": factor,
        })

    if len(configs) != 20:
        raise RuntimeError(f"Oczekiwano 20 konfiguracji, utworzono {len(configs)}.")
    return configs

METHOD_CONFIGS = make_method_configs()
CONFIG_BY_ID = {config["config_id"]: config for config in METHOD_CONFIGS}


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

    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if "mitbih_train.npz" in files and "mitbih_val.npz" in files:
                return (
                    os.path.join(root, "mitbih_train.npz"),
                    os.path.join(root, "mitbih_val.npz"),
                )

    raise FileNotFoundError(
        "Nie znaleziono mitbih_train.npz i mitbih_val.npz. "
        "Podlacz dataset 65x110 jako Kaggle Input."
    )


def validate_npz(data, split_name):
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
    if "WINDOW_BEFORE" in data and int(data["WINDOW_BEFORE"]) != 65:
        raise ValueError(f"{split_name}: niezgodne WINDOW_BEFORE.")
    if "WINDOW_AFTER" in data and int(data["WINDOW_AFTER"]) != 110:
        raise ValueError(f"{split_name}: niezgodne WINDOW_AFTER.")
    if "FS" in data and int(data["FS"]) != FS:
        raise ValueError(f"{split_name}: niezgodne FS.")
    if not np.isfinite(signals).all() or not np.isfinite(rr).all():
        raise ValueError(f"{split_name}: X lub RR zawiera NaN/Inf.")


def standardize_segments(signals):
    signals = signals.astype(np.float32, copy=False)
    means = signals.mean(axis=1, keepdims=True)
    stds = np.maximum(signals.std(axis=1, keepdims=True), 1e-6)
    return ((signals - means) / stds).astype(np.float32, copy=False)


def load_full_split(path, split_name):
    with np.load(path, allow_pickle=False) as data:
        validate_npz(data, split_name)
        labels_raw = np.asarray(data["Y"]).astype(str)
        unknown = sorted(set(labels_raw) - set(CLASS_NAMES) - {"Q"})
        if unknown:
            raise ValueError(f"{split_name}: nieznane klasy {unknown}.")
        mask = labels_raw != "Q"
        signals = standardize_segments(data["X"][mask])
        rr = data["RR"][mask].astype(np.float32, copy=False)
        labels = np.asarray(
            [LABEL_MAP[label] for label in labels_raw[mask]], dtype=np.int64
        )
    return signals, rr, labels


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


def dataset_for_representation(representation, signals, images, rr, labels):
    if representation == "RAW_1D":
        return RawRRDataset(signals, rr, labels)
    if representation == "STFT_2D":
        if images is None:
            images = process_to_stft_images(signals)
        return ImageRRDataset(images, rr, labels)
    raise ValueError(f"Nieznana reprezentacja: {representation}")


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


class TunableBiLSTM1D(nn.Module):
    def __init__(
        self, hidden_size, n_layers, bidirectional, lstm_dropout,
        rr_dim, head_dim, dropout,
    ):
        super().__init__()
        self.bidirectional = bidirectional
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


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def find_architecture_candidates():
    paths = []
    if os.path.exists(ARCHITECTURE_SEARCH_ROOT):
        for root, _, files in os.walk(ARCHITECTURE_SEARCH_ROOT):
            if "architecture_confirmation_manifest.json" in files:
                paths.append(os.path.join(root, "architecture_confirmation_manifest.json"))

    valid = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as file:
                manifest = json.load(file)
            if manifest.get("protocol") != "paired architecture confirmation on DS1":
                continue
            if not manifest.get("complete") or manifest.get("ds2_used"):
                continue
            by_id = {row["candidate_id"]: row for row in manifest.get("candidates", [])}
            required_ids = {
                value["candidate_id"] for value in EXPECTED_ARCHITECTURES.values()
            }
            if not required_ids.issubset(by_id):
                continue
            valid.append((int(manifest.get("completed_runs", 0)), path, manifest, by_id))
        except Exception as error:
            pass

    if not valid:
        raise FileNotFoundError(
            "Nie znaleziono kompletnego architecture_confirmation_manifest.json. "
            "Podlacz output etapu potwierdzenia architektur jako Kaggle Input."
        )

    _, path, manifest, by_id = max(valid, key=lambda item: item[0])
    selected = {}
    for representation, expected in EXPECTED_ARCHITECTURES.items():
        row = by_id[expected["candidate_id"]]
        if row["model_id"] != expected["model_id"]:
            raise ValueError(f"{path}: niezgodny model dla {row['candidate_id']}.")
        if int(row["n_parameters"]) != expected["n_parameters"]:
            raise ValueError(f"{path}: niezgodna liczba parametrow {row['candidate_id']}.")
        selected[representation] = row

    if manifest.get("raw_window") != {"before": 65, "after": 110, "length": 175}:
        raise ValueError(f"{path}: niezgodne okno RAW.")
    if manifest.get("stft_config") != STFT_CONFIG:
        raise ValueError(f"{path}: niezgodne STFT T29.")

    return path, manifest, selected


def build_model(representation, architecture):
    params = dict(architecture["params"])
    params.pop("learning_rate")
    params.pop("weight_decay")
    common = {
        "rr_dim": int(params.pop("rr_dim")),
        "head_dim": int(params.pop("head_dim")),
        "dropout": float(params.pop("dropout")),
    }

    if representation == "RAW_1D":
        n_layers = int(params.pop("n_layers"))
        model = TunableBiLSTM1D(
            hidden_size=int(params.pop("hidden_size")),
            n_layers=n_layers,
            bidirectional=True,
            lstm_dropout=float(params.pop("lstm_dropout", 0.0)),
            **common,
        )
    elif representation == "STFT_2D":
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
    else:
        raise ValueError(f"Nieznana reprezentacja: {representation}")

    if params:
        raise ValueError(f"Niewykorzystane parametry {representation}: {params}")
    return model


def class_weights(labels, power):
    counts = np.bincount(labels, minlength=4).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Brak klasy w TRAIN: {counts.tolist()}")
    weights = np.power(counts, -float(power))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = float(gamma)

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        probability_true = torch.exp(-ce)
        loss = self.alpha[targets] * (1.0 - probability_true).pow(self.gamma) * ce
        return loss.mean()


class WGANGenerator1D(nn.Module):

    def __init__(self, latent_dim=WGAN_LATENT_DIM, base_channels=WGAN_BASE_CHANNELS):
        super().__init__()
        self.base_channels = int(base_channels)
        self.project = nn.Linear(latent_dim, self.base_channels * 22)
        self.network = nn.Sequential(
            nn.ConvTranspose1d(
                self.base_channels, self.base_channels, 4, stride=2, padding=1,
                bias=False,
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
        features = self.project(latent).view(
            latent.size(0), self.base_channels, 22
        )
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
        features = self.network(signal)
        return self.output(features.flatten(1)).view(-1)


def shift_without_wrap(signal, shift):
    result = np.empty_like(signal)
    if shift > 0:
        result[:shift] = signal[0]
        result[shift:] = signal[:-shift]
    elif shift < 0:
        amount = -shift
        result[-amount:] = signal[-1]
        result[:-amount] = signal[amount:]
    else:
        result[:] = signal
    return result


def augment_one_signal(signal, rng):
    augmented = signal.astype(np.float32, copy=True)
    shift = int(rng.integers(-5, 6))
    augmented = shift_without_wrap(augmented, shift)
    augmented += rng.normal(0.0, 0.025, size=augmented.shape).astype(np.float32)

    time_axis = np.arange(len(augmented), dtype=np.float32) / FS
    frequency = float(rng.uniform(0.5, 2.0))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    amplitude = float(rng.uniform(0.0, 0.035))
    augmented += amplitude * np.sin(2.0 * np.pi * frequency * time_axis + phase)
    return standardize_segments(augmented[None, :])[0]


def make_augmented_extras(signals, rr, labels, multiplier):
    rare_indices = np.flatnonzero(np.isin(labels, RARE_CLASS_INDICES))
    extra_signals, extra_rr, extra_labels = [], [], []
    for copy_index in range(multiplier):
        rng = np.random.default_rng(AUGMENT_DATA_SEED + copy_index)
        for index in rare_indices:
            extra_signals.append(augment_one_signal(signals[index], rng))
            extra_rr.append(rr[index])
            extra_labels.append(labels[index])
    return (
        np.asarray(extra_signals, dtype=np.float32),
        np.asarray(extra_rr, dtype=np.float32),
        np.asarray(extra_labels, dtype=np.int64),
    )


def make_smote_extras(signals, rr, labels, factor):
    counts = np.bincount(labels, minlength=4)
    strategy = {
        class_index: int(counts[class_index] * factor)
        for class_index in RARE_CLASS_INDICES
    }
    features = np.hstack([signals, rr]).astype(np.float32, copy=False)
    smote = SMOTE(
        sampling_strategy=strategy,
        random_state=SMOTE_DATA_SEED + int(factor),
        k_neighbors=5,
    )
    resampled_features, resampled_labels = smote.fit_resample(features, labels)
    n_original = len(labels)
    if not np.array_equal(resampled_labels[:n_original], labels):
        raise RuntimeError("SMOTE zmienilo kolejnosc oryginalnych obserwacji.")
    extras = resampled_features[n_original:]
    extra_signals = standardize_segments(extras[:, :EXPECTED_SEGMENT_LENGTH])
    extra_rr = extras[:, EXPECTED_SEGMENT_LENGTH:].astype(np.float32, copy=False)
    extra_labels = resampled_labels[n_original:].astype(np.int64, copy=False)
    return extra_signals, extra_rr, extra_labels


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


def train_wgan_for_class(real_signals, class_index):
    class_name = CLASS_NAMES[class_index]
    seed = WGAN_DATA_SEED + int(class_index)
    set_deterministic(seed)

    robust_scale = float(np.percentile(np.abs(real_signals), 99.5))
    robust_scale = max(robust_scale, 1e-6)
    scaled = np.clip(real_signals / robust_scale, -1.0, 1.0).astype(np.float32)
    tensor_dataset = torch.from_numpy(scaled).unsqueeze(1)
    loader = DataLoader(
        tensor_dataset,
        batch_size=min(WGAN_BATCH_SIZE, len(tensor_dataset)),
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )

    generator = WGANGenerator1D().to(DEVICE)
    critic = WGANCritic1D().to(DEVICE)
    optimizer_g = torch.optim.Adam(
        generator.parameters(), lr=WGAN_LR, betas=(0.0, 0.9)
    )
    optimizer_c = torch.optim.Adam(
        critic.parameters(), lr=WGAN_LR, betas=(0.0, 0.9)
    )

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
                    score_fake.mean()
                    - score_real.mean()
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

        epoch_row = {
            "epoch": epoch,
            "critic_loss": float(np.mean(critic_losses)),
            "generator_loss": float(np.mean(generator_losses)),
            "gradient_penalty": float(np.mean(penalties)),
        }
        history.append(epoch_row)

    checkpoint = {
        "class_index": int(class_index),
        "class_name": class_name,
        "seed": seed,
        "robust_scale": robust_scale,
        "generator_state_dict": {
            key: value.detach().cpu() for key, value in generator.state_dict().items()
        },
        "generator_config": {
            "latent_dim": WGAN_LATENT_DIM,
            "base_channels": WGAN_BASE_CHANNELS,
        },
    }
    torch.save(
        checkpoint,
        os.path.join(OUTPUT_DIR, f"wgan_gp_generator_{class_name}.pt"),
    )
    pd.DataFrame(history).to_csv(
        os.path.join(OUTPUT_DIR, f"wgan_gp_training_{class_name}.csv"),
        index=False,
    )

    del critic, optimizer_g, optimizer_c, loader
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return generator, robust_scale


def generate_wgan_candidates(generator, robust_scale, n_candidates, seed):
    set_deterministic(seed)
    generator.eval()
    batches = []
    with torch.no_grad():
        for start in range(0, n_candidates, 256):
            batch_size = min(256, n_candidates - start)
            latent = torch.randn(batch_size, WGAN_LATENT_DIM, device=DEVICE)
            generated = generator(latent).squeeze(1).cpu().numpy()
            batches.append(generated)
    signals = np.concatenate(batches, axis=0) * robust_scale
    return standardize_segments(signals.astype(np.float32, copy=False))


def select_wgan_candidates(real_signals, real_rr, candidates, n_needed, class_index):
    if len(real_signals) < 3:
        raise ValueError(f"Za malo realnych probek klasy {CLASS_NAMES[class_index]}.")
    if len(candidates) < n_needed:
        raise ValueError("WGAN-GP wygenerowal za malo kandydatow.")

    real_nn_model = NearestNeighbors(n_neighbors=2, metric="euclidean", n_jobs=-1)
    real_nn_model.fit(real_signals)
    real_distances = real_nn_model.kneighbors(real_signals, return_distance=True)[0][:, 1]

    nearest_real_model = NearestNeighbors(
        n_neighbors=1, metric="euclidean", n_jobs=-1
    )
    nearest_real_model.fit(real_signals)
    candidate_distances, nearest_indices = nearest_real_model.kneighbors(
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

    log_distance_score = np.abs(
        np.log((candidate_distances + 1e-8) / (real_median + 1e-8))
    )
    penalty = np.where(acceptable, 0.0, 100.0)
    order = np.argsort(log_distance_score + penalty)
    selected_indices = order[:n_needed]
    selected = candidates[selected_indices]
    selected_distances = candidate_distances[selected_indices]
    selected_nearest = nearest_indices[selected_indices]
    selected_rr = real_rr[selected_nearest].astype(np.float32, copy=True)

    if len(selected) >= 3:
        generated_nn = NearestNeighbors(
            n_neighbors=2, metric="euclidean", n_jobs=-1
        ).fit(selected)
        generated_self_distances = generated_nn.kneighbors(
            selected, return_distance=True
        )[0][:, 1]
        generated_self_median = float(np.median(generated_self_distances))
    else:
        generated_self_median = 0.0

    quality = {
        "class_index": int(class_index),
        "class_name": CLASS_NAMES[class_index],
        "n_real": int(len(real_signals)),
        "n_candidates": int(len(candidates)),
        "n_selected": int(len(selected)),
        "candidate_acceptable_fraction": float(np.mean(acceptable)),
        "selected_acceptable_fraction": float(np.mean(acceptable[selected_indices])),
        "real_nn_median": real_median,
        "real_nn_q01": float(np.quantile(real_distances, 0.01)),
        "real_nn_q99": float(np.quantile(real_distances, 0.99)),
        "screen_lower_limit": lower_limit,
        "screen_upper_limit": upper_limit,
        "selected_nn_mean": float(np.mean(selected_distances)),
        "selected_nn_median": float(np.median(selected_distances)),
        "selected_nn_min": float(np.min(selected_distances)),
        "selected_nn_max": float(np.max(selected_distances)),
        "generated_self_nn_median": generated_self_median,
        "memorization_warning": bool(
            np.median(selected_distances) < 0.25 * real_median
        ),
        "mode_collapse_warning": bool(
            generated_self_median < 0.25 * real_median
        ),
    }
    return selected, selected_rr, quality


def write_wgan_reports(pool_signals, pool_labels, quality_rows):
    pd.DataFrame(quality_rows).to_csv(
        os.path.join(OUTPUT_DIR, "wgan_gp_quality_summary.csv"), index=False
    )
    example_rows = []
    for class_index in RARE_CLASS_INDICES:
        class_signals = pool_signals[pool_labels == class_index][:10]
        for sample_index, signal in enumerate(class_signals):
            row = {
                "class": CLASS_NAMES[class_index],
                "sample_index": sample_index,
            }
            row.update({f"t_{idx:03d}": float(value) for idx, value in enumerate(signal)})
            example_rows.append(row)
    pd.DataFrame(example_rows).to_csv(
        os.path.join(OUTPUT_DIR, "wgan_gp_synthetic_examples.csv"), index=False
    )


def find_existing_wgan_pool(train_labels):
    filename = "wgan_gp_synthetic_pool.npz"
    paths = []
    direct = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(direct):
        paths.append(direct)
    if os.path.exists(RESUME_SEARCH_ROOT):
        for root, _, files in os.walk(RESUME_SEARCH_ROOT):
            if filename in files:
                paths.append(os.path.join(root, filename))

    expected_counts = np.bincount(train_labels, minlength=4)
    valid = []
    for path in dict.fromkeys(paths):
        try:
            with np.load(path, allow_pickle=False) as data:
                required = {
                    "X", "RR", "Y", "WINDOW_LENGTH", "MAX_FACTOR",
                    "DATA_SEED", "SOURCE_CLASS_COUNTS", "QUALITY_JSON",
                }
                if not required.issubset(data.files):
                    continue
                signals = data["X"].astype(np.float32)
                rr = data["RR"].astype(np.float32)
                labels = data["Y"].astype(np.int64)
                if int(data["WINDOW_LENGTH"]) != EXPECTED_SEGMENT_LENGTH:
                    continue
                if int(data["MAX_FACTOR"]) != WGAN_MAX_FACTOR:
                    continue
                if int(data["DATA_SEED"]) != WGAN_DATA_SEED:
                    continue
                if not np.array_equal(data["SOURCE_CLASS_COUNTS"], expected_counts):
                    continue
                if not (len(signals) == len(rr) == len(labels)):
                    continue
                if signals.shape[1] != EXPECTED_SEGMENT_LENGTH or rr.shape[1] != 4:
                    continue
                if set(labels.tolist()) - set(RARE_CLASS_INDICES):
                    continue
                quality_rows = json.loads(str(data["QUALITY_JSON"]))
            valid.append((len(labels), path, signals, rr, labels, quality_rows))
        except Exception:
            continue

    if not valid:
        return None
    _, path, signals, rr, labels, quality_rows = max(valid, key=lambda item: item[0])

    write_wgan_reports(signals, labels, quality_rows)
    return signals, rr, labels, quality_rows


def build_or_load_wgan_pool(train_signals, train_rr, train_labels):
    existing = find_existing_wgan_pool(train_labels)
    if existing is not None:
        return existing

    pool_signals, pool_rr, pool_labels, quality_rows = [], [], [], []
    counts = np.bincount(train_labels, minlength=4)
    for class_index in RARE_CLASS_INDICES:
        class_mask = train_labels == class_index
        real_signals = train_signals[class_mask]
        real_rr = train_rr[class_mask]
        n_needed = int((WGAN_MAX_FACTOR - 1) * counts[class_index])
        n_candidates = max(
            n_needed * WGAN_CANDIDATE_MULTIPLIER,
            n_needed + 512,
        )

        generator, robust_scale = train_wgan_for_class(
            real_signals, class_index
        )
        candidates = generate_wgan_candidates(
            generator,
            robust_scale,
            n_candidates=n_candidates,
            seed=WGAN_DATA_SEED + 100 + int(class_index),
        )
        selected, selected_rr, quality = select_wgan_candidates(
            real_signals,
            real_rr,
            candidates,
            n_needed=n_needed,
            class_index=class_index,
        )
        quality["robust_scale"] = robust_scale
        pool_signals.append(selected)
        pool_rr.append(selected_rr)
        pool_labels.append(
            np.full(n_needed, class_index, dtype=np.int64)
        )
        quality_rows.append(quality)

        del generator, candidates
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    pool_signals = np.concatenate(pool_signals, axis=0).astype(np.float32)
    pool_rr = np.concatenate(pool_rr, axis=0).astype(np.float32)
    pool_labels = np.concatenate(pool_labels, axis=0).astype(np.int64)
    quality_json = json.dumps(quality_rows, ensure_ascii=False)

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "wgan_gp_synthetic_pool.npz"),
        X=pool_signals,
        RR=pool_rr,
        Y=pool_labels,
        WINDOW_LENGTH=np.asarray(EXPECTED_SEGMENT_LENGTH, dtype=np.int64),
        MAX_FACTOR=np.asarray(WGAN_MAX_FACTOR, dtype=np.int64),
        DATA_SEED=np.asarray(WGAN_DATA_SEED, dtype=np.int64),
        SOURCE_CLASS_COUNTS=np.bincount(train_labels, minlength=4).astype(np.int64),
        QUALITY_JSON=np.asarray(quality_json),
    )
    write_wgan_reports(pool_signals, pool_labels, quality_rows)
    return pool_signals, pool_rr, pool_labels, quality_rows


def get_wgan_extras(train_signals, train_rr, train_labels, factor, extra_cache):
    pool_key = ("wgan_gp_pool", WGAN_MAX_FACTOR)
    if pool_key not in extra_cache:
        extra_cache[pool_key] = build_or_load_wgan_pool(
            train_signals, train_rr, train_labels
        )
    pool_signals, pool_rr, pool_labels, _ = extra_cache[pool_key]

    counts = np.bincount(train_labels, minlength=4)
    selected_indices = []
    for class_index in RARE_CLASS_INDICES:
        class_indices = np.flatnonzero(pool_labels == class_index)
        n_needed = int((factor - 1) * counts[class_index])
        if len(class_indices) < n_needed:
            raise RuntimeError(
                f"Pula WGAN-GP ma za malo probek klasy {CLASS_NAMES[class_index]}."
            )
        selected_indices.extend(class_indices[:n_needed].tolist())
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    return (
        pool_signals[selected_indices],
        pool_rr[selected_indices],
        pool_labels[selected_indices],
    )


def get_extra_dataset(
    representation, config, train_signals, train_rr, train_labels, extra_cache,
):
    kind = config["kind"]
    if kind not in {"augmentation", "smote", "wgan_gp"}:
        return None, np.empty(0, dtype=np.int64)

    if kind == "augmentation":
        variant = int(config["augment_multiplier"])
        raw_key = ("augmentation", variant)
        if raw_key not in extra_cache:
            extra_cache[raw_key] = make_augmented_extras(
                train_signals, train_rr, train_labels, variant
            )
    elif kind == "smote":
        variant = int(config["smote_factor"])
        raw_key = ("smote", variant)
        if raw_key not in extra_cache:
            extra_cache[raw_key] = make_smote_extras(
                train_signals, train_rr, train_labels, variant
            )
    else:
        variant = int(config["wgan_factor"])
        raw_key = ("wgan_gp", variant)
        if raw_key not in extra_cache:
            extra_cache[raw_key] = get_wgan_extras(
                train_signals,
                train_rr,
                train_labels,
                variant,
                extra_cache,
            )

    extra_signals, extra_rr, extra_labels = extra_cache[raw_key]
    dataset_key = (representation,) + raw_key
    if dataset_key not in extra_cache:
        if representation == "RAW_1D":
            extra_cache[dataset_key] = RawRRDataset(
                extra_signals, extra_rr, extra_labels
            )
        else:

            extra_images = process_to_stft_images(extra_signals)
            extra_cache[dataset_key] = ImageRRDataset(
                extra_images, extra_rr, extra_labels
            )
    return extra_cache[dataset_key], extra_labels


def build_loss(config, original_train_labels):
    if config["kind"] == "class_weight":
        weights = class_weights(original_train_labels, config["weight_power"])
        return nn.CrossEntropyLoss(weight=weights.to(DEVICE))
    if config["kind"] == "focal":
        alpha = class_weights(original_train_labels, config["alpha_power"])
        return FocalLoss(alpha=alpha, gamma=config["gamma"]).to(DEVICE)
    return nn.CrossEntropyLoss()


def make_train_loader(dataset, config, labels_for_dataset, seed):
    generator = torch.Generator().manual_seed(seed)
    pin_memory = DEVICE.type == "cuda"
    common = {
        "dataset": dataset,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": pin_memory,
    }
    if config["kind"] == "sampler":
        counts = np.bincount(labels_for_dataset, minlength=4).astype(np.float64)
        sample_weights = np.power(counts[labels_for_dataset], -config["sampler_power"])
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(labels_for_dataset),
            replacement=True,
            generator=generator,
        )
        return DataLoader(sampler=sampler, shuffle=False, **common)
    return DataLoader(shuffle=True, generator=generator, **common)

METRIC_NAMES = [
    "Macro_F1",
    "Rare_Macro_SF",
    "Min_Rare_F1",
    "Rare_HMean_SF",
    "F1_N",
    "F1_S",
    "F1_V",
    "F1_F",
    "Precision_S",
    "Precision_F",
    "Recall_S",
    "Recall_F",
]


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
    f1_values = f1_score(
        labels, predictions, labels=[0, 1, 2, 3], average=None, zero_division=0
    )
    precision = precision_score(
        labels, predictions, labels=[0, 1, 2, 3], average=None, zero_division=0
    )
    recall = recall_score(
        labels, predictions, labels=[0, 1, 2, 3], average=None, zero_division=0
    )
    f1_s, f1_f = float(f1_values[1]), float(f1_values[3])
    rare_hmean = (
        2.0 * f1_s * f1_f / (f1_s + f1_f)
        if f1_s + f1_f > 0.0 else 0.0
    )
    metrics = {
        "Macro_F1": float(np.mean(f1_values)),
        "Rare_Macro_SF": float((f1_s + f1_f) / 2.0),
        "Min_Rare_F1": float(min(f1_s, f1_f)),
        "Rare_HMean_SF": float(rare_hmean),
        "F1_N": float(f1_values[0]),
        "F1_S": f1_s,
        "F1_V": float(f1_values[2]),
        "F1_F": f1_f,
        "Precision_S": float(precision[1]),
        "Precision_F": float(precision[3]),
        "Recall_S": float(recall[1]),
        "Recall_F": float(recall[3]),
    }
    matrix = confusion_matrix(labels, predictions, labels=[0, 1, 2, 3])
    return metrics, matrix


def is_better_checkpoint(metrics, best_metrics):
    if best_metrics is None:
        return True
    priorities = ["Min_Rare_F1", "Rare_Macro_SF", "Macro_F1"]
    for metric in priorities:
        difference = metrics[metric] - best_metrics[metric]
        if difference > MIN_DELTA:
            return True
        if difference < -MIN_DELTA:
            return False
    return False


def train_one_run(
    representation,
    architecture,
    config,
    train_dataset,
    labels_for_dataset,
    original_train_labels,
    val_dataset,
    seed,
):
    set_deterministic(seed)
    train_loader = make_train_loader(
        train_dataset, config, labels_for_dataset, seed
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
    )

    model = build_model(representation, architecture)
    n_parameters = count_parameters(model)
    if n_parameters != int(architecture["n_parameters"]):
        raise RuntimeError(
            f"{representation}: oczekiwano {architecture['n_parameters']} parametrow, "
            f"zbudowano {n_parameters}."
        )
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(architecture["params"]["learning_rate"]),
        weight_decay=float(architecture["params"]["weight_decay"]),
    )
    criterion = build_loss(config, original_train_labels)

    best_metrics = None
    best_matrix = None
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

        metrics, matrix = evaluate_model(model, val_loader)
        if is_better_checkpoint(metrics, best_metrics):
            best_metrics = dict(metrics)
            best_matrix = matrix.copy()
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            break

    result = {
        "representation": representation,
        "architecture_candidate_id": architecture["candidate_id"],
        "config_id": config["config_id"],
        "family": config["family"],
        "config_json": json.dumps(config, sort_keys=True),
        "seed": int(seed),
        "n_parameters": int(n_parameters),
        "n_train_samples": int(len(labels_for_dataset)),
        "train_class_counts_json": json.dumps(named_counts(labels_for_dataset)),
        "best_epoch": int(best_epoch),
        "time_s": float(time.time() - started),
        "confusion_json": json.dumps(best_matrix.astype(int).tolist()),
    }
    result.update(best_metrics)

    del model, optimizer, criterion, train_loader, val_loader
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return result


def find_best_progress_file():
    filename = "balancing_screen_runs.csv"
    paths = []
    direct = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(direct):
        paths.append(direct)
    if os.path.exists(RESUME_SEARCH_ROOT):
        for root, _, files in os.walk(RESUME_SEARCH_ROOT):
            if filename in files:
                paths.append(os.path.join(root, filename))

    valid = []
    for path in dict.fromkeys(paths):
        try:
            frame = pd.read_csv(path)
            required = {"representation", "config_id", "seed", "Min_Rare_F1"}
            if required.issubset(frame.columns):
                valid.append((len(frame), path, frame))
        except Exception:
            pass
    if not valid:
        return pd.DataFrame()
    _, path, frame = max(valid, key=lambda item: item[0])

    return frame


def make_summary(runs):
    rows = []
    group_columns = ["representation", "config_id", "family"]
    for keys, group in runs.groupby(group_columns):
        representation, config_id, family = keys
        row = {
            "representation": representation,
            "config_id": config_id,
            "family": family,
            "n_seeds": int(group["seed"].nunique()),
            "n_train_samples": int(group["n_train_samples"].iloc[0]),
            "n_parameters": int(group["n_parameters"].iloc[0]),
            "best_epoch_mean": float(group["best_epoch"].mean()),
            "time_s_mean": float(group["time_s"].mean()),
            "config_json": group["config_json"].iloc[0],
        }
        for metric in METRIC_NAMES:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(
        ["representation", "Min_Rare_F1_mean", "Rare_Macro_SF_mean", "Macro_F1_mean"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def choose_family_winners(summary):
    if summary.empty:
        return pd.DataFrame()
    winners = []
    for (_, _), group in summary.groupby(["representation", "family"]):
        winner = group.sort_values(
            ["Min_Rare_F1_mean", "Rare_Macro_SF_mean", "Macro_F1_mean"],
            ascending=False,
        ).iloc[0]
        winners.append(winner.to_dict())
    result = pd.DataFrame(winners)
    return result.sort_values(
        ["representation", "Min_Rare_F1_mean", "Rare_Macro_SF_mean"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def make_pareto(summary):
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for representation, group in summary.groupby("representation"):
        for _, candidate in group.iterrows():
            dominated = False
            for _, other in group.iterrows():
                if other["config_id"] == candidate["config_id"]:
                    continue
                non_worse = (
                    other["Min_Rare_F1_mean"] >= candidate["Min_Rare_F1_mean"]
                    and other["Rare_Macro_SF_mean"] >= candidate["Rare_Macro_SF_mean"]
                    and other["Macro_F1_mean"] >= candidate["Macro_F1_mean"]
                )
                strictly_better = (
                    other["Min_Rare_F1_mean"] > candidate["Min_Rare_F1_mean"]
                    or other["Rare_Macro_SF_mean"] > candidate["Rare_Macro_SF_mean"]
                    or other["Macro_F1_mean"] > candidate["Macro_F1_mean"]
                )
                if non_worse and strictly_better:
                    dominated = True
                    break
            if not dominated:
                rows.append(candidate.to_dict())
    return pd.DataFrame(rows)


def save_mean_confusions(runs):
    for (representation, config_id), group in runs.groupby(
        ["representation", "config_id"]
    ):
        matrices = np.stack([
            np.asarray(json.loads(value), dtype=float)
            for value in group["confusion_json"]
        ])
        mean_matrix = matrices.mean(axis=0)
        filename = f"mean_confusion_{representation}_{config_id}.csv"
        pd.DataFrame(
            mean_matrix,
            index=[f"true_{name}" for name in CLASS_NAMES],
            columns=[f"pred_{name}" for name in CLASS_NAMES],
        ).to_csv(os.path.join(OUTPUT_DIR, filename))


def save_outputs(
    runs, architecture_manifest_path, architecture_manifest, selected_architectures,
    elapsed_hours,
):
    runs = runs.sort_values(
        ["config_id", "representation", "seed"]
    ).reset_index(drop=True)
    runs.to_csv(os.path.join(OUTPUT_DIR, "balancing_screen_runs.csv"), index=False)

    summary = make_summary(runs)
    summary.to_csv(os.path.join(OUTPUT_DIR, "balancing_screen_summary.csv"), index=False)
    winners = choose_family_winners(summary)
    winners.to_csv(
        os.path.join(OUTPUT_DIR, "balancing_screen_family_winners.csv"), index=False
    )
    pareto = make_pareto(summary)
    pareto.to_csv(os.path.join(OUTPUT_DIR, "balancing_screen_pareto.csv"), index=False)
    save_mean_confusions(runs)

    expected_runs = (
        len(REPRESENTATIONS_TO_RUN) * len(METHOD_CONFIGS) * len(SCREEN_SEEDS)
    )
    manifest = {
        "protocol": "balancing method screen on full DS1 TRAIN and DS1 VAL",
        "screen_only": True,
        "architecture_manifest_source": architecture_manifest_path,
        "architecture_manifest_protocol": architecture_manifest.get("protocol"),
        "raw_window": {"before": 65, "after": 110, "length": 175},
        "stft_config": STFT_CONFIG,
        "representations": REPRESENTATIONS_TO_RUN,
        "architectures": selected_architectures,
        "screen_seeds": SCREEN_SEEDS,
        "n_method_configs": len(METHOD_CONFIGS),
        "method_configs": METHOD_CONFIGS,
        "expected_runs": expected_runs,
        "completed_runs": int(len(runs)),
        "complete": bool(len(runs) == expected_runs),
        "uses_full_ds1_train": True,
        "validation": "full DS1 VAL",
        "n_epochs": N_EPOCHS,
        "patience": PATIENCE,
        "checkpoint_selection": ["Min_Rare_F1", "Rare_Macro_SF", "Macro_F1"],
        "family_winner_selection": ["Min_Rare_F1 mean", "Rare_Macro_SF mean", "Macro_F1 mean"],
        "techniques_are_isolated": True,
        "combinations_deferred_to_next_stage": True,
        "augmentation": {
            "classes": ["S", "F"],
            "time_shift_samples": [-5, 5],
            "gaussian_noise_std": 0.025,
            "baseline_wander_frequency_hz": [0.5, 2.0],
            "baseline_wander_amplitude": [0.0, 0.035],
            "offline_and_deterministic": True,
        },
        "smote": {
            "space": "standardized RAW signal concatenated with RR",
            "train_only": True,
            "k_neighbors": 5,
        },
        "wgan_gp": {
            "space": "standardized RAW signal; the same synthetic pool is used by RAW and STFT",
            "classes": ["S", "F"],
            "class_specific_generators": True,
            "generator": "1D convolutional WGAN-GP",
            "epochs": WGAN_EPOCHS,
            "latent_dim": WGAN_LATENT_DIM,
            "critic_steps": WGAN_CRITIC_STEPS,
            "lambda_gp": WGAN_LAMBDA_GP,
            "data_seed": WGAN_DATA_SEED,
            "fixed_synthetic_pool_during_screen": True,
            "generator_variability_deferred_to_confirmation": True,
            "rr_assignment": "RR of nearest real beat from the same class",
            "quality_screen": "aligned Euclidean nearest-neighbour distance",
            "quality_files": [
                "wgan_gp_quality_summary.csv",
                "wgan_gp_synthetic_examples.csv",
                "wgan_gp_synthetic_pool.npz",
            ],
            "train_only": True,
            "combined_with_class_weights": False,
        },
        "requires_ds2": False,
        "ds2_used": False,
        "elapsed_hours_this_session": elapsed_hours,
    }
    with open(
        os.path.join(OUTPUT_DIR, "balancing_screen_manifest.json"),
        "w", encoding="utf-8",
    ) as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    return summary, winners


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("Ten screen wymaga akceleratora GPU w Kaggle.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session_start = time.time()
    deadline = session_start + MAX_RUNTIME_HOURS * 3600.0

    (
        architecture_manifest_path,
        architecture_manifest,
        selected_architectures,
    ) = find_architecture_candidates()

    for representation in REPRESENTATIONS_TO_RUN:
        architecture = selected_architectures[representation]
        model = build_model(representation, architecture)
        actual_parameters = count_parameters(model)
        del model
        if actual_parameters != int(architecture["n_parameters"]):
            raise RuntimeError(
                f"{representation}: zapisano {architecture['n_parameters']}, "
                f"zbudowano {actual_parameters}."
            )

    train_path, val_path = resolve_data_paths()
    train_signals, train_rr, train_labels = load_full_split(train_path, "DS1 TRAIN")
    val_signals, val_rr, val_labels = load_full_split(val_path, "DS1 VAL")

    base_datasets = {
        "RAW_1D": RawRRDataset(train_signals, train_rr, train_labels),
    }
    val_datasets = {
        "RAW_1D": RawRRDataset(val_signals, val_rr, val_labels),
    }

    train_images = process_to_stft_images(train_signals)
    val_images = process_to_stft_images(val_signals)

    base_datasets["STFT_2D"] = ImageRRDataset(
        train_images, train_rr, train_labels
    )
    val_datasets["STFT_2D"] = ImageRRDataset(
        val_images, val_rr, val_labels
    )

    extra_cache = {}
    runs = find_best_progress_file()
    valid_representations = set(REPRESENTATIONS_TO_RUN)
    valid_configs = set(CONFIG_BY_ID)
    valid_seeds = set(SCREEN_SEEDS)
    if not runs.empty:
        unknown_representations = set(runs["representation"]) - valid_representations
        unknown_configs = set(runs["config_id"]) - valid_configs
        unknown_seeds = set(runs["seed"].astype(int)) - valid_seeds
        if unknown_representations or unknown_configs or unknown_seeds:
            raise RuntimeError(
                "Niezgodny plik wznowienia: "
                f"representations={unknown_representations}, "
                f"configs={unknown_configs}, seeds={unknown_seeds}"
            )
        runs = runs.drop_duplicates(
            ["representation", "config_id", "seed"], keep="last"
        )

    completed_keys = set()
    if not runs.empty:
        completed_keys = {
            (str(row.representation), str(row.config_id), int(row.seed))
            for row in runs[["representation", "config_id", "seed"]].itertuples(index=False)
        }

    stopped_by_time = False
    for config in METHOD_CONFIGS:
        for representation in REPRESENTATIONS_TO_RUN:
            pending_seeds = [
                seed for seed in SCREEN_SEEDS
                if (representation, config["config_id"], int(seed)) not in completed_keys
            ]
            if not pending_seeds:
                continue
            if time.time() >= deadline:
                stopped_by_time = True
                break

            base_dataset = base_datasets[representation]
            extra_dataset, extra_labels = get_extra_dataset(
                representation,
                config,
                train_signals,
                train_rr,
                train_labels,
                extra_cache,
            )
            if extra_dataset is None:
                train_dataset = base_dataset
                labels_for_dataset = train_labels
            else:
                train_dataset = ConcatDataset([base_dataset, extra_dataset])
                labels_for_dataset = np.concatenate([train_labels, extra_labels])

            for seed in pending_seeds:
                key = (representation, config["config_id"], int(seed))
                if time.time() >= deadline:
                    stopped_by_time = True
                    break

                result = train_one_run(
                    representation=representation,
                    architecture=selected_architectures[representation],
                    config=config,
                    train_dataset=train_dataset,
                    labels_for_dataset=labels_for_dataset,
                    original_train_labels=train_labels,
                    val_dataset=val_datasets[representation],
                    seed=seed,
                )

                runs = pd.concat([runs, pd.DataFrame([result])], ignore_index=True)
                runs = runs.drop_duplicates(
                    ["representation", "config_id", "seed"], keep="last"
                )
                runs.to_csv(
                    os.path.join(OUTPUT_DIR, "balancing_screen_runs.csv"),
                    index=False,
                )
                completed_keys.add(key)

            if stopped_by_time:
                break
        if stopped_by_time:
            break

    elapsed_hours = (time.time() - session_start) / 3600.0
    save_outputs(
        runs,
        architecture_manifest_path,
        architecture_manifest,
        selected_architectures,
        elapsed_hours,
    )


if __name__ == "__main__":
    main()
