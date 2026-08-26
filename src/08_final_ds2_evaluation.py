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
from joblib import Parallel, delayed
from scipy.signal import spectrogram
from scipy.stats import t as student_t
from scipy.stats import wilcoxon
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
)


warnings.filterwarnings("ignore", category=UserWarning)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)
cv2.setNumThreads(1)


OUTPUT_DIR = "/kaggle/working/final_ds2_locked_raw_stft_65x110"
ARCHITECTURE_SEARCH_ROOT = "/kaggle/input"
BALANCING_CONFIRMATION_SEARCH_ROOT = "/kaggle/input"
RESUME_SEARCH_ROOT = "/kaggle/input"

DATA_DIR_CANDIDATES = [
    "/kaggle/working/datasetostrrfixed_65x110",
    "/kaggle/input/datasetostrrfixed-65x110",
    "/kaggle/input/datasetostrrfixed_65x110",
    "/kaggle/input/datasets/patrykc01/datasetostrrfixed-65x110",
    "/kaggle/input/datasets/patrykc01/datasetostrrfixed_65x110",
]

REPRESENTATIONS_TO_RUN = ["STFT_2D", "RAW_1D"]
FINAL_SEEDS = list(range(7100, 7110))
MAX_RUNTIME_HOURS = 10.5
MIN_REMAINING_MINUTES_FOR_NEW_RUN = 12.0

FS = 360
EXPECTED_SEGMENT_LENGTH = 175
EXPECTED_WINDOW_BEFORE = 65
EXPECTED_WINDOW_AFTER = 110

CLASS_NAMES = ["N", "S", "V", "F"]
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
RARE_CLASS_INDICES = (1, 3)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
NUM_WORKERS = 2
N_JOBS_PREPROCESSING = max(1, min(3, (os.cpu_count() or 2) - 1))

WGAN_SEED_BASE = 810_000

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
WGAN_MAX_QUALITY_ATTEMPTS = 3
WGAN_MIN_SELECTED_ACCEPTABLE_FRACTION = 0.99
WGAN_MIN_DISTANCE_RATIO = 0.25
WGAN_MAX_DISTANCE_RATIO = 2.00
WGAN_MIN_DIVERSITY_RATIO = 0.25
WGAN_MAX_DIVERSITY_RATIO = 2.00
SAVE_WGAN_GENERATORS = False
SAVE_MODEL_CHECKPOINTS = True
SAVE_TEST_PREDICTIONS = True

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


def make_final_configs():
    configs = [{
        "config_id": "BASELINE_CE",
        "family": "baseline",
        "kind": "baseline",
    }, {
        "config_id": "FOCAL_A025_G3",
        "family": "focal",
        "kind": "focal",
        "alpha_power": 0.25,
        "gamma": 3.0,
    }, {
        "config_id": "WGAN_GP_X5",
        "family": "wgan_gp",
        "kind": "wgan_gp",
        "wgan_factor": 5,
    }]
    return configs


METHOD_CONFIGS = make_final_configs()
CONFIG_BY_ID = {config["config_id"]: config for config in METHOD_CONFIGS}

CONFIG_IDS_BY_REPRESENTATION = {
    "STFT_2D": [
        "BASELINE_CE",
        "WGAN_GP_X5",
    ],
    "RAW_1D": [
        "BASELINE_CE",
        "FOCAL_A025_G3",
    ],
}

FIXED_EPOCHS = {
    ("STFT_2D", "BASELINE_CE"): 10,
    ("STFT_2D", "WGAN_GP_X5"): 10,
    ("RAW_1D", "BASELINE_CE"): 15,
    ("RAW_1D", "FOCAL_A025_G3"): 12,
}

SELECTED_CONFIG_BY_REPRESENTATION = {
    "STFT_2D": "WGAN_GP_X5",
    "RAW_1D": "FOCAL_A025_G3",
}

EXPECTED_RUNS = sum(
    len(CONFIG_IDS_BY_REPRESENTATION[representation])
    for representation in REPRESENTATIONS_TO_RUN
) * len(FINAL_SEEDS)


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
        test_path = os.path.join(candidate, "mitbih_test.npz")
        if (
            os.path.exists(train_path)
            and os.path.exists(val_path)
            and os.path.exists(test_path)
        ):
            return train_path, val_path, test_path

        backslash_train = candidate + "\\mitbih_train.npz"
        backslash_val = candidate + "\\mitbih_val.npz"
        backslash_test = candidate + "\\mitbih_test.npz"
        if (
            os.path.exists(backslash_train)
            and os.path.exists(backslash_val)
            and os.path.exists(backslash_test)
        ):
            return backslash_train, backslash_val, backslash_test

    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if {
                "mitbih_train.npz", "mitbih_val.npz", "mitbih_test.npz"
            }.issubset(files):
                return (
                    os.path.join(root, "mitbih_train.npz"),
                    os.path.join(root, "mitbih_val.npz"),
                    os.path.join(root, "mitbih_test.npz"),
                )

    raise FileNotFoundError(
        "Nie znaleziono mitbih_train.npz, mitbih_val.npz i mitbih_test.npz. "
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
        except Exception:
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


def find_balancing_confirmation_source():
    manifest_paths = []
    if os.path.exists(BALANCING_CONFIRMATION_SEARCH_ROOT):
        for root, _, files in os.walk(BALANCING_CONFIRMATION_SEARCH_ROOT):
            if "balancing_confirmation_manifest.json" in files:
                manifest_paths.append(
                    os.path.join(root, "balancing_confirmation_manifest.json")
                )

    required_pairs = {
        (representation, config_id)
        for representation, config_ids in CONFIG_IDS_BY_REPRESENTATION.items()
        for config_id in config_ids
    }
    valid = []
    for manifest_path in dict.fromkeys(manifest_paths):
        try:
            with open(manifest_path, "r", encoding="utf-8") as file:
                manifest = json.load(file)
            if manifest.get("protocol") != (
                "paired balancing-method confirmation on full DS1 TRAIN and DS1 VAL"
            ):
                continue
            if not manifest.get("complete") or manifest.get("ds2_used"):
                continue
            if int(manifest.get("expected_runs", -1)) != 120:
                continue
            if int(manifest.get("completed_runs", -1)) != 120:
                continue
            if manifest.get("stft_config") != STFT_CONFIG:
                continue
            architectures = manifest.get("architectures", {})
            if any(
                architectures.get(representation, {}).get("candidate_id")
                != expected["candidate_id"]
                for representation, expected in EXPECTED_ARCHITECTURES.items()
            ):
                continue
            selected_path = os.path.join(
                os.path.dirname(manifest_path),
                "balancing_confirmation_selected.csv",
            )
            runs_path = os.path.join(
                os.path.dirname(manifest_path),
                "balancing_confirmation_runs.csv",
            )
            if not os.path.exists(selected_path) or not os.path.exists(runs_path):
                continue

            selected = pd.read_csv(selected_path)
            selected_map = dict(zip(selected["representation"], selected["config_id"]))
            if selected_map != SELECTED_CONFIG_BY_REPRESENTATION:
                continue
            if set(selected["selection_status"].astype(str)) != {"final_DS1"}:
                continue

            runs = pd.read_csv(runs_path)
            if len(runs) != 120:
                continue
            if runs.duplicated(["representation", "config_id", "seed"]).any():
                continue
            if set(runs["seed"].astype(int)) != set(range(6100, 6110)):
                continue
            available_pairs = set(zip(runs["representation"], runs["config_id"]))
            if not required_pairs.issubset(available_pairs):
                continue

            derived_epochs = {}
            for pair in required_pairs:
                representation, config_id = pair
                values = runs[
                    (runs["representation"] == representation)
                    & (runs["config_id"] == config_id)
                ]["best_epoch"].astype(int)
                if len(values) != 10:
                    raise ValueError(f"Brak 10 epok dla {pair}.")
                derived_epochs[pair] = int(np.floor(np.median(values) + 0.5))
            if derived_epochs != FIXED_EPOCHS:
                raise ValueError(
                    f"Niezgodne zamrozone epoki: {derived_epochs} != {FIXED_EPOCHS}."
                )

            valid.append((
                int(manifest.get("completed_runs", 0)),
                manifest_path,
                selected_path,
                runs_path,
                manifest,
                selected,
            ))
        except Exception:
            pass

    if not valid:
        raise FileNotFoundError(
            "Nie znaleziono kompletnego balancing_confirmation_manifest.json "
            "wraz z balancing_confirmation_selected.csv oraz "
            "balancing_confirmation_runs.csv. Podlacz output etapu 3B jako "
            "Kaggle Input."
        )

    _, manifest_path, selected_path, runs_path, manifest, selected = max(
        valid, key=lambda item: item[0]
    )


    return manifest_path, selected_path, runs_path, manifest, selected


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


def train_wgan_for_class(
    real_signals, class_index, pool_seed, final_seed, quality_attempt,
):
    class_name = CLASS_NAMES[class_index]
    seed = int(pool_seed) + int(class_index)
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


    if SAVE_WGAN_GENERATORS:
        checkpoint = {
            "class_index": int(class_index),
            "class_name": class_name,
            "seed": seed,
            "final_seed": int(final_seed),
            "quality_attempt": int(quality_attempt),
            "robust_scale": robust_scale,
            "generator_state_dict": {
                key: value.detach().cpu()
                for key, value in generator.state_dict().items()
            },
            "generator_config": {
                "latent_dim": WGAN_LATENT_DIM,
                "base_channels": WGAN_BASE_CHANNELS,
            },
        }
        torch.save(
            checkpoint,
            os.path.join(
                OUTPUT_DIR,
                f"wgan_gp_generator_seed_{final_seed}_"
                f"attempt_{quality_attempt}_{class_name}.pt",
            ),
        )
    pd.DataFrame(history).to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"wgan_gp_training_seed_{final_seed}_"
            f"attempt_{quality_attempt}_{class_name}.csv",
        ),
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

    selected_median = float(np.median(selected_distances))
    distance_ratio = float(selected_median / max(real_median, 1e-8))
    diversity_ratio = float(generated_self_median / max(real_median, 1e-8))
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
        "selected_nn_median": selected_median,
        "selected_nn_min": float(np.min(selected_distances)),
        "selected_nn_max": float(np.max(selected_distances)),
        "generated_self_nn_median": generated_self_median,
        "selected_to_real_nn_ratio": distance_ratio,
        "generated_self_to_real_nn_ratio": diversity_ratio,
        "memorization_warning": bool(
            selected_median < WGAN_MIN_DISTANCE_RATIO * real_median
        ),
        "mode_collapse_warning": bool(
            generated_self_median < WGAN_MIN_DIVERSITY_RATIO * real_median
        ),
        "distribution_shift_warning": bool(
            distance_ratio > WGAN_MAX_DISTANCE_RATIO
        ),
        "excess_diversity_warning": bool(
            diversity_ratio > WGAN_MAX_DIVERSITY_RATIO
        ),
    }
    return selected, selected_rr, quality


def wgan_quality_passes(quality):
    return bool(
        quality["selected_acceptable_fraction"]
        >= WGAN_MIN_SELECTED_ACCEPTABLE_FRACTION
        and WGAN_MIN_DISTANCE_RATIO
        <= quality["selected_to_real_nn_ratio"]
        <= WGAN_MAX_DISTANCE_RATIO
        and WGAN_MIN_DIVERSITY_RATIO
        <= quality["generated_self_to_real_nn_ratio"]
        <= WGAN_MAX_DIVERSITY_RATIO
        and not quality["memorization_warning"]
        and not quality["mode_collapse_warning"]
        and not quality["distribution_shift_warning"]
        and not quality["excess_diversity_warning"]
    )


def write_wgan_reports(
    pool_signals, pool_labels, quality_rows, final_seed, suffix="",
):
    suffix = f"_{suffix}" if suffix else ""
    pd.DataFrame(quality_rows).to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"wgan_gp_quality_seed_{final_seed}{suffix}.csv",
        ),
        index=False,
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
        os.path.join(
            OUTPUT_DIR,
            f"wgan_gp_examples_seed_{final_seed}{suffix}.csv",
        ),
        index=False,
    )


def wgan_pool_filename(final_seed):
    return f"wgan_gp_pool_seed_{int(final_seed)}.npz"


def save_wgan_pool(
    path, pool_signals, pool_rr, pool_labels, train_labels,
    final_seed, pool_seed, quality_attempt, quality_rows,
):
    np.savez_compressed(
        path,
        X=pool_signals.astype(np.float32, copy=False),
        RR=pool_rr.astype(np.float32, copy=False),
        Y=pool_labels.astype(np.int64, copy=False),
        WINDOW_LENGTH=np.asarray(EXPECTED_SEGMENT_LENGTH, dtype=np.int64),
        MAX_FACTOR=np.asarray(WGAN_MAX_FACTOR, dtype=np.int64),
        FINAL_SEED=np.asarray(final_seed, dtype=np.int64),
        POOL_SEED=np.asarray(pool_seed, dtype=np.int64),
        QUALITY_ATTEMPT=np.asarray(quality_attempt, dtype=np.int64),
        SOURCE_CLASS_COUNTS=np.bincount(train_labels, minlength=4).astype(np.int64),
        QUALITY_JSON=np.asarray(json.dumps(quality_rows, ensure_ascii=False)),
    )


def find_existing_wgan_pool(train_labels, final_seed):
    filename = wgan_pool_filename(final_seed)
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
                    "FINAL_SEED", "POOL_SEED", "QUALITY_ATTEMPT",
                    "SOURCE_CLASS_COUNTS", "QUALITY_JSON",
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
                if int(data["FINAL_SEED"]) != int(final_seed):
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
                pool_seed = int(data["POOL_SEED"])
                quality_attempt = int(data["QUALITY_ATTEMPT"])
            if not quality_rows or not all(
                bool(row.get("quality_gate_passed", False))
                for row in quality_rows
            ):
                continue
            valid.append((
                len(labels), path, signals, rr, labels, quality_rows,
                pool_seed, quality_attempt,
            ))
        except Exception:
            continue

    if not valid:
        return None
    (
        _, path, signals, rr, labels, quality_rows,
        pool_seed, quality_attempt,
    ) = max(valid, key=lambda item: item[0])


    output_path = os.path.join(OUTPUT_DIR, filename)
    save_wgan_pool(
        output_path, signals, rr, labels, train_labels,
        final_seed, pool_seed, quality_attempt, quality_rows,
    )
    write_wgan_reports(signals, labels, quality_rows, final_seed)
    return (
        signals, rr, labels, quality_rows,
        {"pool_seed": pool_seed, "quality_attempt": quality_attempt},
    )


def build_or_load_wgan_pool(
    train_signals, train_rr, train_labels, final_seed,
):
    existing = find_existing_wgan_pool(train_labels, final_seed)
    if existing is not None:
        return existing

    counts = np.bincount(train_labels, minlength=4)
    for quality_attempt in range(1, WGAN_MAX_QUALITY_ATTEMPTS + 1):
        pool_seed = (
            WGAN_SEED_BASE
            + int(final_seed) * 100
            + int(quality_attempt) * 10_000
        )
        pool_signals, pool_rr, pool_labels, quality_rows = [], [], [], []


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
                real_signals,
                class_index,
                pool_seed,
                final_seed,
                quality_attempt,
            )
            candidates = generate_wgan_candidates(
                generator,
                robust_scale,
                n_candidates=n_candidates,
                seed=pool_seed + 1_000 + int(class_index),
            )
            selected, selected_rr, quality = select_wgan_candidates(
                real_signals,
                real_rr,
                candidates,
                n_needed=n_needed,
                class_index=class_index,
            )
            quality.update({
                "final_seed": int(final_seed),
                "pool_seed": int(pool_seed),
                "quality_attempt": int(quality_attempt),
                "robust_scale": float(robust_scale),
            })
            quality["quality_gate_passed"] = wgan_quality_passes(quality)
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
        normalized = bool(
            np.max(np.abs(pool_signals.mean(axis=1))) < 1e-4
            and np.max(np.abs(pool_signals.std(axis=1) - 1.0)) < 1e-4
        )
        finite = bool(
            np.isfinite(pool_signals).all() and np.isfinite(pool_rr).all()
        )
        for quality in quality_rows:
            quality["pool_finite"] = finite
            quality["pool_standardized"] = normalized
            quality["quality_gate_passed"] = bool(
                quality["quality_gate_passed"] and finite and normalized
            )

        if all(row["quality_gate_passed"] for row in quality_rows):
            output_path = os.path.join(
                OUTPUT_DIR, wgan_pool_filename(final_seed)
            )
            save_wgan_pool(
                output_path,
                pool_signals,
                pool_rr,
                pool_labels,
                train_labels,
                final_seed,
                pool_seed,
                quality_attempt,
                quality_rows,
            )
            write_wgan_reports(
                pool_signals, pool_labels, quality_rows, final_seed
            )


            return (
                pool_signals,
                pool_rr,
                pool_labels,
                quality_rows,
                {"pool_seed": pool_seed, "quality_attempt": quality_attempt},
            )

        write_wgan_reports(
            pool_signals,
            pool_labels,
            quality_rows,
            final_seed,
            suffix=f"attempt_{quality_attempt}_rejected",
        )


    raise RuntimeError(
        f"WGAN-GP seed={final_seed}: zadna z "
        f"{WGAN_MAX_QUALITY_ATTEMPTS} pul nie przeszla QA."
    )


def get_wgan_extras(
    train_signals, train_rr, train_labels, factor, final_seed, extra_cache,
):
    pool_key = ("wgan_gp_pool", int(final_seed), WGAN_MAX_FACTOR)
    if pool_key not in extra_cache:
        extra_cache[pool_key] = build_or_load_wgan_pool(
            train_signals, train_rr, train_labels, final_seed
        )
    pool_signals, pool_rr, pool_labels, _, pool_info = extra_cache[pool_key]

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
        pool_info,
    )


def get_extra_dataset(
    representation, config, train_signals, train_rr, train_labels,
    final_seed, extra_cache,
):
    kind = config["kind"]
    if kind != "wgan_gp":
        return None, np.empty(0, dtype=np.int64), {
            "method_data_seed": None,
            "wgan_pool_seed": None,
            "wgan_quality_attempt": None,
        }

    variant = int(config["wgan_factor"])
    raw_key = ("wgan_gp", variant, int(final_seed))
    if raw_key not in extra_cache:
        wgan_signals, wgan_rr, wgan_labels, pool_info = get_wgan_extras(
            train_signals,
            train_rr,
            train_labels,
            variant,
            final_seed,
            extra_cache,
        )
        extra_cache[raw_key] = (wgan_signals, wgan_rr, wgan_labels)
        extra_cache[("wgan_info", int(final_seed))] = pool_info
    pool_info = extra_cache[("wgan_info", int(final_seed))]
    method_data_seed = int(pool_info["pool_seed"])

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
    return extra_cache[dataset_key], extra_labels, {
        "method_data_seed": int(method_data_seed),
        "wgan_pool_seed": int(pool_info["pool_seed"]),
        "wgan_quality_attempt": int(pool_info["quality_attempt"]),
    }


def build_loss(config, original_train_labels):
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
    return DataLoader(shuffle=True, generator=generator, **common)


METRIC_NAMES = [
    "Accuracy",
    "Weighted_F1",
    "Macro_F1",
    "Rare_Macro_SF",
    "Min_Rare_F1",
    "Rare_HMean_SF",
    "F1_N",
    "F1_S",
    "F1_V",
    "F1_F",
    "Precision_N",
    "Precision_S",
    "Precision_V",
    "Precision_F",
    "Recall_N",
    "Recall_S",
    "Recall_V",
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
        "Accuracy": float(np.mean(labels == predictions)),
        "Weighted_F1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "Macro_F1": float(np.mean(f1_values)),
        "Rare_Macro_SF": float((f1_s + f1_f) / 2.0),
        "Min_Rare_F1": float(min(f1_s, f1_f)),
        "Rare_HMean_SF": float(rare_hmean),
        "F1_N": float(f1_values[0]),
        "F1_S": f1_s,
        "F1_V": float(f1_values[2]),
        "F1_F": f1_f,
        "Precision_N": float(precision[0]),
        "Precision_S": float(precision[1]),
        "Precision_V": float(precision[2]),
        "Precision_F": float(precision[3]),
        "Recall_N": float(recall[0]),
        "Recall_S": float(recall[1]),
        "Recall_V": float(recall[2]),
        "Recall_F": float(recall[3]),
    }
    matrix = confusion_matrix(labels, predictions, labels=[0, 1, 2, 3])
    return metrics, matrix, labels.astype(np.int8), predictions.astype(np.int8)


def train_one_run(
    representation,
    architecture,
    config,
    train_dataset,
    labels_for_dataset,
    original_train_labels,
    test_dataset,
    seed,
    method_data_info,
    fixed_epochs,
):
    set_deterministic(seed)
    train_loader = make_train_loader(
        train_dataset, config, labels_for_dataset, seed
    )
    test_loader = DataLoader(
        test_dataset,
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

    history = []
    started = time.time()

    for epoch in range(1, int(fixed_epochs) + 1):
        model.train()
        losses = []
        for inputs, rr, labels in train_loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            rr = rr.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs, rr), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        history.append({
            "epoch": int(epoch),
            "train_loss_mean": float(np.mean(losses)),
        })


    metrics, matrix, y_true, y_pred = evaluate_model(model, test_loader)
    run_name = f"{representation}_{config['config_id']}_seed_{int(seed)}"
    pd.DataFrame(history).to_csv(
        os.path.join(OUTPUT_DIR, f"training_history_{run_name}.csv"),
        index=False,
    )
    if SAVE_TEST_PREDICTIONS:
        np.savez_compressed(
            os.path.join(OUTPUT_DIR, f"test_predictions_{run_name}.npz"),
            Y_TRUE=y_true,
            Y_PRED=y_pred,
            CLASS_NAMES=np.asarray(CLASS_NAMES),
            REPRESENTATION=np.asarray(representation),
            CONFIG_ID=np.asarray(config["config_id"]),
            SEED=np.asarray(int(seed), dtype=np.int64),
        )
    if SAVE_MODEL_CHECKPOINTS:
        torch.save({
            "protocol": "locked final DS2 evaluation",
            "representation": representation,
            "architecture_candidate_id": architecture["candidate_id"],
            "architecture_params": architecture["params"],
            "config": config,
            "seed": int(seed),
            "fixed_epochs": int(fixed_epochs),
            "n_parameters": int(n_parameters),
            "train_class_counts": named_counts(labels_for_dataset),
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
        }, os.path.join(OUTPUT_DIR, f"model_{run_name}.pt"))

    result = {
        "representation": representation,
        "architecture_candidate_id": architecture["candidate_id"],
        "config_id": config["config_id"],
        "family": config["family"],
        "config_json": json.dumps(config, sort_keys=True),
        "seed": int(seed),
        "method_data_seed": method_data_info.get("method_data_seed"),
        "wgan_pool_seed": method_data_info.get("wgan_pool_seed"),
        "wgan_quality_attempt": method_data_info.get("wgan_quality_attempt"),
        "n_parameters": int(n_parameters),
        "n_train_samples": int(len(labels_for_dataset)),
        "train_class_counts_json": json.dumps(named_counts(labels_for_dataset)),
        "fixed_epochs": int(fixed_epochs),
        "train_loss_final": float(history[-1]["train_loss_mean"]),
        "time_s": float(time.time() - started),
        "confusion_json": json.dumps(matrix.astype(int).tolist()),
    }
    result.update(metrics)

    del model, optimizer, criterion, train_loader, test_loader
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return result


def find_best_progress_file():
    filename = "final_ds2_runs.csv"
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
    _, _, frame = max(valid, key=lambda item: item[0])

    return frame


def t_confidence_interval(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean
    sem = float(values.std(ddof=1) / np.sqrt(len(values)))
    margin = float(
        student_t.ppf((1.0 + confidence) / 2.0, len(values) - 1) * sem
    )
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


def make_summary(runs):
    rows = []
    group_columns = ["representation", "config_id", "family"]
    for keys, group in runs.groupby(group_columns):
        representation, config_id, family = keys
        row = {
            "representation": representation,
            "config_id": config_id,
            "role": (
                "selected_on_DS1"
                if config_id == SELECTED_CONFIG_BY_REPRESENTATION[representation]
                else "baseline"
            ),
            "family": family,
            "n_seeds": int(group["seed"].nunique()),
            "n_train_samples": int(group["n_train_samples"].iloc[0]),
            "n_parameters": int(group["n_parameters"].iloc[0]),
            "fixed_epochs": int(group["fixed_epochs"].iloc[0]),
            "time_s_mean": float(group["time_s"].mean()),
            "config_json": group["config_json"].iloc[0],
        }
        for metric in METRIC_NAMES:
            values = group[metric].astype(float).to_numpy()
            low, high = t_confidence_interval(values)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["_role_order"] = summary["role"].map({
        "selected_on_DS1": 0,
        "baseline": 1,
    })
    summary = summary.sort_values(
        ["representation", "_role_order"], ascending=[True, True]
    ).drop(columns="_role_order").reset_index(drop=True)
    return summary


def make_vs_baseline(runs):
    rows = []
    metrics = ["Min_Rare_F1", "Rare_Macro_SF", "Macro_F1"]
    for representation in REPRESENTATIONS_TO_RUN:
        baseline = runs[
            (runs["representation"] == representation)
            & (runs["config_id"] == "BASELINE_CE")
        ].set_index("seed")
        for config_id in CONFIG_IDS_BY_REPRESENTATION[representation]:
            if config_id == "BASELINE_CE":
                continue
            candidate = runs[
                (runs["representation"] == representation)
                & (runs["config_id"] == config_id)
            ].set_index("seed")
            common_seeds = sorted(set(baseline.index) & set(candidate.index))
            for metric in metrics:
                base_values = baseline.loc[common_seeds, metric].astype(float).to_numpy()
                candidate_values = (
                    candidate.loc[common_seeds, metric].astype(float).to_numpy()
                )
                differences = candidate_values - base_values
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
                difference_std = (
                    float(differences.std(ddof=1))
                    if len(differences) > 1 else 0.0
                )
                cohens_dz = (
                    float(differences.mean() / difference_std)
                    if difference_std > 0.0 else 0.0
                )
                difference_ci_low, difference_ci_high = t_confidence_interval(
                    differences
                ) if len(differences) else (np.nan, np.nan)
                rows.append({
                    "representation": representation,
                    "metric": metric,
                    "candidate_config_id": config_id,
                    "baseline_config_id": "BASELINE_CE",
                    "n_paired_seeds": len(common_seeds),
                    "candidate_mean": (
                        float(candidate_values.mean())
                        if len(candidate_values) else np.nan
                    ),
                    "baseline_mean": (
                        float(base_values.mean()) if len(base_values) else np.nan
                    ),
                    "mean_difference_candidate_minus_baseline": (
                        float(differences.mean()) if len(differences) else np.nan
                    ),
                    "mean_difference_ci95_low": difference_ci_low,
                    "mean_difference_ci95_high": difference_ci_high,
                    "median_difference_candidate_minus_baseline": (
                        float(np.median(differences)) if len(differences) else np.nan
                    ),
                    "paired_cohens_dz": cohens_dz,
                    "wins_candidate": int(np.sum(differences > 0.0)),
                    "ties": int(np.sum(np.isclose(differences, 0.0))),
                    "wins_baseline": int(np.sum(differences < 0.0)),
                    "wilcoxon_p_raw": p_value,
                })

    comparisons = pd.DataFrame(rows)
    if not comparisons.empty:
        comparisons["wilcoxon_p_holm"] = np.nan
        for _, indices in comparisons.groupby("representation").groups.items():
            indices = list(indices)
            comparisons.loc[indices, "wilcoxon_p_holm"] = holm_adjust(
                comparisons.loc[indices, "wilcoxon_p_raw"].to_numpy()
            )
        comparisons["significant_holm_0_05"] = (
            comparisons["wilcoxon_p_holm"] < 0.05
        )
    return comparisons


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


def collect_wgan_quality_rows():
    rows = []
    seen_paths = set()
    search_roots = [OUTPUT_DIR, RESUME_SEARCH_ROOT]
    for search_root in search_roots:
        if not os.path.exists(search_root):
            continue
        for root, _, files in os.walk(search_root):
            for filename in files:
                if not (
                    filename.startswith("wgan_gp_pool_seed_")
                    and filename.endswith(".npz")
                ):
                    continue
                path = os.path.join(root, filename)
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                try:
                    with np.load(path, allow_pickle=False) as data:
                        final_seed = int(data["FINAL_SEED"])
                        if final_seed not in FINAL_SEEDS:
                            continue
                        quality_rows = json.loads(str(data["QUALITY_JSON"]))
                    for row in quality_rows:
                        item = dict(row)
                        item["source_pool_file"] = path
                        rows.append(item)
                except Exception:
                    continue
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        ["final_seed", "class_index", "quality_attempt"]
    ).drop_duplicates(["final_seed", "class_index"], keep="first")
    return frame.reset_index(drop=True)


def save_outputs(
    runs, architecture_manifest_path, architecture_manifest, selected_architectures,
    balancing_manifest_path, balancing_selected_path, balancing_runs_path,
    balancing_manifest, data_info, elapsed_hours,
):
    runs = runs.sort_values(
        ["representation", "config_id", "seed"]
    ).reset_index(drop=True)
    runs.to_csv(
        os.path.join(OUTPUT_DIR, "final_ds2_runs.csv"), index=False
    )

    summary = make_summary(runs)
    summary.to_csv(
        os.path.join(OUTPUT_DIR, "final_ds2_summary.csv"), index=False
    )
    comparisons = make_vs_baseline(runs)
    comparisons.to_csv(
        os.path.join(OUTPUT_DIR, "final_ds2_selected_vs_baseline.csv"),
        index=False,
    )
    complete = bool(len(runs) == EXPECTED_RUNS)
    save_mean_confusions(runs)
    wgan_quality = collect_wgan_quality_rows()
    wgan_quality.to_csv(
        os.path.join(OUTPUT_DIR, "final_wgan_gp_quality_all_seeds.csv"),
        index=False,
    )

    manifest = {
        "protocol": "locked final evaluation on DS2 after selection on DS1",
        "final_test_stage": True,
        "architecture_manifest_source": architecture_manifest_path,
        "architecture_manifest_protocol": architecture_manifest.get("protocol"),
        "balancing_confirmation_manifest_source": balancing_manifest_path,
        "balancing_confirmation_selected_source": balancing_selected_path,
        "balancing_confirmation_runs_source": balancing_runs_path,
        "balancing_confirmation_protocol": balancing_manifest.get("protocol"),
        "raw_window": {"before": 65, "after": 110, "length": 175},
        "stft_config": STFT_CONFIG,
        "representations": REPRESENTATIONS_TO_RUN,
        "architectures": selected_architectures,
        "final_seeds": FINAL_SEEDS,
        "config_ids_by_representation": CONFIG_IDS_BY_REPRESENTATION,
        "selected_config_by_representation": SELECTED_CONFIG_BY_REPRESENTATION,
        "baseline_config": "BASELINE_CE",
        "fixed_epochs": {
            f"{representation}|{config_id}": int(epochs)
            for (representation, config_id), epochs in FIXED_EPOCHS.items()
        },
        "fixed_epoch_rule": (
            "rounded median best_epoch across the 10 DS1 confirmation seeds"
        ),
        "method_configs": [
            CONFIG_BY_ID[config_id]
            for config_id in CONFIG_BY_ID
        ],
        "expected_runs": EXPECTED_RUNS,
        "completed_runs": int(len(runs)),
        "complete": complete,
        "data": data_info,
        "training_split": "DS1 TRAIN concatenated with DS1 VAL",
        "test_split": "DS2 TEST",
        "validation_during_final_training": False,
        "early_stopping": False,
        "checkpoint_selection_on_ds2": False,
        "techniques_are_isolated": True,
        "paired_by_seed": True,
        "wgan_gp": {
            "space": "standardized RAW signal generated only from full DS1",
            "classes": ["S", "F"],
            "class_specific_generators": True,
            "generator": "1D convolutional WGAN-GP",
            "epochs": WGAN_EPOCHS,
            "latent_dim": WGAN_LATENT_DIM,
            "critic_steps": WGAN_CRITIC_STEPS,
            "lambda_gp": WGAN_LAMBDA_GP,
            "seed_base": WGAN_SEED_BASE,
            "fresh_pool_per_final_seed": True,
            "max_quality_attempts": WGAN_MAX_QUALITY_ATTEMPTS,
            "rr_assignment": "RR of nearest real beat from the same class",
            "quality_screen": "aligned Euclidean nearest-neighbour distance",
            "quality_thresholds": {
                "minimum_selected_acceptable_fraction": (
                    WGAN_MIN_SELECTED_ACCEPTABLE_FRACTION
                ),
                "selected_to_real_nn_ratio": [
                    WGAN_MIN_DISTANCE_RATIO, WGAN_MAX_DISTANCE_RATIO
                ],
                "generated_self_to_real_nn_ratio": [
                    WGAN_MIN_DIVERSITY_RATIO, WGAN_MAX_DIVERSITY_RATIO
                ],
            },
            "train_only": True,
            "combined_with_class_weights": False,
        },
        "statistical_analysis": {
            "confidence_intervals": (
                "95% Student-t across seeds; describes training/data-generation "
                "stochasticity, not patient-sampling uncertainty"
            ),
            "planned_comparisons": (
                "DS1-selected pipeline versus its frozen baseline within each "
                "representation"
            ),
            "test": "two-sided paired Wilcoxon by seed",
            "multiplicity": (
                "Holm correction across MinRare, RareMacro and Macro within "
                "each representation"
            ),
            "alpha": 0.05,
        },
        "model_checkpoints_saved": SAVE_MODEL_CHECKPOINTS,
        "test_predictions_saved": SAVE_TEST_PREDICTIONS,
        "requires_ds2": True,
        "ds2_used": bool(len(runs) > 0),
        "ds2_used_for_training": False,
        "ds2_used_for_hyperparameter_selection": False,
        "all_final_seed_results_reported": True,
        "post_ds2_changes_forbidden": True,
        "elapsed_hours_this_session": elapsed_hours,
    }
    with open(
        os.path.join(OUTPUT_DIR, "final_ds2_manifest.json"),
        "w", encoding="utf-8",
    ) as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    return summary, comparisons


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("Finalny test DS2 wymaga akceleratora GPU w Kaggle.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session_start = time.time()
    deadline = session_start + MAX_RUNTIME_HOURS * 3600.0

    (
        architecture_manifest_path,
        architecture_manifest,
        selected_architectures,
    ) = find_architecture_candidates()
    (
        balancing_manifest_path,
        balancing_selected_path,
        balancing_runs_path,
        balancing_manifest,
        _,
    ) = find_balancing_confirmation_source()


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


    train_path, val_path, test_path = resolve_data_paths()
    train_signals, train_rr, train_labels = load_full_split(train_path, "DS1 TRAIN")
    val_signals, val_rr, val_labels = load_full_split(val_path, "DS1 VAL")
    test_signals, test_rr, test_labels = load_full_split(test_path, "DS2 TEST")

    full_train_signals = np.concatenate([train_signals, val_signals], axis=0)
    full_train_rr = np.concatenate([train_rr, val_rr], axis=0)
    full_train_labels = np.concatenate([train_labels, val_labels], axis=0)
    del train_signals, train_rr, train_labels, val_signals, val_rr, val_labels
    gc.collect()

    data_info = {
        "ds1_train_path": train_path,
        "ds1_val_path": val_path,
        "ds2_test_path": test_path,
        "full_ds1_n_samples": int(len(full_train_labels)),
        "full_ds1_class_counts": named_counts(full_train_labels),
        "ds2_n_samples": int(len(test_labels)),
        "ds2_class_counts": named_counts(test_labels),
    }


    base_datasets = {
        "RAW_1D": RawRRDataset(
            full_train_signals, full_train_rr, full_train_labels
        ),
    }
    test_datasets = {
        "RAW_1D": RawRRDataset(test_signals, test_rr, test_labels),
    }


    train_images = process_to_stft_images(full_train_signals)
    test_images = process_to_stft_images(test_signals)


    base_datasets["STFT_2D"] = ImageRRDataset(
        train_images, full_train_rr, full_train_labels
    )
    test_datasets["STFT_2D"] = ImageRRDataset(
        test_images, test_rr, test_labels
    )

    runs = find_best_progress_file()
    valid_representations = set(REPRESENTATIONS_TO_RUN)
    valid_seeds = set(FINAL_SEEDS)
    valid_pairs = {
        (representation, config_id)
        for representation, config_ids in CONFIG_IDS_BY_REPRESENTATION.items()
        for config_id in config_ids
    }
    if not runs.empty:
        unknown_representations = set(runs["representation"]) - valid_representations
        unknown_pairs = set(zip(runs["representation"], runs["config_id"])) - valid_pairs
        unknown_seeds = set(runs["seed"].astype(int)) - valid_seeds
        wrong_architectures = set()
        wrong_epochs = set()
        for representation in REPRESENTATIONS_TO_RUN:
            observed = set(
                runs.loc[
                    runs["representation"] == representation,
                    "architecture_candidate_id",
                ].astype(str)
            )
            expected = selected_architectures[representation]["candidate_id"]
            wrong_architectures.update(observed - {expected})
        for row in runs[["representation", "config_id", "fixed_epochs"]].itertuples(
            index=False
        ):
            expected_epochs = FIXED_EPOCHS[(row.representation, row.config_id)]
            if int(row.fixed_epochs) != int(expected_epochs):
                wrong_epochs.add((
                    row.representation, row.config_id, int(row.fixed_epochs)
                ))
        if (
            unknown_representations
            or unknown_pairs
            or unknown_seeds
            or wrong_architectures
            or wrong_epochs
        ):
            raise RuntimeError(
                "Niezgodny plik wznowienia: "
                f"representations={unknown_representations}, "
                f"pairs={unknown_pairs}, seeds={unknown_seeds}, "
                f"architectures={wrong_architectures}, epochs={wrong_epochs}"
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
    for seed in FINAL_SEEDS:
        pending_for_seed = [
            (representation, config_id)
            for representation in REPRESENTATIONS_TO_RUN
            for config_id in CONFIG_IDS_BY_REPRESENTATION[representation]
            if (representation, config_id, int(seed)) not in completed_keys
        ]
        if not pending_for_seed:
            continue

        extra_cache = {}
        for representation in REPRESENTATIONS_TO_RUN:
            for config_id in CONFIG_IDS_BY_REPRESENTATION[representation]:
                key = (representation, config_id, int(seed))
                if key in completed_keys:
                    continue

                remaining_minutes = (deadline - time.time()) / 60.0
                if remaining_minutes < MIN_REMAINING_MINUTES_FOR_NEW_RUN:
                    stopped_by_time = True
                    break

                config = CONFIG_BY_ID[config_id]
                base_dataset = base_datasets[representation]
                try:
                    extra_dataset, extra_labels, method_data_info = get_extra_dataset(
                        representation,
                        config,
                        full_train_signals,
                        full_train_rr,
                        full_train_labels,
                        seed,
                        extra_cache,
                    )
                except Exception:
                    elapsed_hours = (time.time() - session_start) / 3600.0
                    save_outputs(
                        runs,
                        architecture_manifest_path,
                        architecture_manifest,
                        selected_architectures,
                        balancing_manifest_path,
                        balancing_selected_path,
                        balancing_runs_path,
                        balancing_manifest,
                        data_info,
                        elapsed_hours,
                    )
                    raise

                if extra_dataset is None:
                    train_dataset = base_dataset
                    labels_for_dataset = full_train_labels
                else:
                    train_dataset = ConcatDataset([base_dataset, extra_dataset])
                    labels_for_dataset = np.concatenate(
                        [full_train_labels, extra_labels]
                    )


                result = train_one_run(
                    representation=representation,
                    architecture=selected_architectures[representation],
                    config=config,
                    train_dataset=train_dataset,
                    labels_for_dataset=labels_for_dataset,
                    original_train_labels=full_train_labels,
                    test_dataset=test_datasets[representation],
                    seed=seed,
                    method_data_info=method_data_info,
                    fixed_epochs=FIXED_EPOCHS[(representation, config_id)],
                )


                runs = pd.concat([runs, pd.DataFrame([result])], ignore_index=True)
                runs = runs.drop_duplicates(
                    ["representation", "config_id", "seed"], keep="last"
                )
                runs.to_csv(
                    os.path.join(OUTPUT_DIR, "final_ds2_runs.csv"),
                    index=False,
                )
                completed_keys.add(key)

                del train_dataset, labels_for_dataset, extra_dataset
                gc.collect()
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

            if stopped_by_time:
                break
        del extra_cache
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        if stopped_by_time:
            break

    elapsed_hours = (time.time() - session_start) / 3600.0
    save_outputs(
        runs,
        architecture_manifest_path,
        architecture_manifest,
        selected_architectures,
        balancing_manifest_path,
        balancing_selected_path,
        balancing_runs_path,
        balancing_manifest,
        data_info,
        elapsed_hours,
    )


if __name__ == "__main__":
    main()

