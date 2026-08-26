import csv
import gc
import json
import os
import random
import shutil
import time
import warnings
from collections import Counter
from itertools import combinations

import cv2
import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from scipy.signal import spectrogram
from scipy.stats import t as student_t
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=UserWarning)
torch.backends.cuda.matmul.allow_tf32 = True
cv2.setNumThreads(1)

OUTPUT_DIR = "/kaggle/working/representation_comparison_65x110"

RESUME_INPUT_DIR = None

DATA_DIR_CANDIDATES = [
    "/kaggle/working/datasetostrrfixed_65x110",
    "/kaggle/input/datasets/patrykc01/datasetostrrfixed-65x110",
    "/kaggle/input/datasets/patrykc01/datasetostrrfixed_65x110",
]

REPRESENTATION_IDS_TO_RUN = [
    "RAW_1D_65x110",
    "STFT_T29",
    "CWT_T39",
    "DWT_T7",
    "SWT_T0",
]

MAX_RUNTIME_HOURS = 10.5

FS = 360
EXPECTED_SEGMENT_LENGTH = 175
EXPECTED_WINDOW_BEFORE = 65
EXPECTED_WINDOW_AFTER = 110

CLASS_NAMES = ["N", "S", "V", "F"]
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
N_EPOCHS = 30
PATIENCE = 8
MIN_DELTA = 1e-4
LR = 1e-3
WEIGHT_DECAY = 0.0
NUM_WORKERS = 2
N_JOBS_PREPROCESSING = max(1, min(3, (os.cpu_count() or 2) - 1))

TRAIN_SUBSET_SEED = 20260810
TRAIN_CLASS_LIMITS = {"N": 2000, "S": None, "V": 2000, "F": None}

MODEL_SEEDS = list(range(2000, 2010))

RUNS_CSV_NAME = "representation_runs.csv"
SUMMARY_CSV_NAME = "representation_summary.csv"
PARETO_CSV_NAME = "representation_pareto.csv"
COMPARISONS_CSV_NAME = "representation_pairwise_comparisons.csv"
MANIFEST_NAME = "representation_manifest.json"

REPRESENTATIONS = {
    "RAW_1D_65x110": {
        "method": "RAW_1D",
        "source_trial": None,
        "role": "raw_window_winner",
        "config": {
            "method": "RAW_1D",
            "window_before": 65,
            "window_after": 110,
            "segment_length": 175,
        },
    },
    "STFT_T29": {
        "method": "STFT",
        "source_trial": 29,
        "role": "confirmed_method_winner",
        "config": {
            "method": "STFT",
            "image_size": [128, 128],
            "norm_type": "sqrt",
            "clip_pct": 5,
            "nperseg": 64,
            "window": "hamming",
            "noverlap_pct": 0.25,
            "nfft": 64,
        },
    },
    "CWT_T39": {
        "method": "CWT",
        "source_trial": 39,
        "role": "confirmed_method_winner",
        "config": {
            "method": "CWT",
            "image_size": [128, 160],
            "norm_type": "linear",
            "clip_pct": 2,
            "wavelet": "gaus8",
            "scale_min": 2.0,
            "scale_max": 130.0,
            "scale_type": "logarithmic",
            "num_scales": 80,
        },
    },
    "DWT_T7": {
        "method": "DWT",
        "source_trial": 7,
        "role": "method_winner_by_primary_objective",
        "config": {
            "method": "DWT",
            "image_size": [224, 160],
            "norm_type": "linear",
            "clip_pct": 5,
            "wavelet": "bior3.5",
            "level": 3,
        },
    },
    "SWT_T0": {
        "method": "SWT",
        "source_trial": 0,
        "role": "confirmed_method_winner",
        "config": {
            "method": "SWT",
            "image_size": [224, 192],
            "norm_type": "linear",
            "clip_pct": 3,
            "wavelet": "sym4",
            "level": 4,
        },
    },
}

METRIC_NAMES = [
    "Macro_F1",
    "Rare_Macro_SF",
    "Min_Rare_F1",
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

RUN_COLUMNS = [
    "representation_id",
    "method",
    "source_trial",
    "role",
    "seed",
    "train_subset_seed",
    "n_train",
    "n_val",
    *METRIC_NAMES,
    "best_epoch",
    "time_s",
    "n_parameters",
    "cm_json",
    "config_json",
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

    raise FileNotFoundError(
        "Nie znaleziono mitbih_train.npz i mitbih_val.npz. "
        "Uzupelnij DATA_DIR_CANDIDATES."
    )


def validate_npz_metadata(data, split_name):
    required = {"X", "RR", "Y"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"{split_name}: brakuje tablic {sorted(missing)}.")

    signals = data["X"]
    rr = data["RR"]
    labels = data["Y"]

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
    stds = signals.std(axis=1, keepdims=True)
    stds = np.maximum(stds, 1e-6)
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


def apply_norm(array, norm_type):
    magnitude = np.abs(array)
    if norm_type == "log1p":
        return np.log1p(magnitude + 1e-8)
    if norm_type == "sqrt":
        return np.sqrt(magnitude + 1e-8)
    return magnitude


def build_cwt_scales(config):
    if config["scale_type"] == "logarithmic":
        return np.logspace(
            np.log10(config["scale_min"]),
            np.log10(config["scale_max"]),
            num=config["num_scales"],
            dtype=np.float32,
        )
    return np.linspace(
        config["scale_min"],
        config["scale_max"],
        num=config["num_scales"],
        dtype=np.float32,
    )


def transform_one_signal(signal, config, cwt_scales=None):
    method = config["method"]

    if method == "STFT":
        nperseg = config["nperseg"]
        noverlap = int(round(nperseg * config["noverlap_pct"]))
        _, _, coefficients = spectrogram(
            signal,
            fs=FS,
            window=config["window"],
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=config["nfft"],
            scaling="density",
            mode="psd",
        )
        image = apply_norm(coefficients, config["norm_type"])

    elif method == "CWT":
        coefficients, _ = pywt.cwt(signal, cwt_scales, config["wavelet"])
        image = apply_norm(coefficients, config["norm_type"])

    elif method == "DWT":
        coefficients = pywt.wavedec(signal, config["wavelet"], level=config["level"])
        target_axis = np.linspace(0.0, 1.0, len(signal))
        rows = []
        for coefficient in coefficients:
            source_axis = np.linspace(0.0, 1.0, len(coefficient))
            rows.append(np.interp(target_axis, source_axis, coefficient))
        image = apply_norm(np.vstack(rows), config["norm_type"])

    elif method == "SWT":
        level = config["level"]
        divisor = 2 ** level
        target_length = int(np.ceil(len(signal) / divisor) * divisor)
        pad_length = target_length - len(signal)
        padded = np.pad(
            signal,
            (pad_length // 2, pad_length - pad_length // 2),
            mode="edge",
        )
        coefficients = pywt.swt(padded, config["wavelet"], level=level)
        rows = [coefficients[0][0]] + [detail for _, detail in coefficients]
        image = apply_norm(np.vstack(rows), config["norm_type"])

    else:
        raise ValueError(f"Nieobslugiwana metoda 2D: {method}")

    image_height, image_width = config["image_size"]
    image = cv2.resize(
        image,
        (image_width, image_height),
        interpolation=cv2.INTER_CUBIC,
    )
    clip_pct = config["clip_pct"]
    low, high = np.percentile(image, [clip_pct, 100 - clip_pct])
    image = np.clip(image, low, high)
    minimum, maximum = float(image.min()), float(image.max())
    image = (image - minimum) / (maximum - minimum + 1e-8)
    return np.rint(image * 255.0).astype(np.uint8)


def process_to_images(signals, config):
    cwt_scales = build_cwt_scales(config) if config["method"] == "CWT" else None
    images = Parallel(n_jobs=N_JOBS_PREPROCESSING, backend="loky")(
        delayed(transform_one_signal)(signal, config, cwt_scales)
        for signal in signals
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
        image = self.images[index].float().div_(255.0)
        return image, self.rr[index], self.labels[index]


class ProbeCNN1DWithRR(nn.Module):

    def __init__(self, n_classes=4, n_rr_features=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(16),
        )
        self.rr_projector = nn.Sequential(
            nn.Linear(n_rr_features, 16),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64 * 16 + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, signal, rr):
        signal_features = torch.flatten(self.features(signal), 1)
        rr_features = self.rr_projector(rr)
        return self.classifier(torch.cat([signal_features, rr_features], dim=1))


class ProbeCNN2DWithRR(nn.Module):

    def __init__(self, n_classes=4, n_rr_features=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.rr_projector = nn.Sequential(
            nn.Linear(n_rr_features, 16),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64 * 4 * 4 + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, image, rr):
        image_features = torch.flatten(self.features(image), 1)
        rr_features = self.rr_projector(rr)
        return self.classifier(torch.cat([image_features, rr_features], dim=1))


def count_parameters(model):
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


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
    f1 = f1_score(
        labels, predictions, labels=[0, 1, 2, 3], average=None, zero_division=0
    )
    precision = precision_score(
        labels, predictions, labels=[0, 1, 2, 3], average=None, zero_division=0
    )
    recall = recall_score(
        labels, predictions, labels=[0, 1, 2, 3], average=None, zero_division=0
    )
    cm = confusion_matrix(labels, predictions, labels=[0, 1, 2, 3])

    metrics = {
        "Macro_F1": float(np.mean(f1)),
        "Rare_Macro_SF": float((f1[1] + f1[3]) / 2.0),
        "Min_Rare_F1": float(min(f1[1], f1[3])),
    }
    for index, class_name in enumerate(CLASS_NAMES):
        metrics[f"F1_{class_name}"] = float(f1[index])
        metrics[f"Precision_{class_name}"] = float(precision[index])
        metrics[f"Recall_{class_name}"] = float(recall[index])
    metrics["cm"] = cm
    return metrics


def make_datasets(method, train_inputs, train_rr, train_labels, val_inputs, val_rr, val_labels):
    if method == "RAW_1D":
        return (
            RawRRDataset(train_inputs, train_rr, train_labels),
            RawRRDataset(val_inputs, val_rr, val_labels),
            ProbeCNN1DWithRR,
        )
    return (
        ImageRRDataset(train_inputs, train_rr, train_labels),
        ImageRRDataset(val_inputs, val_rr, val_labels),
        ProbeCNN2DWithRR,
    )


def train_one_seed(
    method,
    train_inputs,
    train_rr,
    train_labels,
    val_inputs,
    val_rr,
    val_labels,
    seed,
):
    set_deterministic(seed)
    train_dataset, val_dataset, model_class = make_datasets(
        method,
        train_inputs,
        train_rr,
        train_labels,
        val_inputs,
        val_rr,
        val_labels,
    )

    pin_memory = DEVICE.type == "cuda"
    generator = torch.Generator().manual_seed(seed)
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

    model = model_class(n_classes=4, n_rr_features=4).to(DEVICE)
    n_parameters = count_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
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

    del model, optimizer, train_loader, val_loader, train_dataset, val_dataset
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return best_metrics


def restore_progress_if_requested(runs_path):
    if RESUME_INPUT_DIR is None or os.path.exists(runs_path):
        return

    source = os.path.join(RESUME_INPUT_DIR, RUNS_CSV_NAME)
    if not os.path.exists(source):
        for root, _, files in os.walk(RESUME_INPUT_DIR):
            if RUNS_CSV_NAME in files:
                source = os.path.join(root, RUNS_CSV_NAME)
                break

    if os.path.exists(source):
        shutil.copy2(source, runs_path)


def load_existing_runs(runs_path):
    if not os.path.exists(runs_path):
        return pd.DataFrame(columns=RUN_COLUMNS)
    data = pd.read_csv(runs_path, on_bad_lines="skip")
    missing = set(RUN_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Plik postepu nie ma kolumn: {sorted(missing)}")
    return data


def append_run(runs_path, row):
    exists = os.path.exists(runs_path) and os.path.getsize(runs_path) > 0
    with open(runs_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RUN_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row[key] for key in RUN_COLUMNS})
        file.flush()
        os.fsync(file.fileno())


def mean_ci95(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, np.nan, np.nan, np.nan
    std = float(np.std(values, ddof=1))
    critical = float(student_t.ppf(0.975, df=len(values) - 1))
    half_width = critical * std / np.sqrt(len(values))
    return mean, std, mean - half_width, mean + half_width


def build_summary(runs, summary_path, pareto_path):
    rows = []
    for representation_id in REPRESENTATIONS:
        data = runs[runs["representation_id"] == representation_id].copy()
        if data.empty:
            continue
        representation = REPRESENTATIONS[representation_id]
        row = {
            "representation_id": representation_id,
            "method": representation["method"],
            "source_trial": representation["source_trial"],
            "role": representation["role"],
            "n_runs": int(len(data)),
            "n_parameters": int(pd.to_numeric(data["n_parameters"]).iloc[0]),
            "config_json": json.dumps(representation["config"], ensure_ascii=False),
        }
        for metric in METRIC_NAMES:
            values = pd.to_numeric(data[metric], errors="coerce").dropna().to_numpy()
            mean, std, ci_low, ci_high = mean_ci95(values)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_CI95_low"] = ci_low
            row[f"{metric}_CI95_high"] = ci_high
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary, summary

    summary = summary.sort_values(
        ["Rare_Macro_SF_mean", "Macro_F1_mean", "Min_Rare_F1_mean"],
        ascending=False,
    )
    summary.to_csv(summary_path, index=False)

    objectives = ["Rare_Macro_SF_mean", "Macro_F1_mean", "Min_Rare_F1_mean"]
    values = summary[objectives].to_numpy(dtype=float)
    nondominated = np.ones(len(summary), dtype=bool)
    for i in range(len(summary)):
        for j in range(len(summary)):
            if i == j:
                continue
            if np.all(values[j] >= values[i]) and np.any(values[j] > values[i]):
                nondominated[i] = False
                break

    pareto = summary.loc[nondominated].copy()
    pareto.to_csv(pareto_path, index=False)
    return summary, pareto


def holm_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if len(valid) == 0:
        return adjusted

    order = valid[np.argsort(values[valid])]
    running = 0.0
    total = len(order)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def build_pairwise_comparisons(runs, output_path):
    comparison_rows = []
    metrics = ["Rare_Macro_SF", "Macro_F1", "Min_Rare_F1"]
    available = [
        representation_id
        for representation_id in REPRESENTATIONS
        if representation_id in set(runs["representation_id"])
    ]

    for metric in metrics:
        pivot = runs.pivot_table(
            index="seed", columns="representation_id", values=metric, aggfunc="first"
        )
        metric_start = len(comparison_rows)

        for representation_a, representation_b in combinations(available, 2):
            paired = pivot[[representation_a, representation_b]].dropna()
            if len(paired) < 2:
                continue

            a = paired[representation_a].to_numpy(dtype=float)
            b = paired[representation_b].to_numpy(dtype=float)
            diff = a - b
            mean_diff = float(np.mean(diff))
            std_diff = float(np.std(diff, ddof=1))
            sem = std_diff / np.sqrt(len(diff))
            critical = float(student_t.ppf(0.975, df=len(diff) - 1))
            p_t = float(ttest_rel(a, b).pvalue)
            try:
                p_w = float(wilcoxon(diff, zero_method="wilcox").pvalue)
            except ValueError:
                p_w = np.nan

            comparison_rows.append({
                "metric": metric,
                "representation_a": representation_a,
                "representation_b": representation_b,
                "n_pairs": int(len(diff)),
                "mean_a": float(np.mean(a)),
                "mean_b": float(np.mean(b)),
                "mean_diff_a_minus_b": mean_diff,
                "CI95_low": mean_diff - critical * sem,
                "CI95_high": mean_diff + critical * sem,
                "cohens_dz": mean_diff / std_diff if std_diff > 0 else np.nan,
                "p_ttest": p_t,
                "p_wilcoxon": p_w,
            })

        metric_end = len(comparison_rows)
        if metric_end > metric_start:
            subset = comparison_rows[metric_start:metric_end]
            adjusted_t = holm_adjust([row["p_ttest"] for row in subset])
            adjusted_w = holm_adjust([row["p_wilcoxon"] for row in subset])
            for row, p_t_holm, p_w_holm in zip(subset, adjusted_t, adjusted_w):
                row["p_ttest_holm"] = p_t_holm
                row["p_wilcoxon_holm"] = p_w_holm

    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(output_path, index=False)
    return comparisons


def save_mean_confusions(runs, output_dir):
    for representation_id in REPRESENTATIONS:
        data = runs[runs["representation_id"] == representation_id]
        matrices = []
        for value in data["cm_json"].dropna():
            matrices.append(np.asarray(json.loads(value), dtype=float))
        if not matrices:
            continue
        mean_cm = np.mean(matrices, axis=0)
        path = os.path.join(output_dir, f"mean_confusion_{representation_id}.csv")
        pd.DataFrame(mean_cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(path)


def save_manifest(path, runs, summary, pareto, elapsed_hours):
    completed_pairs = {
        (str(row.representation_id), int(row.seed))
        for row in runs.itertuples(index=False)
    }
    expected_pairs = {
        (representation_id, seed)
        for representation_id in REPRESENTATION_IDS_TO_RUN
        for seed in MODEL_SEEDS
    }
    payload = {
        "protocol": "controlled input-representation comparison on DS1",
        "window_before": EXPECTED_WINDOW_BEFORE,
        "window_after": EXPECTED_WINDOW_AFTER,
        "segment_length": EXPECTED_SEGMENT_LENGTH,
        "rr_features_used_by_every_representation": True,
        "train_subset_seed": TRAIN_SUBSET_SEED,
        "train_class_limits": TRAIN_CLASS_LIMITS,
        "validation": "full DS1 VAL",
        "model_seeds": MODEL_SEEDS,
        "n_epochs": N_EPOCHS,
        "patience": PATIENCE,
        "early_stopping_metric": "Macro F1",
        "selection_metrics": ["Rare_Macro_SF", "Macro_F1", "Min_Rare_F1"],
        "representations_requested": REPRESENTATION_IDS_TO_RUN,
        "completed_runs": len(completed_pairs & expected_pairs),
        "expected_runs": len(expected_pairs),
        "all_requested_runs_complete": expected_pairs.issubset(completed_pairs),
        "elapsed_hours_this_session": elapsed_hours,
        "pareto_representation_ids": (
            pareto["representation_id"].tolist() if not pareto.empty else []
        ),
        "ds2_used": False,
        "representations": REPRESENTATIONS,
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def main():
    invalid = set(REPRESENTATION_IDS_TO_RUN) - set(REPRESENTATIONS)
    if invalid:
        raise ValueError(f"Nieznane representation_id: {sorted(invalid)}")
    if not REPRESENTATION_IDS_TO_RUN:
        raise ValueError("REPRESENTATION_IDS_TO_RUN nie moze byc puste.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    runs_path = os.path.join(OUTPUT_DIR, RUNS_CSV_NAME)
    summary_path = os.path.join(OUTPUT_DIR, SUMMARY_CSV_NAME)
    pareto_path = os.path.join(OUTPUT_DIR, PARETO_CSV_NAME)
    comparisons_path = os.path.join(OUTPUT_DIR, COMPARISONS_CSV_NAME)
    manifest_path = os.path.join(OUTPUT_DIR, MANIFEST_NAME)
    restore_progress_if_requested(runs_path)

    train_path, val_path = resolve_data_paths()

    (
        train_signals,
        train_rr,
        train_labels,
        val_signals,
        val_rr,
        val_labels,
    ) = load_data(train_path, val_path)

    existing = load_existing_runs(runs_path)
    completed = {
        (str(row.representation_id), int(row.seed))
        for row in existing.itertuples(index=False)
    }

    session_start = time.time()
    stop_requested = False

    for representation_id in REPRESENTATION_IDS_TO_RUN:
        pending_seeds = [
            seed for seed in MODEL_SEEDS
            if (representation_id, seed) not in completed
        ]
        if not pending_seeds:
            continue
        if (time.time() - session_start) >= MAX_RUNTIME_HOURS * 3600:
            stop_requested = True
            break

        representation = REPRESENTATIONS[representation_id]
        method = representation["method"]
        config = representation["config"]

        if method == "RAW_1D":
            train_inputs = train_signals
            val_inputs = val_signals
        else:
            train_inputs = process_to_images(train_signals, config)
            val_inputs = process_to_images(val_signals, config)

        for seed in pending_seeds:
            if (time.time() - session_start) >= MAX_RUNTIME_HOURS * 3600:
                stop_requested = True
                break

            run_start = time.time()
            metrics = train_one_seed(
                method,
                train_inputs,
                train_rr,
                train_labels,
                val_inputs,
                val_rr,
                val_labels,
                seed,
            )
            cm = metrics.pop("cm")
            row = {
                "representation_id": representation_id,
                "method": method,
                "source_trial": representation["source_trial"],
                "role": representation["role"],
                "seed": seed,
                "train_subset_seed": TRAIN_SUBSET_SEED,
                "n_train": len(train_labels),
                "n_val": len(val_labels),
                **metrics,
                "time_s": time.time() - run_start,
                "cm_json": json.dumps(cm.tolist(), separators=(",", ":")),
                "config_json": json.dumps(config, ensure_ascii=False),
            }
            append_run(runs_path, row)
            completed.add((representation_id, seed))

        if method != "RAW_1D":
            del train_inputs, val_inputs
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        if stop_requested:
            break

    runs = load_existing_runs(runs_path)
    summary, pareto = build_summary(runs, summary_path, pareto_path)
    build_pairwise_comparisons(runs, comparisons_path)
    save_mean_confusions(runs, OUTPUT_DIR)

    elapsed_hours = (time.time() - session_start) / 3600.0
    save_manifest(manifest_path, runs, summary, pareto, elapsed_hours)


if __name__ == "__main__":
    main()
