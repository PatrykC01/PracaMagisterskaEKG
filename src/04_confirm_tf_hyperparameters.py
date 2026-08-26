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
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=UserWarning)
torch.backends.cuda.matmul.allow_tf32 = True
cv2.setNumThreads(1)

OUTPUT_DIR = "/kaggle/working/confirmation_tfa_65x110"

RESUME_INPUT_DIR = None

DATA_DIR_CANDIDATES = [
    "/kaggle/working/datasetostrrfixed_65x110",
    "/kaggle/input/datasets/patrykc01/datasetostrrfixed-65x110",
    "/kaggle/input/datasets/patrykc01/datasetostrrfixed_65x110",
]

CANDIDATE_IDS_TO_RUN = [
    "STFT_T29_BALANCED",
    "CWT_T42_RARE",
    "CWT_T39_BALANCED",
    "DWT_T7_RARE",
    "DWT_T39_BALANCED",
    "SWT_T25_RARE",
    "SWT_T0_BALANCED",
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
N_EPOCHS = 20
PATIENCE = 6
MIN_DELTA = 1e-4
LR = 1e-3
WEIGHT_DECAY = 0.0
NUM_WORKERS = 2
N_JOBS_PREPROCESSING = max(1, min(3, (os.cpu_count() or 2) - 1))

CONFIRM_TRAIN_SUBSET_SEED = 20260810
TRAIN_CLASS_LIMITS = {"N": 2000, "S": None, "V": 2000, "F": None}

CONFIRM_SEEDS = list(range(1000, 1010))

RUNS_CSV_NAME = "confirmation_runs.csv"
SUMMARY_CSV_NAME = "confirmation_summary.csv"
PARETO_CSV_NAME = "confirmation_pareto.csv"
COMPARISONS_CSV_NAME = "confirmation_pairwise_comparisons.csv"
MANIFEST_NAME = "confirmation_manifest.json"

CANDIDATES = {
    "STFT_T29_BALANCED": {
        "method": "STFT",
        "source_trial": 29,
        "role": "rare_and_balanced",
        "screen_Rare_Macro_SF": 0.23881021469905228,
        "screen_Macro_F1": 0.42099432963060684,
        "screen_Min_Rare_F1": 0.18778610070744903,
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
    "CWT_T42_RARE": {
        "method": "CWT",
        "source_trial": 42,
        "role": "rare_priority",
        "screen_Rare_Macro_SF": 0.31153400771338574,
        "screen_Macro_F1": 0.4648193293926326,
        "screen_Min_Rare_F1": 0.012345679012345678,
        "config": {
            "method": "CWT",
            "image_size": [192, 192],
            "norm_type": "linear",
            "clip_pct": 2,
            "wavelet": "morl",
            "scale_min": 4.0,
            "scale_max": 20.0,
            "scale_type": "logarithmic",
            "num_scales": 48,
        },
    },
    "CWT_T39_BALANCED": {
        "method": "CWT",
        "source_trial": 39,
        "role": "balanced_min_rare",
        "screen_Rare_Macro_SF": 0.1505142780230355,
        "screen_Macro_F1": 0.4061768593370429,
        "screen_Min_Rare_F1": 0.0791245791245791,
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
    "DWT_T7_RARE": {
        "method": "DWT",
        "source_trial": 7,
        "role": "rare_priority",
        "screen_Rare_Macro_SF": 0.21551660598717878,
        "screen_Macro_F1": 0.4550311733279391,
        "screen_Min_Rare_F1": 0.0,
        "config": {
            "method": "DWT",
            "image_size": [224, 160],
            "norm_type": "linear",
            "clip_pct": 5,
            "wavelet": "bior3.5",
            "level": 3,
        },
    },
    "DWT_T39_BALANCED": {
        "method": "DWT",
        "source_trial": 39,
        "role": "balanced_min_rare",
        "screen_Rare_Macro_SF": 0.1500005822545536,
        "screen_Macro_F1": 0.4088544109068333,
        "screen_Min_Rare_F1": 0.0957207207207207,
        "config": {
            "method": "DWT",
            "image_size": [160, 256],
            "norm_type": "linear",
            "clip_pct": 3,
            "wavelet": "sym5",
            "level": 3,
        },
    },
    "SWT_T25_RARE": {
        "method": "SWT",
        "source_trial": 25,
        "role": "rare_priority",
        "screen_Rare_Macro_SF": 0.25019626167200637,
        "screen_Macro_F1": 0.42194503180313775,
        "screen_Min_Rare_F1": 0.09438867207829287,
        "config": {
            "method": "SWT",
            "image_size": [96, 128],
            "norm_type": "linear",
            "clip_pct": 1,
            "wavelet": "sym5",
            "level": 2,
        },
    },
    "SWT_T0_BALANCED": {
        "method": "SWT",
        "source_trial": 0,
        "role": "balanced_pareto",
        "screen_Rare_Macro_SF": 0.2139594081918428,
        "screen_Macro_F1": 0.4415207464243522,
        "screen_Min_Rare_F1": 0.18540051679586564,
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

RUN_COLUMNS = [
    "candidate_id",
    "method",
    "source_trial",
    "role",
    "seed",
    "train_subset_seed",
    "n_train",
    "n_val",
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
    "best_epoch",
    "time_s",
    "config_json",
]

SUMMARY_METRICS = [
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

    raise FileNotFoundError(
        "Nie znaleziono mitbih_train.npz i mitbih_val.npz. "
        "Uzupełnij DATA_DIR_CANDIDATES."
    )


def validate_npz_metadata(data, split_name):
    signals = data["X"]
    labels = data["Y"]
    if signals.ndim != 2 or signals.shape[1] != EXPECTED_SEGMENT_LENGTH:
        raise ValueError(
            f"{split_name}: oczekiwano X[:, {EXPECTED_SEGMENT_LENGTH}], "
            f"otrzymano {signals.shape}."
        )
    if len(signals) != len(labels):
        raise ValueError(f"{split_name}: różna liczba X i Y.")
    if "WINDOW_BEFORE" in data and int(data["WINDOW_BEFORE"]) != EXPECTED_WINDOW_BEFORE:
        raise ValueError(f"{split_name}: niepoprawne WINDOW_BEFORE.")
    if "WINDOW_AFTER" in data and int(data["WINDOW_AFTER"]) != EXPECTED_WINDOW_AFTER:
        raise ValueError(f"{split_name}: niepoprawne WINDOW_AFTER.")
    if "FS" in data and int(data["FS"]) != FS:
        raise ValueError(f"{split_name}: niepoprawne FS.")
    if not np.isfinite(signals).all():
        raise ValueError(f"{split_name}: X zawiera NaN lub Inf.")


def encode_labels(labels_raw):
    labels_raw = np.asarray(labels_raw).astype(str)
    mask = labels_raw != "Q"
    labels = np.asarray([LABEL_MAP[label] for label in labels_raw[mask]], dtype=np.int64)
    return mask, labels


def deterministic_train_subset(signals, labels_raw):
    mask, labels = encode_labels(labels_raw)
    signals = signals[mask].astype(np.float32, copy=False)
    rng = np.random.default_rng(CONFIRM_TRAIN_SUBSET_SEED)
    selected = []
    for class_name in CLASS_NAMES:
        class_idx = LABEL_MAP[class_name]
        indices = np.flatnonzero(labels == class_idx)
        limit = TRAIN_CLASS_LIMITS[class_name]
        if limit is not None and len(indices) > limit:
            indices = rng.permutation(indices)[:limit]
        selected.append(np.sort(indices))
    selected = np.concatenate(selected)
    return signals[selected], labels[selected]


def full_validation(signals, labels_raw):
    mask, labels = encode_labels(labels_raw)
    return signals[mask].astype(np.float32, copy=False), labels


def load_confirmation_data(train_path, val_path):
    with np.load(train_path, allow_pickle=False) as train_data:
        validate_npz_metadata(train_data, "DS1 TRAIN")
        train_signals, train_labels = deterministic_train_subset(
            train_data["X"], train_data["Y"]
        )
    with np.load(val_path, allow_pickle=False) as val_data:
        validate_npz_metadata(val_data, "DS1 VAL")
        val_signals, val_labels = full_validation(val_data["X"], val_data["Y"])

    return train_signals, train_labels, val_signals, val_labels


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
        raise ValueError(f"Nieobsługiwana metoda: {method}")

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


class ProbeImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.from_numpy(images).unsqueeze(1)
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index].float().div_(255.0), self.labels[index]


class ProbingCNN2D(nn.Module):
    def __init__(self, n_classes=4):
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
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def evaluate_model(model, loader):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            predictions.extend(model(images).argmax(1).cpu().numpy())
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


def train_one_seed(train_images, train_labels, val_images, val_labels, seed):
    set_deterministic(seed)
    pin_memory = DEVICE.type == "cuda"
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        ProbeImageDataset(train_images, train_labels),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        ProbeImageDataset(val_images, val_labels),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    model = ProbingCNN2D(n_classes=4).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()

    best_metrics = None
    best_macro = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
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
    del model, optimizer, train_loader, val_loader
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
        raise ValueError(f"Plik postępu nie ma kolumn: {sorted(missing)}")
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


def build_summary(runs, summary_path, pareto_path):
    rows = []
    for candidate_id in CANDIDATES:
        data = runs[runs["candidate_id"] == candidate_id].copy()
        if data.empty:
            continue
        candidate = CANDIDATES[candidate_id]
        row = {
            "candidate_id": candidate_id,
            "method": candidate["method"],
            "source_trial": candidate["source_trial"],
            "role": candidate["role"],
            "n_runs": int(len(data)),
            "screen_Rare_Macro_SF": candidate["screen_Rare_Macro_SF"],
            "screen_Macro_F1": candidate["screen_Macro_F1"],
            "screen_Min_Rare_F1": candidate["screen_Min_Rare_F1"],
            "config_json": json.dumps(candidate["config"], ensure_ascii=False),
        }
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(data[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
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


def build_pairwise_comparisons(runs, output_path):
    comparison_rows = []
    metrics = ["Rare_Macro_SF", "Macro_F1", "Min_Rare_F1"]
    available = [
        candidate_id for candidate_id in CANDIDATES
        if candidate_id in set(runs["candidate_id"])
    ]
    for metric in metrics:
        pivot = runs.pivot_table(
            index="seed", columns="candidate_id", values=metric, aggfunc="first"
        )
        metric_start = len(comparison_rows)
        for candidate_a, candidate_b in combinations(available, 2):
            paired = pivot[[candidate_a, candidate_b]].dropna()
            if len(paired) < 2:
                continue
            a = paired[candidate_a].to_numpy(dtype=float)
            b = paired[candidate_b].to_numpy(dtype=float)
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
                "candidate_a": candidate_a,
                "candidate_b": candidate_b,
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


def save_manifest(path, runs, summary, pareto, elapsed_hours):
    completed_pairs = {
        (str(row.candidate_id), int(row.seed))
        for row in runs.itertuples(index=False)
    }
    expected_pairs = {
        (candidate_id, seed)
        for candidate_id in CANDIDATE_IDS_TO_RUN
        for seed in CONFIRM_SEEDS
    }
    payload = {
        "protocol": "TFA confirmation on DS1",
        "window_before": EXPECTED_WINDOW_BEFORE,
        "window_after": EXPECTED_WINDOW_AFTER,
        "segment_length": EXPECTED_SEGMENT_LENGTH,
        "train_subset_seed": CONFIRM_TRAIN_SUBSET_SEED,
        "train_class_limits": TRAIN_CLASS_LIMITS,
        "validation": "full DS1 VAL",
        "model_seeds": CONFIRM_SEEDS,
        "early_stopping_metric": "Macro F1",
        "selection_metrics": ["Rare_Macro_SF", "Macro_F1", "Min_Rare_F1"],
        "candidate_ids_requested": CANDIDATE_IDS_TO_RUN,
        "completed_runs": len(completed_pairs & expected_pairs),
        "expected_runs": len(expected_pairs),
        "all_requested_runs_complete": expected_pairs.issubset(completed_pairs),
        "elapsed_hours_this_session": elapsed_hours,
        "pareto_candidate_ids": (
            pareto["candidate_id"].tolist() if not pareto.empty else []
        ),
        "ds2_used": False,
        "candidates": CANDIDATES,
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def main():
    invalid = set(CANDIDATE_IDS_TO_RUN) - set(CANDIDATES)
    if invalid:
        raise ValueError(f"Nieznane candidate_id: {sorted(invalid)}")
    if not CANDIDATE_IDS_TO_RUN:
        raise ValueError("CANDIDATE_IDS_TO_RUN nie może być puste.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    runs_path = os.path.join(OUTPUT_DIR, RUNS_CSV_NAME)
    summary_path = os.path.join(OUTPUT_DIR, SUMMARY_CSV_NAME)
    pareto_path = os.path.join(OUTPUT_DIR, PARETO_CSV_NAME)
    comparisons_path = os.path.join(OUTPUT_DIR, COMPARISONS_CSV_NAME)
    manifest_path = os.path.join(OUTPUT_DIR, MANIFEST_NAME)
    restore_progress_if_requested(runs_path)

    train_path, val_path = resolve_data_paths()

    train_signals, train_labels, val_signals, val_labels = load_confirmation_data(
        train_path, val_path
    )
    existing = load_existing_runs(runs_path)
    completed = {
        (str(row.candidate_id), int(row.seed))
        for row in existing.itertuples(index=False)
    }
    session_start = time.time()
    stop_requested = False

    for candidate_id in CANDIDATE_IDS_TO_RUN:
        pending_seeds = [
            seed for seed in CONFIRM_SEEDS
            if (candidate_id, seed) not in completed
        ]
        if not pending_seeds:
            continue
        if (time.time() - session_start) >= MAX_RUNTIME_HOURS * 3600:
            stop_requested = True
            break

        candidate = CANDIDATES[candidate_id]
        config = candidate["config"]

        train_images = process_to_images(train_signals, config)
        val_images = process_to_images(val_signals, config)

        for seed in pending_seeds:
            if (time.time() - session_start) >= MAX_RUNTIME_HOURS * 3600:
                stop_requested = True
                break
            run_start = time.time()
            metrics = train_one_seed(
                train_images, train_labels, val_images, val_labels, seed
            )
            row = {
                "candidate_id": candidate_id,
                "method": candidate["method"],
                "source_trial": candidate["source_trial"],
                "role": candidate["role"],
                "seed": seed,
                "train_subset_seed": CONFIRM_TRAIN_SUBSET_SEED,
                "n_train": len(train_labels),
                "n_val": len(val_labels),
                **metrics,
                "time_s": time.time() - run_start,
                "config_json": json.dumps(config, ensure_ascii=False),
            }
            append_run(runs_path, row)
            completed.add((candidate_id, seed))

        del train_images, val_images
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        if stop_requested:
            break

    runs = load_existing_runs(runs_path)
    summary, pareto = build_summary(runs, summary_path, pareto_path)
    build_pairwise_comparisons(runs, comparisons_path)
    elapsed_hours = (time.time() - session_start) / 3600.0
    save_manifest(manifest_path, runs, summary, pareto, elapsed_hours)


if __name__ == "__main__":
    main()
