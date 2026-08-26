import gc
import json
import os
import random
import shutil
import time
import warnings

import cv2
import numpy as np
import optuna
import pywt
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from scipy.signal import spectrogram
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=UserWarning)
torch.backends.cuda.matmul.allow_tf32 = True
cv2.setNumThreads(1)

METHODS_TO_RUN = ["DWT", "SWT"]

RESUME_INPUT_DIR = None

OUTPUT_DIR = "/kaggle/working/results_tfa_65x110"

DATA_DIR_CANDIDATES = [
    "/kaggle/working/datasetostrrfixed_65x110",
    "/kaggle/input/datasets/patrykc01/datasetostrrfixed-65x110",
    "/kaggle/input/datasets/patrykc01/datasetostrrfixed_65x110",
]

TIME_BUDGET_HOURS = {
    "STFT": 2.5,
    "CWT": 4.5,
    "DWT": 1.5,
    "SWT": 1.5,
}
MAX_TOTAL_TRIALS = {
    "STFT": 80,
    "CWT": 100,
    "DWT": 40,
    "SWT": 40,
}
NSGA_POPULATION_SIZE = {
    "STFT": 12,
    "CWT": 18,
    "DWT": 8,
    "SWT": 8,
}

IMAGE_SIZE_CANDIDATES = [
    "96x128",
    "128x128",
    "128x160",
    "160x160",
    "160x192",
    "160x224",
    "160x256",
    "192x160",
    "192x192",
    "192x224",
    "224x160",
    "224x192",
    "224x224",
    "224x256",
]

FS = 360
EXPECTED_SEGMENT_LENGTH = 175
EXPECTED_WINDOW_BEFORE = 65
EXPECTED_WINDOW_AFTER = 110

CLASS_NAMES = ["N", "S", "V", "F"]
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
N_EPOCHS = 12
PATIENCE = 4
MIN_DELTA = 1e-4
LR = 1e-3
WEIGHT_DECAY = 0.0
RUN_SEEDS = [42, 43, 44]
NUM_WORKERS = 2
SUBSET_SEED = 42

TRAIN_CLASS_LIMITS = {"N": 2000, "S": None, "V": 2000, "F": None}
VAL_CLASS_LIMITS = {"N": 400, "S": None, "V": 400, "F": None}

N_JOBS_PREPROCESSING = max(1, min(3, (os.cpu_count() or 2) - 1))
ALLOWED_METHODS = {"STFT", "CWT", "DWT", "SWT"}

assert len(RUN_SEEDS) == 3


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        "Uzupełnij DATA_DIR_CANDIDATES. Sprawdzono: "
        + ", ".join(DATA_DIR_CANDIDATES)
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
        raise ValueError(f"{split_name}: niepoprawna częstotliwość próbkowania.")
    if not np.isfinite(signals).all():
        raise ValueError(f"{split_name}: X zawiera NaN lub Inf.")


def deterministic_probe_subset(signals, labels_raw, limits, seed):
    labels_raw = np.asarray(labels_raw).astype(str)
    mask = labels_raw != "Q"
    signals = signals[mask].astype(np.float32, copy=False)
    labels_raw = labels_raw[mask]
    labels = np.asarray([LABEL_MAP[label] for label in labels_raw], dtype=np.int64)

    rng = np.random.default_rng(seed)
    selected_indices = []
    for class_name in CLASS_NAMES:
        class_idx = LABEL_MAP[class_name]
        indices = np.flatnonzero(labels == class_idx)
        limit = limits[class_name]
        if limit is not None and len(indices) > limit:
            indices = rng.permutation(indices)[:limit]
        selected_indices.append(np.sort(indices))

    selected_indices = np.concatenate(selected_indices)
    return signals[selected_indices], labels[selected_indices]


def load_probe_data(train_path, val_path):
    with np.load(train_path, allow_pickle=False) as train_data:
        validate_npz_metadata(train_data, "DS1 TRAIN")
        train_signals, train_labels = deterministic_probe_subset(
            train_data["X"], train_data["Y"], TRAIN_CLASS_LIMITS, SUBSET_SEED
        )

    with np.load(val_path, allow_pickle=False) as val_data:
        validate_npz_metadata(val_data, "DS1 VAL")
        val_signals, val_labels = deterministic_probe_subset(
            val_data["X"], val_data["Y"], VAL_CLASS_LIMITS, SUBSET_SEED
        )

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
        coefficients = pywt.wavedec(
            signal, config["wavelet"], level=config["level"]
        )
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
        image = self.images[index].float().div_(255.0)
        return image, self.labels[index]


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
            logits = model(images)
            predictions.extend(logits.argmax(1).cpu().numpy())
            labels.extend(batch_labels.numpy())

    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    macro = f1_score(labels, predictions, average="macro", zero_division=0)
    per_class = f1_score(
        labels, predictions, labels=[0, 1, 2, 3], average=None, zero_division=0
    )
    recall = recall_score(
        labels, predictions, labels=[0, 1, 2, 3], average=None, zero_division=0
    )
    return float(macro), per_class, recall


def run_training(train_images, train_labels, val_images, val_labels):
    run_results = []
    pin_memory = DEVICE.type == "cuda"

    for seed in RUN_SEEDS:
        set_deterministic(seed)
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

        best_macro = -np.inf
        best_per_class = None
        best_recall = None
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

            macro, per_class, recall = evaluate_model(model, val_loader)
            if macro > best_macro + MIN_DELTA:
                best_macro = macro
                best_per_class = per_class.copy()
                best_recall = recall.copy()
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

        rare_macro_sf = float((best_per_class[1] + best_per_class[3]) / 2.0)
        min_rare_f1 = float(min(best_per_class[1], best_per_class[3]))
        run_results.append({
            "seed": seed,
            "Macro_F1": float(best_macro),
            "Rare_Macro_SF": rare_macro_sf,
            "Min_Rare_F1": min_rare_f1,
            "F1_N": float(best_per_class[0]),
            "F1_S": float(best_per_class[1]),
            "F1_V": float(best_per_class[2]),
            "F1_F": float(best_per_class[3]),
            "Recall_S": float(best_recall[1]),
            "Recall_F": float(best_recall[3]),
            "best_epoch": int(best_epoch),
        })

        del model, optimizer, train_loader, val_loader
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return run_results


def valid_dwt_pairs():
    wavelets = ["db4", "db6", "sym4", "sym5", "bior3.5", "bior6.8"]
    pairs = []
    for wavelet in wavelets:
        maximum = min(5, pywt.dwt_max_level(EXPECTED_SEGMENT_LENGTH, wavelet))
        for level in range(2, maximum + 1):
            pairs.append(f"{wavelet}|{level}")
    return pairs


def suggest_config(trial, method):
    image_size_text = trial.suggest_categorical(
        "image_size", IMAGE_SIZE_CANDIDATES
    )
    image_height, image_width = map(int, image_size_text.split("x"))
    config = {
        "method": method,
        "image_size": (image_height, image_width),
        "norm_type": trial.suggest_categorical(
            "norm_type", ["log1p", "sqrt", "linear"]
        ),
        "clip_pct": trial.suggest_int("clip_pct", 1, 5),
    }

    if method == "STFT":
        config["nperseg"] = trial.suggest_categorical(
            "nperseg", [32, 48, 64, 80, 96, 112, 128]
        )
        config["window"] = trial.suggest_categorical(
            "stft_window", ["hann", "hamming", "blackman"]
        )
        config["noverlap_pct"] = trial.suggest_categorical(
            "noverlap_pct", [0.25, 0.50, 0.75]
        )
        nfft_multiplier = trial.suggest_categorical("nfft_multiplier", [1, 2, 4])
        config["nfft"] = config["nperseg"] * nfft_multiplier

    elif method == "CWT":
        config["wavelet"] = trial.suggest_categorical(
            "cwt_wavelet", ["morl", "gaus4", "gaus8", "mexh", "cmor1.5-1.0"]
        )
        config["scale_min"] = trial.suggest_categorical(
            "scale_min", [1.0, 2.0, 4.0, 6.0, 8.0]
        )
        scale_span = trial.suggest_categorical(
            "scale_span", [16, 32, 48, 64, 96, 128]
        )
        config["scale_max"] = config["scale_min"] + scale_span
        config["scale_type"] = trial.suggest_categorical(
            "scale_type", ["linear", "logarithmic"]
        )
        config["num_scales"] = trial.suggest_categorical(
            "num_scales", [32, 48, 64, 80, 96, 128]
        )

    elif method == "DWT":
        pair = trial.suggest_categorical("dwt_wavelet_level", valid_dwt_pairs())
        config["wavelet"], level = pair.split("|")
        config["level"] = int(level)

    elif method == "SWT":
        config["wavelet"] = trial.suggest_categorical(
            "swt_wavelet", ["db4", "db6", "sym4", "sym5"]
        )
        config["level"] = trial.suggest_categorical("swt_level", [2, 3, 4])

    else:
        raise ValueError(f"Nieobsługiwana metoda: {method}")

    return config


def serializable_config(config):
    result = dict(config)
    result["image_size"] = list(result["image_size"])
    for key, value in list(result.items()):
        if isinstance(value, np.generic):
            result[key] = value.item()
    return result


def objective(trial, method, train_signals, train_labels, val_signals, val_labels):
    config = suggest_config(trial, method)
    start = time.time()

    train_images = None
    val_images = None
    try:
        preprocessing_start = time.time()
        train_images = process_to_images(train_signals, config)
        val_images = process_to_images(val_signals, config)
        preprocessing_seconds = time.time() - preprocessing_start

        training_start = time.time()
        run_results = run_training(
            train_images, train_labels, val_images, val_labels
        )
        training_seconds = time.time() - training_start

        macro_values = np.asarray([row["Macro_F1"] for row in run_results])
        rare_values = np.asarray(
            [row["Rare_Macro_SF"] for row in run_results]
        )
        min_rare_values = np.asarray(
            [row["Min_Rare_F1"] for row in run_results]
        )
        summary = {
            metric: float(np.mean([row[metric] for row in run_results]))
            for metric in [
                "F1_N", "F1_S", "F1_V", "F1_F", "Recall_S", "Recall_F"
            ]
        }
        macro_mean = float(np.mean(macro_values))
        macro_std = float(np.std(macro_values, ddof=1))
        rare_mean = float(np.mean(rare_values))
        rare_std = float(np.std(rare_values, ddof=1))
        min_rare_mean = float(np.mean(min_rare_values))
        min_rare_std = float(np.std(min_rare_values, ddof=1))

        trial.set_user_attr("config", serializable_config(config))
        trial.set_user_attr("rare_macro_sf_mean", rare_mean)
        trial.set_user_attr("rare_macro_sf_std", rare_std)
        trial.set_user_attr("min_rare_f1_mean", min_rare_mean)
        trial.set_user_attr("min_rare_f1_std", min_rare_std)
        trial.set_user_attr("macro_f1_mean", macro_mean)
        trial.set_user_attr("macro_f1_std", macro_std)
        for key, value in summary.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr(
            "best_epoch_median",
            float(np.median([row["best_epoch"] for row in run_results])),
        )
        trial.set_user_attr("preprocessing_seconds", preprocessing_seconds)
        trial.set_user_attr("training_seconds", training_seconds)
        trial.set_user_attr("total_seconds", time.time() - start)
        trial.set_user_attr("run_results", run_results)

        return rare_mean, macro_mean

    finally:
        if train_images is not None:
            del train_images
        if val_images is not None:
            del val_images
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()


def restore_database_if_requested(method, destination_path):
    if RESUME_INPUT_DIR is None or os.path.exists(destination_path):
        return
    database_name = f"study_{method.lower()}_65x110_multi_v2.db"
    source_path = os.path.join(RESUME_INPUT_DIR, database_name)
    if not os.path.exists(source_path):
        for root, _, files in os.walk(RESUME_INPUT_DIR):
            if database_name in files:
                source_path = os.path.join(root, database_name)
                break
    if os.path.exists(source_path):
        shutil.copy2(source_path, destination_path)


def trial_payload(trial, method):
    return {
        "method": method,
        "trial_number": int(trial.number),
        "Rare_Macro_SF_mean": float(trial.values[0]),
        "Rare_Macro_SF_std": trial.user_attrs["rare_macro_sf_std"],
        "Macro_F1_mean": float(trial.values[1]),
        "Macro_F1_std": trial.user_attrs["macro_f1_std"],
        "Min_Rare_F1_mean": trial.user_attrs["min_rare_f1_mean"],
        "Min_Rare_F1_std": trial.user_attrs["min_rare_f1_std"],
        "F1_N": trial.user_attrs["F1_N"],
        "F1_S": trial.user_attrs["F1_S"],
        "F1_V": trial.user_attrs["F1_V"],
        "F1_F": trial.user_attrs["F1_F"],
        "Recall_S": trial.user_attrs["Recall_S"],
        "Recall_F": trial.user_attrs["Recall_F"],
        "config": trial.user_attrs["config"],
    }


def export_study(study, method):
    complete_trials = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not complete_trials:
        raise RuntimeError(
            f"{method}: żadna próba nie zakończyła się poprawnie; "
            "sprawdź wcześniejsze komunikaty błędów."
        )

    trials = study.trials_dataframe()
    trials = trials.sort_values(
        ["values_0", "values_1"],
        ascending=[False, False],
        na_position="last",
    )
    trials_path = os.path.join(OUTPUT_DIR, f"trials_{method.lower()}.csv")
    trials.to_csv(trials_path, index=False)

    pareto_trials = sorted(
        study.best_trials,
        key=lambda trial: (trial.values[0], trial.values[1]),
        reverse=True,
    )
    pareto_payload = {
        "method": method,
        "study_name": study.study_name,
        "objectives": [
            "maximize mean validation Rare-Macro F1(S,F)",
            "maximize mean validation Macro F1(N,S,V,F)",
        ],
        "run_seeds": RUN_SEEDS,
        "ds2_used": False,
        "pareto_trials": [
            trial_payload(trial, method) for trial in pareto_trials
        ],
    }
    pareto_path = os.path.join(OUTPUT_DIR, f"pareto_{method.lower()}.json")
    with open(pareto_path, "w", encoding="utf-8") as file:
        json.dump(pareto_payload, file, ensure_ascii=False, indent=2)

    rare_priority_trial = pareto_trials[0]
    candidate_payload = trial_payload(rare_priority_trial, method)
    candidate_payload.update({
        "selection_rule": (
            "highest Rare-Macro F1 on Pareto front; Macro F1 is co-objective"
        ),
        "requires_confirmation": True,
        "run_seeds": RUN_SEEDS,
        "ds2_used": False,
    })
    candidate_path = os.path.join(
        OUTPUT_DIR, f"candidate_rare_priority_{method.lower()}.json"
    )
    with open(candidate_path, "w", encoding="utf-8") as file:
        json.dump(candidate_payload, file, ensure_ascii=False, indent=2)

def run_method(method, train_signals, train_labels, val_signals, val_labels):
    database_path = os.path.join(
        OUTPUT_DIR, f"study_{method.lower()}_65x110_multi_v2.db"
    )
    restore_database_if_requested(method, database_path)

    study_name = f"tfa_{method.lower()}_65x110_raremacro_macro_v2"
    sampler = optuna.samplers.NSGAIISampler(
        seed=42,
        population_size=NSGA_POPULATION_SIZE[method],
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{database_path}",
        directions=["maximize", "maximize"],
        sampler=sampler,
        load_if_exists=True,
    )

    completed_trials = sum(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in study.trials
    )
    remaining_trials = max(0, MAX_TOTAL_TRIALS[method] - completed_trials)
    if remaining_trials > 0:
        study.optimize(
            lambda trial: objective(
                trial,
                method,
                train_signals,
                train_labels,
                val_signals,
                val_labels,
            ),
            n_trials=remaining_trials,
            timeout=TIME_BUDGET_HOURS[method] * 3600,
            gc_after_trial=True,
        )

    export_study(study, method)


def main():
    invalid_methods = set(METHODS_TO_RUN) - ALLOWED_METHODS
    if invalid_methods:
        raise ValueError(f"Nieznane metody: {sorted(invalid_methods)}")
    if not METHODS_TO_RUN:
        raise ValueError("METHODS_TO_RUN nie może być puste.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_path, val_path = resolve_data_paths()

    train_signals, train_labels, val_signals, val_labels = load_probe_data(
        train_path, val_path
    )

    for method in METHODS_TO_RUN:
        run_method(
            method,
            train_signals,
            train_labels,
            val_signals,
            val_labels,
        )


if __name__ == "__main__":
    main()
