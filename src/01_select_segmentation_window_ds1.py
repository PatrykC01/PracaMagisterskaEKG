import gc
import json
import os
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb
from imblearn.over_sampling import RandomOverSampler
from scipy import stats
from scipy.signal import butter, filtfilt
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset


warnings.filterwarnings("ignore", category=UserWarning)

MITDB_PATH = "/kaggle/input/datasets/patrykc01/mitdbdatabase/mitdb"
OUTPUT_DIR = "/kaggle/working/results_window_ds1_clean"
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "runs_progress.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FS = 360
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["N", "S", "V", "F"]
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

SCREEN_SEEDS = list(range(100, 106))
CONFIRM_SEEDS = list(range(1000, 1015))
TOP_K = 4

N_EPOCHS = 40
PATIENCE = 8
MIN_DELTA = 1e-4
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
ALPHA = 0.05

USE_OVERSAMPLING = True

AAMI_MAPPING = {
    ".": "N", "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}
BEAT_SYMBOLS = set(AAMI_MAPPING)

DS1_TRAIN_RECORDS = [
    "101", "106", "108", "109", "112", "115", "116", "119",
    "124", "201", "203", "205", "208", "209", "215", "230",
]
DS1_VAL_RECORDS = ["114", "118", "122", "207", "220", "223"]

CANDIDATE_WINDOWS = sorted({
    (65, 110), (70, 105), (70, 110), (70, 115), (75, 110),
    (75, 120), (80, 120), (80, 130), (80, 140), (85, 137),
    (85, 144), (90, 140), (90, 144), (90, 150), (95, 144),
})
BASELINE_WINDOW = (90, 144)


def window_name(window):
    return f"{window[0]}x{window[1]}"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=FS, order=3):
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
    return filtfilt(b, a, data)


def normalize_segment(segment):
    return (segment - np.mean(segment)) / (np.std(segment) + 1e-8)


_RECORD_CACHE = {}


def load_record_cached(record_name):
    if record_name in _RECORD_CACHE:
        return _RECORD_CACHE[record_name]

    record_path = os.path.join(MITDB_PATH, record_name)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, "atr")

    if "MLII" in record.sig_name:
        channel_idx = record.sig_name.index("MLII")
    elif "II" in record.sig_name:
        channel_idx = record.sig_name.index("II")
    else:
        channel_idx = 0

    clean_signal = bandpass_filter(record.p_signal[:, channel_idx])
    all_symbols = np.asarray(annotation.symbol)
    all_samples = np.asarray(annotation.sample)
    beat_mask = np.asarray([symbol in BEAT_SYMBOLS for symbol in all_symbols])
    result = (clean_signal, all_samples[beat_mask], all_symbols[beat_mask])
    _RECORD_CACHE[record_name] = result
    return result


def segment_records(record_list, window_before, window_after):
    signals, rr_features, labels = [], [], []

    for record_name in record_list:
        clean_signal, beat_samples, beat_symbols = load_record_cached(record_name)

        pre_rr = np.full(len(beat_samples), FS, dtype=np.float32)
        post_rr = np.full(len(beat_samples), FS, dtype=np.float32)
        if len(beat_samples) > 1:
            rr_differences = np.diff(beat_samples).astype(np.float32)
            pre_rr[1:] = rr_differences
            post_rr[:-1] = rr_differences

        for beat_idx, (symbol, position) in enumerate(zip(beat_symbols, beat_samples)):
            aami_class = AAMI_MAPPING[symbol]
            if aami_class == "Q":
                continue

            start_idx = int(position) - window_before
            end_idx = int(position) + window_after
            if start_idx < 0 or end_idx > len(clean_signal):
                continue

            segment = normalize_segment(clean_signal[start_idx:end_idx])
            local_start = max(0, beat_idx - 10)
            local_mean = (
                float(np.mean(pre_rr[local_start:beat_idx]))
                if beat_idx > 0 else float(FS)
            )
            if local_mean <= 0:
                local_mean = float(FS)

            rr = [
                pre_rr[beat_idx] / FS,
                post_rr[beat_idx] / FS,
                local_mean / FS,
                pre_rr[beat_idx] / local_mean,
            ]

            signals.append(segment)
            rr_features.append(rr)
            labels.append(LABEL_MAP[aami_class])

    return (
        np.asarray(signals, dtype=np.float32),
        np.asarray(rr_features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
    )


def oversample_train(signals, rr_features, labels, seed):
    counts = np.bincount(labels, minlength=4)
    strategy = {
        1: max(int(counts[1]), int(counts[0] * 0.15)),
        3: max(int(counts[3]), int(counts[0] * 0.08)),
    }
    sample_indices = np.arange(len(labels), dtype=np.int64).reshape(-1, 1)
    sampler = RandomOverSampler(sampling_strategy=strategy, random_state=seed)
    resampled_indices, resampled_labels = sampler.fit_resample(sample_indices, labels)
    resampled_indices = resampled_indices.ravel()
    return (
        signals[resampled_indices],
        rr_features[resampled_indices],
        resampled_labels.astype(np.int64),
    )


class HybridDataset(Dataset):
    def __init__(self, signals, rr_features, labels):
        self.signals = torch.from_numpy(signals).unsqueeze(1)
        self.rr_features = torch.from_numpy(rr_features).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.signals[index], self.rr_features[index], self.labels[index]


class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.convolutions = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(x + self.convolutions(x))


class HybridCNN1D(nn.Module):
    def __init__(self, n_classes=4, num_rr_features=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.layer1 = nn.Sequential(ResBlock1D(32), nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ResBlock1D(64),
            nn.MaxPool1d(2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            ResBlock1D(128),
            nn.AdaptiveAvgPool1d(4),
        )
        self.rr_projector = nn.Sequential(nn.Linear(num_rr_features, 32), nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, signal, rr_features):
        features = self.layer3(self.layer2(self.layer1(self.stem(signal))))
        features = torch.flatten(features, 1)
        rr_embedding = self.rr_projector(rr_features)
        return self.classifier(torch.cat((features, rr_embedding), dim=1))


def validation_metrics(model, loader):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for signals, rr_features, batch_labels in loader:
            signals = signals.to(DEVICE, non_blocking=True)
            rr_features = rr_features.to(DEVICE, non_blocking=True)
            logits = model(signals, rr_features)
            predictions.extend(logits.argmax(1).cpu().numpy())
            labels.extend(batch_labels.numpy())

    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    per_class = f1_score(
        labels,
        predictions,
        labels=[0, 1, 2, 3],
        average=None,
        zero_division=0,
    )
    macro = f1_score(labels, predictions, average="macro", zero_division=0)
    return per_class, float(macro)


def train_one_run(
    train_signals,
    train_rr,
    train_labels,
    val_signals,
    val_rr,
    val_labels,
    seed,
):
    set_seed(seed)

    if USE_OVERSAMPLING:
        run_signals, run_rr, run_labels = oversample_train(
            train_signals, train_rr, train_labels, seed
        )
    else:
        run_signals, run_rr, run_labels = train_signals, train_rr, train_labels

    generator = torch.Generator().manual_seed(seed)
    pin_memory = DEVICE.type == "cuda"
    train_loader = DataLoader(
        HybridDataset(run_signals, run_rr, run_labels),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        HybridDataset(val_signals, val_rr, val_labels),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    model = HybridCNN1D(n_classes=4).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS
    )

    criterion = nn.CrossEntropyLoss()

    best_macro = -np.inf
    best_per_class = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        for signals, rr_features, labels in train_loader:
            signals = signals.to(DEVICE, non_blocking=True)
            rr_features = rr_features.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(signals, rr_features), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        per_class, macro = validation_metrics(model, val_loader)
        if macro > best_macro + MIN_DELTA:
            best_macro = macro
            best_per_class = per_class.copy()
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            break

    result = {
        "F1_N": float(best_per_class[0]),
        "F1_S": float(best_per_class[1]),
        "F1_V": float(best_per_class[2]),
        "F1_F": float(best_per_class[3]),
        "Macro_F1": float(best_macro),
        "Min_Rare_F1": float(min(best_per_class[1], best_per_class[3])),
        "Rare_F1_weighted": float(0.4 * best_per_class[1] + 0.6 * best_per_class[3]),
        "best_epoch": int(best_epoch),
        "n_train_after_resampling": int(len(run_labels)),
    }

    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return result


def read_progress():
    if not os.path.exists(PROGRESS_FILE):
        return pd.DataFrame()
    return pd.read_csv(PROGRESS_FILE)


def is_completed(progress, phase, window, seed):
    if progress.empty:
        return False
    mask = (
        (progress["phase"] == phase)
        & (progress["window"] == window_name(window))
        & (progress["seed"] == seed)
    )
    return bool(mask.any())


def append_progress(row):
    file_exists = os.path.exists(PROGRESS_FILE)
    pd.DataFrame([row]).to_csv(
        PROGRESS_FILE,
        mode="a",
        header=not file_exists,
        index=False,
    )


def evaluate_windows(phase, windows, seeds):
    progress = read_progress()

    for window_before, window_after in windows:
        window = (window_before, window_after)
        pending_seeds = [
            seed for seed in seeds
            if not is_completed(progress, phase, window, seed)
        ]
        if not pending_seeds:
            continue

        train_data = segment_records(
            DS1_TRAIN_RECORDS, window_before, window_after
        )
        val_data = segment_records(
            DS1_VAL_RECORDS, window_before, window_after
        )

        for seed in pending_seeds:
            start_time = time.time()
            metrics = train_one_run(*train_data, *val_data, seed=seed)
            elapsed = time.time() - start_time
            row = {
                "phase": phase,
                "window": window_name(window),
                "window_before": window_before,
                "window_after": window_after,
                "seed": seed,
                "n_train_raw": len(train_data[2]),
                "n_val": len(val_data[2]),
                "time_s": round(elapsed, 1),
                **metrics,
            }
            append_progress(row)
            progress = pd.concat([progress, pd.DataFrame([row])], ignore_index=True)

        del train_data, val_data
        gc.collect()


def aggregate_phase(phase, windows, seeds):
    progress = read_progress()
    requested_names = {window_name(window) for window in windows}
    phase_rows = progress[
        (progress["phase"] == phase)
        & (progress["window"].isin(requested_names))
        & (progress["seed"].isin(seeds))
    ].copy()
    phase_rows = phase_rows.drop_duplicates(
        subset=["phase", "window", "seed"], keep="last"
    )

    counts = phase_rows.groupby("window")["seed"].nunique()
    incomplete = {
        name: len(seeds) - int(counts.get(name, 0))
        for name in requested_names
        if int(counts.get(name, 0)) != len(seeds)
    }
    if incomplete:
        raise RuntimeError(f"Niekompletne przebiegi fazy {phase}: {incomplete}")

    rows = []
    for window in windows:
        name = window_name(window)
        data = phase_rows[phase_rows["window"] == name].sort_values("seed")
        row = {
            "window": name,
            "window_before": window[0],
            "window_after": window[1],
            "n_runs": len(data),
            "Macro_mean": data["Macro_F1"].mean(),
            "Macro_std": data["Macro_F1"].std(ddof=1),
            "F1_N_mean": data["F1_N"].mean(),
            "F1_S_mean": data["F1_S"].mean(),
            "F1_V_mean": data["F1_V"].mean(),
            "F1_F_mean": data["F1_F"].mean(),
            "Min_Rare_mean": data["Min_Rare_F1"].mean(),
            "Rare_weighted_mean": data["Rare_F1_weighted"].mean(),
            "best_epoch_median": data["best_epoch"].median(),
            "time_s": data["time_s"].sum(),
        }
        rows.append(row)

    ranking = pd.DataFrame(rows).sort_values(
        by=["Macro_mean", "Macro_std", "Rare_weighted_mean"],
        ascending=[False, True, False],
    )
    return ranking.reset_index(drop=True), phase_rows


def bootstrap_ci_diff(a, b, n_boot=10000, confidence=95, seed=42):
    rng = np.random.default_rng(seed)
    differences = np.asarray(a) - np.asarray(b)
    boot_means = np.asarray([
        np.mean(rng.choice(differences, size=len(differences), replace=True))
        for _ in range(n_boot)
    ])
    tail = (100 - confidence) / 2
    return float(np.percentile(boot_means, tail)), float(
        np.percentile(boot_means, 100 - tail)
    )


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running_max = 0.0
    m = len(p_values)
    for rank, original_idx in enumerate(order):
        candidate = min(1.0, (m - rank) * p_values[original_idx])
        running_max = max(running_max, candidate)
        adjusted[original_idx] = running_max
    return adjusted


def compare_with_baseline(final_rows, finalist_windows, seeds):
    baseline_name = window_name(BASELINE_WINDOW)
    baseline = final_rows[final_rows["window"] == baseline_name].sort_values("seed")
    if len(baseline) != len(seeds):
        raise RuntimeError("Brakuje przebiegów okna bazowego w fazie confirm.")

    comparisons = []
    for window in finalist_windows:
        name = window_name(window)
        if name == baseline_name:
            continue

        candidate = final_rows[final_rows["window"] == name].sort_values("seed")
        a = candidate["Macro_F1"].to_numpy()
        b = baseline["Macro_F1"].to_numpy()
        _, p_ttest = stats.ttest_rel(a, b)
        try:
            _, p_wilcoxon = stats.wilcoxon(a, b)
        except ValueError:
            p_wilcoxon = 1.0
        ci_low, ci_high = bootstrap_ci_diff(a, b)
        differences = a - b
        effect = float(
            np.mean(differences) / (np.std(differences, ddof=1) + 1e-8)
        )
        comparisons.append({
            "window": name,
            "mean_diff_macro": float(np.mean(differences)),
            "p_ttest": float(p_ttest),
            "p_wilcoxon": float(p_wilcoxon),
            "CI95_low": ci_low,
            "CI95_high": ci_high,
            "cohens_d_paired": effect,
        })

    result = pd.DataFrame(comparisons)
    result["p_ttest_holm"] = holm_adjust(result["p_ttest"].to_numpy())
    result["significant_improvement"] = (
        (result["mean_diff_macro"] > 0)
        & (result["p_ttest_holm"] < ALPHA)
        & (result["CI95_low"] > 0)
    )
    return result.sort_values("mean_diff_macro", ascending=False)


def main():
    assert set(DS1_TRAIN_RECORDS).isdisjoint(DS1_VAL_RECORDS)
    assert len(DS1_TRAIN_RECORDS) == 16
    assert len(DS1_VAL_RECORDS) == 6
    assert BASELINE_WINDOW in CANDIDATE_WINDOWS

    evaluate_windows("screen", CANDIDATE_WINDOWS, SCREEN_SEEDS)
    screen_ranking, _ = aggregate_phase(
        "screen", CANDIDATE_WINDOWS, SCREEN_SEEDS
    )
    screen_path = os.path.join(OUTPUT_DIR, "screen_ranking.csv")
    screen_ranking.to_csv(screen_path, index=False)

    top_names = screen_ranking.head(TOP_K)["window"].tolist()
    finalists = [
        window for window in CANDIDATE_WINDOWS
        if window_name(window) in top_names
    ]
    if BASELINE_WINDOW not in finalists:
        finalists.append(BASELINE_WINDOW)
    finalists = sorted(finalists)

    evaluate_windows("confirm", finalists, CONFIRM_SEEDS)
    final_ranking, final_rows = aggregate_phase(
        "confirm", finalists, CONFIRM_SEEDS
    )
    final_path = os.path.join(OUTPUT_DIR, "final_ranking.csv")
    final_ranking.to_csv(final_path, index=False)

    comparisons = compare_with_baseline(final_rows, finalists, CONFIRM_SEEDS)
    comparisons_path = os.path.join(OUTPUT_DIR, "comparisons_vs_90x144.csv")
    comparisons.to_csv(comparisons_path, index=False)

    winner = final_ranking.iloc[0]
    winner_name = winner["window"]
    baseline_name = window_name(BASELINE_WINDOW)

    decision = {
        "primary_metric": "mean validation Macro F1",
        "selected_window": winner_name,
        "window_before": int(winner["window_before"]),
        "window_after": int(winner["window_after"]),
        "Macro_mean": float(winner["Macro_mean"]),
        "Macro_std": float(winner["Macro_std"]),
        "baseline_window": baseline_name,
        "ds2_used_for_selection": False,
        "screen_seeds": SCREEN_SEEDS,
        "confirm_seeds": CONFIRM_SEEDS,
    }
    with open(
        os.path.join(OUTPUT_DIR, "selected_window.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(decision, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
