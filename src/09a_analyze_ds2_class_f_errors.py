from __future__ import annotations

import json
import os

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


OUTPUT_DIR = Path("/kaggle/working/ds2_error_analysis_per_record")
SEARCH_ROOTS = [Path("/kaggle/input"), Path("/kaggle/working")]

WINDOW_BEFORE = 65
WINDOW_AFTER = 110
EXPECTED_LENGTH = 175

CLASS_NAMES = ["N", "S", "V", "F"]
LABEL_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}

AAMI_MAPPING = {
    ".": "N", "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}
BEAT_SYMBOLS = set(AAMI_MAPPING)

SPLIT_RECORDS = {
    "DS1_TRAIN": [
        "101", "106", "108", "109", "112", "115", "116", "119",
        "124", "201", "203", "205", "208", "209", "215", "230",
    ],
    "DS1_VAL": ["114", "118", "122", "207", "220", "223"],
    "DS2_TEST": [
        "100", "103", "105", "111", "113", "117", "121", "123",
        "200", "202", "210", "212", "213", "214", "219", "221",
        "222", "228", "231", "232", "233", "234",
    ],
}

NPZ_FILENAME_BY_SPLIT = {
    "DS1_TRAIN": "mitbih_train.npz",
    "DS1_VAL": "mitbih_val.npz",
    "DS2_TEST": "mitbih_test.npz",
}

EXPECTED_FINAL_CONFIGS = {
    ("RAW_1D", "BASELINE_CE"),
    ("RAW_1D", "FOCAL_A025_G3"),
    ("STFT_2D", "BASELINE_CE"),
    ("STFT_2D", "WGAN_GP_X5"),
}
EXPECTED_FINAL_SEEDS = list(range(7100, 7110))
EXPECTED_FINAL_RUNS = len(EXPECTED_FINAL_CONFIGS) * len(EXPECTED_FINAL_SEEDS)


def iter_files(filename: str):

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


def find_dataset_paths() -> dict[str, Path]:
    candidates: dict[Path, dict[str, Path]] = {}
    for split_name, filename in NPZ_FILENAME_BY_SPLIT.items():
        for path in iter_files(filename):
            candidates.setdefault(path.parent, {})[split_name] = path

    complete = [mapping for mapping in candidates.values() if len(mapping) == 3]
    if not complete:
        raise FileNotFoundError(
            "Nie znaleziono jednego katalogu zawierajacego mitbih_train.npz, "
            "mitbih_val.npz i mitbih_test.npz. Dodaj datasetostrrfixed_65x110 "
            "jako Kaggle Input."
        )

    def score(mapping):
        try:
            with np.load(mapping["DS2_TEST"], allow_pickle=False) as data:
                n_test = len(data["Y"])
            return int(n_test == 49_693)
        except Exception:
            return 0

    complete.sort(key=score, reverse=True)
    result = complete[0]
    if score(result) != 1:
        raise RuntimeError(
            "Znaleziony mitbih_test.npz nie ma oczekiwanych 49 693 uderzen."
        )
    return result


def find_raw_mitdb_dir() -> Path:
    all_records = sorted({record for records in SPLIT_RECORDS.values() for record in records})
    candidates = []
    for path in iter_files("100.hea"):
        parent = path.parent
        if all(
            (parent / f"{record}.hea").exists()
            and (parent / f"{record}.atr").exists()
            for record in all_records
        ):
            candidates.append(parent)
    if not candidates:
        raise FileNotFoundError(
            "Nie znaleziono surowej bazy MIT-BIH. Dodaj Kaggle Input z plikami "
            "100.hea, 100.atr, ..., 234.hea, 234.atr."
        )
    candidates.sort(key=lambda path: ("mit" not in str(path).lower(), len(str(path))))
    return candidates[0]


def find_final_results_dir() -> tuple[Path, dict]:
    candidates = []
    for manifest_path in iter_files("final_ds2_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if manifest.get("protocol") != "locked final evaluation on DS2 after selection on DS1":
            continue
        prediction_count = len(list(manifest_path.parent.glob("test_predictions_*.npz")))
        candidates.append((prediction_count, manifest_path.parent, manifest))

    if not candidates:
        raise FileNotFoundError(
            "Nie znaleziono final_ds2_manifest.json. Dodaj pelny output finalnego testu DS2."
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    prediction_count, result_dir, manifest = candidates[0]
    if prediction_count < EXPECTED_FINAL_RUNS:
        raise FileNotFoundError(
            f"Znaleziono tylko {prediction_count}/{EXPECTED_FINAL_RUNS} plikow "
            "test_predictions_*.npz. Zapisz CALY katalog finalnego eksperymentu "
            "jako Kaggle Dataset i dodaj go jako Input. Same pliki CSV nie wystarcza."
        )
    if not bool(manifest.get("complete")) or int(manifest.get("completed_runs", -1)) != 40:
        raise RuntimeError("Manifest finalnego eksperymentu nie potwierdza kompletnych 40 runow.")
    if bool(manifest.get("ds2_used_for_training")):
        raise RuntimeError("Manifest wskazuje uzycie DS2 w treningu; analiza zostaje przerwana.")
    return result_dir, manifest


def labels_to_ids(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.integer):
        result = values.astype(np.int8)
    else:
        normalized = [
            value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
            for value in values
        ]
        result = np.asarray([LABEL_TO_ID[value] for value in normalized], dtype=np.int8)
    if not set(np.unique(result)).issubset({0, 1, 2, 3}):
        raise ValueError(f"Nieznane etykiety: {np.unique(result)}")
    return result


def load_npz_labels(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "Y" not in data.files:
            raise KeyError(f"{path}: brak tablicy Y.")
        labels = labels_to_ids(data["Y"])
        if "X" in data.files and data["X"].shape != (len(labels), EXPECTED_LENGTH):
            raise RuntimeError(
                f"{path}: oczekiwano X=(n,{EXPECTED_LENGTH}), otrzymano {data['X'].shape}."
            )
        if "RR" in data.files and data["RR"].shape != (len(labels), 4):
            raise RuntimeError(f"{path}: nieoczekiwany ksztalt RR={data['RR'].shape}.")
    return labels


def reconstruct_split_metadata(
    raw_mitdb_dir: Path,
    split_name: str,
    records: list[str],
) -> pd.DataFrame:
    rows = []
    split_index = 0
    for record_order, record_id in enumerate(records):
        record_path = str(raw_mitdb_dir / record_id)
        header = wfdb.rdheader(record_path)
        annotation = wfdb.rdann(record_path, "atr")
        signal_length = int(header.sig_len)

        for annotation_index, (sample, symbol) in enumerate(
            zip(annotation.sample, annotation.symbol)
        ):
            if symbol not in BEAT_SYMBOLS:
                continue
            class_name = AAMI_MAPPING[symbol]
            if class_name == "Q":
                continue
            sample = int(sample)
            start_idx = sample - WINDOW_BEFORE
            end_idx = sample + WINDOW_AFTER

            if start_idx < 0 or end_idx >= signal_length:
                continue
            rows.append({
                "split": split_name,
                "split_index": split_index,
                "record_order": record_order,
                "record_id": record_id,
                "annotation_index": int(annotation_index),
                "sample": sample,
                "original_symbol": symbol,
                "class_id": LABEL_TO_ID[class_name],
                "class_name": class_name,
            })
            split_index += 1
    return pd.DataFrame(rows)


def reconstruct_and_validate_all_metadata(
    raw_mitdb_dir: Path,
    dataset_paths: dict[str, Path],
) -> pd.DataFrame:
    frames = []
    for split_name, records in SPLIT_RECORDS.items():
        frame = reconstruct_split_metadata(raw_mitdb_dir, split_name, records)
        npz_labels = load_npz_labels(dataset_paths[split_name])
        reconstructed_labels = frame["class_id"].to_numpy(dtype=np.int8)
        if len(frame) != len(npz_labels):
            raise RuntimeError(
                f"{split_name}: odtworzono {len(frame)} uderzen, a NPZ ma "
                f"{len(npz_labels)}. Nie wolno laczyc predykcji z record_id. "
                "Sprawdz wersje danych i parametry segmentacji."
            )
        mismatch = np.flatnonzero(reconstructed_labels != npz_labels)
        if len(mismatch):
            first = int(mismatch[0])
            raise RuntimeError(
                f"{split_name}: kolejnosc klas nie zgadza sie z NPZ od indeksu "
                f"{first}. Analiza przerwana, aby nie przypisac predykcji do "
                "niewlasciwych pacjentow."
            )
        frames.append(frame)


    return pd.concat(frames, ignore_index=True)


def scalar_string(value) -> str:
    array = np.asarray(value)
    scalar = array.item() if array.ndim == 0 else array.reshape(-1)[0]
    return scalar.decode("utf-8") if isinstance(scalar, (bytes, np.bytes_)) else str(scalar)


def scalar_int(value) -> int:
    array = np.asarray(value)
    return int(array.item() if array.ndim == 0 else array.reshape(-1)[0])


def load_prediction_runs(
    final_results_dir: Path,
    expected_y_true: np.ndarray,
    manifest: dict,
) -> list[dict]:
    files = sorted(final_results_dir.glob("test_predictions_*.npz"))
    runs = []
    keys = set()
    selected = manifest.get("selected_config_by_representation", {})

    for path in files:
        try:
            with np.load(path, allow_pickle=False) as data:
                representation = scalar_string(data["REPRESENTATION"])
                config_id = scalar_string(data["CONFIG_ID"])
                seed = scalar_int(data["SEED"])
                y_true = labels_to_ids(data["Y_TRUE"])
                y_pred = labels_to_ids(data["Y_PRED"])
        except Exception as exc:
            raise RuntimeError(f"Nie mozna odczytac {path}: {exc}") from exc

        key = (representation, config_id, seed)
        if (representation, config_id) not in EXPECTED_FINAL_CONFIGS:
            continue
        if seed not in EXPECTED_FINAL_SEEDS:
            continue
        if key in keys:
            raise RuntimeError(f"Duplikat predykcji: {key}")
        if len(y_true) != len(expected_y_true) or len(y_pred) != len(expected_y_true):
            raise RuntimeError(f"{path}: niezgodna liczba predykcji.")
        if not np.array_equal(y_true, expected_y_true):
            raise RuntimeError(f"{path}: Y_TRUE nie zgadza sie z mitbih_test.npz.")

        role = "selected_on_DS1" if selected.get(representation) == config_id else "baseline"
        runs.append({
            "representation": representation,
            "config_id": config_id,
            "role": role,
            "seed": seed,
            "path": str(path),
            "y_true": y_true,
            "y_pred": y_pred,
        })
        keys.add(key)

    expected_keys = {
        (representation, config_id, seed)
        for representation, config_id in EXPECTED_FINAL_CONFIGS
        for seed in EXPECTED_FINAL_SEEDS
    }
    missing = sorted(expected_keys - keys)
    if missing:
        raise RuntimeError(f"Brakuje {len(missing)} runow predykcji: {missing[:5]}")
    if len(runs) != EXPECTED_FINAL_RUNS:
        raise RuntimeError(f"Oczekiwano 40 runow, wczytano {len(runs)}.")
    return sorted(runs, key=lambda row: (row["representation"], row["config_id"], row["seed"]))


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def confusion_metrics(matrix: np.ndarray) -> dict:
    matrix = np.asarray(matrix, dtype=float)
    total = float(matrix.sum())
    row = matrix.sum(axis=1)
    col = matrix.sum(axis=0)
    result = {"Accuracy": safe_div(float(np.trace(matrix)), total)}
    present_f1 = []
    present_rare_f1 = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        tp = float(matrix[class_index, class_index])
        fn = float(row[class_index] - tp)
        fp = float(col[class_index] - tp)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        if row[class_index] > 0:
            denominator = 2.0 * tp + fp + fn
            class_f1 = safe_div(2.0 * tp, denominator)
            present_f1.append(class_f1)
            if class_name in {"S", "F"}:
                present_rare_f1.append(class_f1)
        else:
            class_f1 = float("nan")

        result.update({
            f"Support_{class_name}": int(row[class_index]),
            f"Predicted_{class_name}": int(col[class_index]),
            f"TP_{class_name}": int(tp),
            f"FP_{class_name}": int(fp),
            f"FN_{class_name}": int(fn),
            f"Precision_{class_name}": precision,
            f"Recall_{class_name}": recall,
            f"F1_{class_name}": class_f1,
        })

    result["Macro_F1_present_classes"] = (
        float(np.nanmean(present_f1)) if present_f1 else float("nan")
    )
    result["Rare_Macro_present_classes"] = (
        float(np.nanmean(present_rare_f1)) if present_rare_f1 else float("nan")
    )
    result["Min_Rare_present_classes"] = (
        float(np.nanmin(present_rare_f1)) if present_rare_f1 else float("nan")
    )
    return result


def global_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    f1 = f1_score(y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    precision = precision_score(
        y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0
    )
    result = {
        "Accuracy": float(np.mean(y_true == y_pred)),
        "Macro_F1": float(np.mean(f1)),
        "Rare_Macro_SF": float((f1[1] + f1[3]) / 2.0),
        "Min_Rare_F1": float(min(f1[1], f1[3])),
    }
    for index, class_name in enumerate(CLASS_NAMES):
        result[f"F1_{class_name}"] = float(f1[index])
        result[f"Precision_{class_name}"] = float(precision[index])
        result[f"Recall_{class_name}"] = float(recall[index])
    return result


def build_record_distribution(metadata: pd.DataFrame) -> pd.DataFrame:
    table = (
        metadata.groupby(["split", "record_order", "record_id", "class_name"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CLASS_NAMES, fill_value=0)
        .reset_index()
    )
    table.columns.name = None
    table = table.rename(columns={name: f"Count_{name}" for name in CLASS_NAMES})
    table["Total"] = table[[f"Count_{name}" for name in CLASS_NAMES]].sum(axis=1)
    table["Rare_SF"] = table["Count_S"] + table["Count_F"]
    return table.sort_values(["split", "record_order"]).reset_index(drop=True)


def analyze_per_record(
    runs: list[dict],
    test_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    record_ids = test_metadata["record_id"].to_numpy(str)
    unique_records = list(dict.fromkeys(record_ids))
    run_rows = []
    global_rows = []

    for run in runs:
        y_true = run["y_true"]
        y_pred = run["y_pred"]
        global_row = {
            key: run[key] for key in ["representation", "config_id", "role", "seed"]
        }
        global_row.update(global_metrics(y_true, y_pred))
        global_rows.append(global_row)

        for record_id in unique_records:
            mask = record_ids == record_id
            matrix = confusion_matrix(y_true[mask], y_pred[mask], labels=[0, 1, 2, 3])
            row = {
                key: run[key] for key in ["representation", "config_id", "role", "seed"]
            }
            row["record_id"] = record_id
            row["n_beats"] = int(mask.sum())
            row["confusion_json"] = json.dumps(matrix.astype(int).tolist())
            row.update(confusion_metrics(matrix))
            run_rows.append(row)

    per_record_runs = pd.DataFrame(run_rows)
    global_recheck = pd.DataFrame(global_rows)

    metric_columns = [
        "Accuracy", "Macro_F1_present_classes", "Rare_Macro_present_classes",
        "Min_Rare_present_classes",
        *[
            f"{prefix}_{class_name}"
            for class_name in CLASS_NAMES
            for prefix in ["Precision", "Recall", "F1"]
        ],
    ]
    group_columns = ["representation", "config_id", "role", "record_id"]
    summary = per_record_runs.groupby(group_columns, sort=False).agg(
        n_seeds=("seed", "nunique"),
        n_beats=("n_beats", "first"),
        **{
            f"{metric}_mean": (metric, "mean")
            for metric in metric_columns
        },
        **{
            f"{metric}_std": (metric, "std")
            for metric in metric_columns
        },
        **{
            f"Support_{class_name}": (f"Support_{class_name}", "first")
            for class_name in CLASS_NAMES
        },
    ).reset_index()
    summary = summary.sort_values(
        ["representation", "config_id", "Support_F", "Support_S", "record_id"],
        ascending=[True, True, False, False, True],
    ).reset_index(drop=True)
    return per_record_runs, summary, global_recheck


def build_f_error_destinations(
    per_record_runs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    config_columns = ["representation", "config_id", "role"]
    for config_values, config_group in per_record_runs.groupby(config_columns, sort=False):
        representation, config_id, role = config_values
        record_groups = [("ALL_DS2", config_group)] + list(config_group.groupby("record_id"))
        for record_id, group in record_groups:
            matrices = np.stack([
                np.asarray(json.loads(value), dtype=int)
                for value in group["confusion_json"]
            ])
            f_row = matrices[:, LABEL_TO_ID["F"], :].sum(axis=0)
            support_across_seeds = int(f_row.sum())
            if support_across_seeds == 0:
                continue
            row = {
                "representation": representation,
                "config_id": config_id,
                "role": role,
                "record_id": str(record_id),
                "n_seeds": int(group["seed"].nunique()),
                "true_F_evaluations": support_across_seeds,
            }
            for class_index, class_name in enumerate(CLASS_NAMES):
                row[f"F_to_{class_name}_count"] = int(f_row[class_index])
                row[f"F_to_{class_name}_fraction"] = safe_div(
                    float(f_row[class_index]), float(support_across_seeds)
                )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["record_id", "representation", "config_id"]
    ).reset_index(drop=True)


def build_true_f_consistency(
    runs: list[dict],
    test_metadata: pd.DataFrame,
) -> pd.DataFrame:
    true_f_mask = test_metadata["class_id"].to_numpy() == LABEL_TO_ID["F"]
    f_metadata = test_metadata.loc[
        true_f_mask,
        ["split_index", "record_id", "sample", "original_symbol"],
    ].reset_index(drop=True)
    rows = []
    selected_map = {}
    for run in runs:
        selected_map.setdefault(
            (run["representation"], run["config_id"], run["role"]), []
        ).append(run)

    for (representation, config_id, role), config_runs in selected_map.items():
        config_runs = sorted(config_runs, key=lambda row: row["seed"])
        predictions = np.stack([run["y_pred"][true_f_mask] for run in config_runs])
        for beat_position in range(predictions.shape[1]):
            counts = np.bincount(predictions[:, beat_position], minlength=4)
            metadata_row = f_metadata.iloc[beat_position]
            row = {
                "representation": representation,
                "config_id": config_id,
                "role": role,
                "split_index": int(metadata_row["split_index"]),
                "record_id": str(metadata_row["record_id"]),
                "sample": int(metadata_row["sample"]),
                "original_symbol": str(metadata_row["original_symbol"]),
                "n_seeds": int(len(config_runs)),
            }
            for class_index, class_name in enumerate(CLASS_NAMES):
                row[f"predicted_{class_name}_seeds"] = int(counts[class_index])
            row["correct_seed_fraction"] = safe_div(
                float(counts[LABEL_TO_ID["F"]]), float(len(config_runs))
            )
            row["all_seeds_missed_F"] = bool(counts[LABEL_TO_ID["F"]] == 0)
            row["all_seeds_predicted_N"] = bool(counts[LABEL_TO_ID["N"]] == len(config_runs))
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["correct_seed_fraction", "record_id", "sample", "representation", "config_id"]
    ).reset_index(drop=True)


def config_label(representation: str, config_id: str) -> str:
    return f"{representation}\n{config_id}"


def save_f_error_plot(f_error_destinations: pd.DataFrame) -> None:
    data = f_error_destinations[f_error_destinations["record_id"] == "ALL_DS2"].copy()
    data = data.sort_values(["representation", "config_id"])
    if data.empty:
        return
    labels = [
        config_label(row.representation, row.config_id)
        for row in data.itertuples(index=False)
    ]
    x = np.arange(len(data))
    bottom = np.zeros(len(data), dtype=float)
    colors = {"N": "#64748B", "S": "#F59E0B", "V": "#3B82F6", "F": "#16A34A"}
    fig, ax = plt.subplots(figsize=(10, 6))
    for class_name in CLASS_NAMES:
        values = data[f"F_to_{class_name}_fraction"].to_numpy(float)
        ax.bar(x, values, bottom=bottom, label=f"F -> {class_name}", color=colors[class_name])
        bottom += values
    ax.set_xticks(x, labels)
    ax.set_ylabel("Odsetek prawdziwych uderzen F")
    ax.set_ylim(0, 1)
    ax.set_title("Kierunki pomylek klasy F na DS2 (suma 10 seedow)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "f_error_destinations_overall.png", dpi=180)
    plt.close(fig)


def save_f_recall_by_record_plot(per_record_summary: pd.DataFrame) -> None:
    data = per_record_summary[per_record_summary["Support_F"] > 0].copy()
    if data.empty:
        return
    records = sorted(data["record_id"].unique())
    configs = list(
        data[["representation", "config_id"]].drop_duplicates().itertuples(index=False, name=None)
    )
    width = 0.8 / max(1, len(configs))
    x = np.arange(len(records))
    fig, ax = plt.subplots(figsize=(max(10, len(records) * 1.2), 6))
    for index, (representation, config_id) in enumerate(configs):
        subset = data[
            (data["representation"] == representation)
            & (data["config_id"] == config_id)
        ].set_index("record_id")
        values = [subset.loc[record, "Recall_F_mean"] if record in subset.index else np.nan for record in records]
        offset = (index - (len(configs) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            label=config_label(representation, config_id).replace("\n", " + "),
        )
    support = (
        data.drop_duplicates("record_id").set_index("record_id")["Support_F"].to_dict()
    )
    ax.set_xticks(x, [f"{record}\nF={int(support[record])}" for record in records])
    ax.set_ylabel("Recall F: srednia po 10 seedach")
    ax.set_ylim(0, 1)
    ax.set_title("Wykrywanie F osobno dla rekordow DS2")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "f_recall_by_record.png", dpi=180)
    plt.close(fig)


def split_distribution(metadata: pd.DataFrame) -> pd.DataFrame:
    table = (
        metadata.groupby(["split", "class_name"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CLASS_NAMES, fill_value=0)
        .reset_index()
    )
    table.columns.name = None
    table = table.rename(columns={name: f"Count_{name}" for name in CLASS_NAMES})
    table["Total"] = table[[f"Count_{name}" for name in CLASS_NAMES]].sum(axis=1)
    for class_name in CLASS_NAMES:
        table[f"Pct_{class_name}"] = table[f"Count_{class_name}"] / table["Total"]
    return table


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_paths = find_dataset_paths()
    raw_mitdb_dir = find_raw_mitdb_dir()
    final_results_dir, final_manifest = find_final_results_dir()


    metadata = reconstruct_and_validate_all_metadata(raw_mitdb_dir, dataset_paths)
    metadata.to_csv(OUTPUT_DIR / "all_split_beat_metadata.csv", index=False)

    record_distribution = build_record_distribution(metadata)
    record_distribution.to_csv(OUTPUT_DIR / "record_class_distribution.csv", index=False)
    split_dist = split_distribution(metadata)
    split_dist.to_csv(OUTPUT_DIR / "split_class_distribution.csv", index=False)

    test_metadata = metadata[metadata["split"] == "DS2_TEST"].copy()
    test_metadata = test_metadata.sort_values("split_index").reset_index(drop=True)
    expected_y_true = load_npz_labels(dataset_paths["DS2_TEST"])
    runs = load_prediction_runs(final_results_dir, expected_y_true, final_manifest)


    per_record_runs, per_record_summary, global_recheck = analyze_per_record(
        runs, test_metadata
    )
    f_error_destinations = build_f_error_destinations(per_record_runs)
    true_f_consistency = build_true_f_consistency(runs, test_metadata)

    per_record_runs.to_csv(OUTPUT_DIR / "per_record_runs.csv", index=False)
    per_record_summary.to_csv(OUTPUT_DIR / "per_record_summary.csv", index=False)
    global_recheck.to_csv(OUTPUT_DIR / "global_metrics_from_predictions.csv", index=False)
    f_error_destinations.to_csv(OUTPUT_DIR / "f_error_destinations.csv", index=False)
    true_f_consistency.to_csv(OUTPUT_DIR / "true_f_beat_consistency.csv", index=False)

    f_records = per_record_summary[per_record_summary["Support_F"] > 0].copy()
    f_records.to_csv(OUTPUT_DIR / "f_records_summary.csv", index=False)
    hardest_f = true_f_consistency[true_f_consistency["all_seeds_missed_F"]].copy()
    hardest_f.to_csv(OUTPUT_DIR / "f_beats_missed_by_all_seeds.csv", index=False)

    save_f_error_plot(f_error_destinations)
    save_f_recall_by_record_plot(per_record_summary)

    manifest = {
        "protocol": "post-hoc per-record error analysis of locked DS2 predictions",
        "analysis_only": True,
        "models_trained": False,
        "model_selection_performed": False,
        "threshold_selection_performed": False,
        "ds2_result_remains_locked": True,
        "raw_window": {
            "before": WINDOW_BEFORE,
            "after": WINDOW_AFTER,
            "length": EXPECTED_LENGTH,
        },
        "raw_mitdb_dir": str(raw_mitdb_dir),
        "dataset_paths": {key: str(value) for key, value in dataset_paths.items()},
        "final_results_dir": str(final_results_dir),
        "n_prediction_runs": len(runs),
        "n_ds2_beats": int(len(test_metadata)),
        "n_ds2_true_f": int((test_metadata["class_name"] == "F").sum()),
        "output_files": [
            "all_split_beat_metadata.csv",
            "record_class_distribution.csv",
            "split_class_distribution.csv",
            "global_metrics_from_predictions.csv",
            "per_record_runs.csv",
            "per_record_summary.csv",
            "f_records_summary.csv",
            "f_error_destinations.csv",
            "true_f_beat_consistency.csv",
            "f_beats_missed_by_all_seeds.csv",
            "f_error_destinations_overall.png",
            "f_recall_by_record.png",
        ],
        "interpretation_warning": (
            "The analysis diagnoses the already observed DS2 errors. It must not "
            "be used to retune, reselect or relabel the final DS2 model."
        ),
    }
    (OUTPUT_DIR / "error_analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

