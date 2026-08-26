import json
import os

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt


FS = 360
WINDOW_BEFORE = 65
WINDOW_AFTER = 110
SEGMENT_LENGTH = WINDOW_BEFORE + WINDOW_AFTER

MITDB_PATH = os.environ.get("MITDB_PATH", "mitdb")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "datasetostrrfixed_65x110")

AAMI_MAPPING = {
    ".": "N",
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    "V": "V",
    "E": "V",
    "F": "F",
    "/": "Q",
    "f": "Q",
    "Q": "Q",
}
BEAT_SYMBOLS = set(AAMI_MAPPING)
OUTPUT_CLASSES = {"N", "S", "V", "F"}

TRAIN_RECORDS = [
    "101",
    "106",
    "108",
    "109",
    "112",
    "115",
    "116",
    "119",
    "124",
    "201",
    "203",
    "205",
    "208",
    "209",
    "215",
    "230",
]
VAL_RECORDS = ["114", "118", "122", "207", "220", "223"]
TEST_RECORDS = [
    "100",
    "103",
    "105",
    "111",
    "113",
    "117",
    "121",
    "123",
    "200",
    "202",
    "210",
    "212",
    "213",
    "214",
    "219",
    "221",
    "222",
    "228",
    "231",
    "232",
    "233",
    "234",
]


def bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=FS, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def normalize_segment(segment):
    mean = float(np.mean(segment))
    std = float(np.std(segment))
    return (segment - mean) / (std + 1e-8)


def select_ecg_channel(record):
    if "MLII" in record.sig_name:
        return record.sig_name.index("MLII")
    if "II" in record.sig_name:
        return record.sig_name.index("II")
    return 0


def process_records(record_list, mitdb_path=MITDB_PATH):
    signals = []
    rr_features = []
    labels = []
    record_ids = []

    for record_name in record_list:
        record_path = os.path.join(mitdb_path, record_name)
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, "atr")

        channel_idx = select_ecg_channel(record)
        raw_signal = record.p_signal[:, channel_idx]
        clean_signal = bandpass_filter(raw_signal)

        all_symbols = np.asarray(annotation.symbol)
        all_samples = np.asarray(annotation.sample)
        beat_mask = np.asarray([symbol in BEAT_SYMBOLS for symbol in all_symbols])
        beat_samples = all_samples[beat_mask]
        beat_symbols = all_symbols[beat_mask]

        pre_rr_all = np.full(len(beat_samples), FS, dtype=np.float32)
        post_rr_all = np.full(len(beat_samples), FS, dtype=np.float32)
        if len(beat_samples) > 1:
            rr_differences = np.diff(beat_samples).astype(np.float32)
            pre_rr_all[1:] = rr_differences
            post_rr_all[:-1] = rr_differences

        for beat_idx, (symbol, position) in enumerate(zip(beat_symbols, beat_samples)):
            aami_class = AAMI_MAPPING[symbol]
            if aami_class == "Q":
                continue

            start_idx = int(position) - WINDOW_BEFORE
            end_idx = int(position) + WINDOW_AFTER
            if start_idx < 0 or end_idx > len(clean_signal):
                continue

            segment = clean_signal[start_idx:end_idx]
            if len(segment) != SEGMENT_LENGTH:
                raise RuntimeError(
                    f"Record {record_name}: segment length {len(segment)}, "
                    f"expected {SEGMENT_LENGTH}."
                )
            segment = normalize_segment(segment)

            local_start = max(0, beat_idx - 10)
            local_mean = (
                float(np.mean(pre_rr_all[local_start:beat_idx]))
                if beat_idx > 0
                else float(FS)
            )
            if local_mean <= 0:
                local_mean = float(FS)

            pre_rr = float(pre_rr_all[beat_idx])
            post_rr = float(post_rr_all[beat_idx])
            rr = [
                pre_rr / FS,
                post_rr / FS,
                local_mean / FS,
                pre_rr / local_mean,
            ]

            signals.append(segment)
            rr_features.append(rr)
            labels.append(aami_class)
            record_ids.append(record_name)

    return (
        np.asarray(signals, dtype=np.float32),
        np.asarray(rr_features, dtype=np.float32),
        np.asarray(labels, dtype="<U1"),
        np.asarray(record_ids, dtype="<U3"),
    )


def validate_dataset(data, expected_records, dataset_name):
    signals, rr_features, labels, record_ids = data

    if signals.ndim != 2 or signals.shape[1] != SEGMENT_LENGTH:
        raise ValueError(f"{dataset_name}: invalid X shape: {signals.shape}")
    if rr_features.shape != (len(signals), 4):
        raise ValueError(f"{dataset_name}: invalid RR shape: {rr_features.shape}")
    if len(labels) != len(signals) or len(record_ids) != len(signals):
        raise ValueError(f"{dataset_name}: inconsistent number of samples")
    if not np.isfinite(signals).all() or not np.isfinite(rr_features).all():
        raise ValueError(f"{dataset_name}: NaN or Inf detected")
    if not set(np.unique(labels)).issubset(OUTPUT_CLASSES):
        raise ValueError(f"{dataset_name}: unexpected classes: {np.unique(labels)}")
    if set(np.unique(record_ids)) != set(expected_records):
        raise ValueError(f"{dataset_name}: record list does not match configuration")


def save_split(filename, data):
    signals, rr_features, labels, record_ids = data
    output_path = os.path.join(OUTPUT_DIR, filename)
    np.savez_compressed(
        output_path,
        X=signals,
        RR=rr_features,
        Y=labels,
        RECORD=record_ids,
        FS=np.int32(FS),
        WINDOW_BEFORE=np.int32(WINDOW_BEFORE),
        WINDOW_AFTER=np.int32(WINDOW_AFTER),
    )


def main():
    assert set(TRAIN_RECORDS).isdisjoint(VAL_RECORDS)
    assert set(TRAIN_RECORDS + VAL_RECORDS).isdisjoint(TEST_RECORDS)
    assert len(TRAIN_RECORDS) == 16
    assert len(VAL_RECORDS) == 6
    assert len(TEST_RECORDS) == 22

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_data = process_records(TRAIN_RECORDS)
    val_data = process_records(VAL_RECORDS)
    test_data = process_records(TEST_RECORDS)

    validate_dataset(train_data, TRAIN_RECORDS, "DS1 TRAIN")
    validate_dataset(val_data, VAL_RECORDS, "DS1 VAL")
    validate_dataset(test_data, TEST_RECORDS, "DS2 TEST")

    save_split("mitbih_train.npz", train_data)
    save_split("mitbih_val.npz", val_data)
    save_split("mitbih_test.npz", test_data)

    manifest = {
        "fs": FS,
        "window_before": WINDOW_BEFORE,
        "window_after": WINDOW_AFTER,
        "segment_length": SEGMENT_LENGTH,
        "filter": {
            "type": "Butterworth band-pass",
            "order": 3,
            "lowcut_hz": 0.5,
            "highcut_hz": 45.0,
            "zero_phase": True,
        },
        "normalization": "per-segment z-score",
        "rr_features": ["pre_rr", "post_rr", "local_mean_10", "rr_ratio"],
        "train_records": TRAIN_RECORDS,
        "val_records": VAL_RECORDS,
        "test_records": TEST_RECORDS,
        "warning": "DS2 must not be used for tuning",
    }
    manifest_path = os.path.join(OUTPUT_DIR, "dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
