#!/usr/bin/env python3
"""
extract_features_from_merged.py

Extracts time- and frequency-domain features from merged accelerometer
and gyroscope CSVs (output of merge_activity_data.py). Saves combined
features into data/features/features.csv.

- Time-domain: mean, std, variance, MAD, SMA, resultant acceleration
- Frequency-domain: dominant frequency, spectral energy, top-k FFT
- Normalization: Z-score per feature
"""

import os
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.signal import welch

WINDOW_SIZE = 128
OVERLAP = 0.5
INPUT_DIR = "data/merged"
OUTPUT_DIR = "data/features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FS = 50.0
TOP_K = 3

def signal_magnitude_area(df, cols):
    return np.mean(np.abs(df[cols]).sum(axis=1))

def _maybe_convert_timestamp(ts):
    if ts is None:
        return None
    try:
        arr = np.array(ts, dtype=float)
    except Exception:
        return None
    if arr.size == 0:
        return None
    meanv = float(np.nanmean(arr))
    if meanv > 1e12:
        return arr / 1e9
    if meanv > 1e9:
        return arr / 1e3
    return arr

def dominant_frequency(signal, fs=FS):
    if len(signal) < 2:
        return 0.0
    yf = np.abs(rfft(signal))
    xf = rfftfreq(len(signal), 1.0/fs)
    if yf.size <= 1:
        return 0.0
    idx = int(np.argmax(yf[1:]) + 1)
    return float(xf[idx])

def spectral_energy(signal, fs=FS):
    if len(signal) < 2:
        return 0.0
    f, Pxx = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    return float(np.sum(Pxx))

def fft_top_k(signal, k=TOP_K, fs=FS):
    if len(signal) < 2:
        return [0.0]*k, [0.0]*k
    yf = np.abs(rfft(signal))
    xf = rfftfreq(len(signal), 1.0/fs)
    if yf.size <= 1:
        return [0.0]*k, [0.0]*k
    mags = yf[1:]
    freqs = xf[1:]
    order = np.argsort(mags)[::-1]
    top_mags = [float(mags[order[i]]) if i < order.size else 0.0 for i in range(k)]
    top_freqs = [float(freqs[order[i]]) if i < order.size else 0.0 for i in range(k)]
    return top_mags, top_freqs

def extract_features_from_window(window):
    features = {}
    for prefix in ["ax","ay","az","gx","gy","gz"]:
        if prefix not in window.columns:
            continue
        arr = window[prefix].values
        features[f"{prefix}_mean"] = float(np.mean(arr))
        features[f"{prefix}_std"] = float(np.std(arr, ddof=0))
        features[f"{prefix}_var"] = float(np.var(arr, ddof=0))
        features[f"{prefix}_mad"] = float(np.mean(np.abs(arr - np.mean(arr))))
        features[f"{prefix}_dom_freq"] = dominant_frequency(arr)
        features[f"{prefix}_spec_energy"] = spectral_energy(arr)
        top_mags, top_freqs = fft_top_k(arr)
        for i in range(TOP_K):
            features[f"{prefix}_fft_top{i+1}_mag"] = top_mags[i]
            features[f"{prefix}_fft_top{i+1}_freq"] = top_freqs[i]
    if all(c in window.columns for c in ["ax","ay","az"]):
        features["acc_sma"] = signal_magnitude_area(window, ["ax","ay","az"])
    else:
        features["acc_sma"] = 0.0
    if all(c in window.columns for c in ["gx","gy","gz"]):
        features["gyro_sma"] = signal_magnitude_area(window, ["gx","gy","gz"])
    else:
        features["gyro_sma"] = 0.0
    if all(c in window.columns for c in ["ax","ay","az"]):
        res = np.sqrt(window["ax"]**2 + window["ay"]**2 + window["az"]**2)
        features["acc_res_mean"] = float(np.mean(res))
        features["acc_res_std"] = float(np.std(res, ddof=0))
        features["acc_res_var"] = float(np.var(res, ddof=0))
        features["acc_res_sma"] = float(np.mean(np.abs(res)))
        features["acc_res_dom_freq"] = dominant_frequency(res)
        features["acc_res_spec_energy"] = spectral_energy(res)
        top_mags_res, top_freqs_res = fft_top_k(res)
        for i in range(TOP_K):
            features[f"acc_res_fft_top{i+1}_mag"] = top_mags_res[i]
            features[f"acc_res_fft_top{i+1}_freq"] = top_freqs_res[i]
    else:
        for key in ["acc_res_mean","acc_res_std","acc_res_var","acc_res_sma",
                    "acc_res_dom_freq","acc_res_spec_energy"] + \
                    [f"acc_res_fft_top{i+1}_{t}" for i in range(TOP_K) for t in ["mag","freq"]]:
            features[key] = 0.0
    return features

def sliding_windows(data_len, window_size, overlap):
    step = int(window_size * (1 - overlap))
    step = max(step, 1)
    for start in range(0, data_len - window_size + 1, step):
        yield start, start + window_size

def process_activity_file(filepath, label):
    df = pd.read_csv(filepath)
    feature_rows = []
    if "time" in df.columns:
        ts = _maybe_convert_timestamp(df["time"].values)
        if ts is not None and len(ts) == len(df):
            df["time"] = ts
    for start, end in sliding_windows(len(df), WINDOW_SIZE, OVERLAP):
        window = df.iloc[start:end].reset_index(drop=True)
        feats = extract_features_from_window(window)
        feats["activity"] = label
        feats["start_time"] = float(window["time"].iloc[0]) if "time" in window.columns else float(start)
        feature_rows.append(feats)
    return pd.DataFrame(feature_rows)

def main():
    print("Extracting features from merged CSV files...")
    all_features = []
    for file in sorted(os.listdir(INPUT_DIR)):
        if not file.endswith(".csv"):
            continue
        label = os.path.basename(file).split("_")[0]
        filepath = os.path.join(INPUT_DIR, file)
        print(f"→ Processing {file} ({label})")
        df_features = process_activity_file(filepath, label)
        print(f"  Extracted {len(df_features)} windows from {file}")
        all_features.append(df_features)
    if all_features:
        final_df = pd.concat(all_features, ignore_index=True)
        feature_cols = [c for c in final_df.columns if c not in ["activity","start_time"]]
        final_df[feature_cols] = (final_df[feature_cols] - final_df[feature_cols].mean()) / final_df[feature_cols].std(ddof=0)
        out_path = os.path.join(OUTPUT_DIR, "features.csv")
        final_df.to_csv(out_path, index=False)
        print(f"Combined features saved to: {out_path}")
        print(f"Total windows: {len(final_df)}")
    else:
        print("No features extracted. Check input files and columns.")

if __name__ == "__main__":
    main()
