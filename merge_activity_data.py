#!/usr/bin/env python3
"""
merge_activity_data.py

This script merges Accelerometer.csv and Gyroscope.csv files for each recording session,
then merges all recordings per activity into a single CSV file.

Merged files are saved under 'data/merged'.

Normalize accelerometer and gyroscope values before saving.
"""

import os
import pandas as pd
import argparse


# Columns to save in output CSV (custom names)
OUTPUT_COLS = ["time", "ax", "ay", "az",
               "gx", "gy", "gz", "activity_type"]


def find_activity_folders(root_folder):
    """Return a list of activity directories under the root folder."""
    return [os.path.join(root_folder, d) for d in os.listdir(root_folder)
            if os.path.isdir(os.path.join(root_folder, d))]


def find_sensor_files(activity_subfolder):
    """
    Return the accelerometer and gyroscope CSV files in a folder.
    Expects filenames exactly: 'Accelerometer.csv' and 'Gyroscope.csv'.
    """
    accel_file = None
    gyro_file = None
    for root, _, files in os.walk(activity_subfolder):
        for f in files:
            if f.lower() == "accelerometer.csv":
                accel_file = os.path.join(root, f)
            elif f.lower() == "gyroscope.csv":
                gyro_file = os.path.join(root, f)
    return accel_file, gyro_file


def detect_timestamp_column(df):
    """Guess which column contains the timestamp."""
    for pattern in ("time", "timestamp", "ts", "seconds", "sec", "elapsed"):
        for col in df.columns:
            if pattern in col.lower():
                return col
    return df.columns[0]


def detect_xyz_columns(df):
    """Detect x, y, z columns for accelerometer or gyroscope."""
    mapping = {}
    for col in df.columns:
        low = col.lower()
        if "time" in low or "timestamp" in low:
            continue
        for axis in ("x", "y", "z"):
            if axis in low:
                mapping[axis] = col
    if len(mapping) < 3:
        raise ValueError(f"Could not detect x, y, z columns in {list(df.columns)}")
    return mapping


def merge_session(accel_file, gyro_file, normalize=False):
    """Merge one accelerometer and gyroscope CSV on timestamp."""
    accel = pd.read_csv(accel_file)
    gyro = pd.read_csv(gyro_file)

    ts_accel = detect_timestamp_column(accel)
    ts_gyro = detect_timestamp_column(gyro)

    common_ts = "time"
    accel = accel.rename(columns={ts_accel: common_ts})
    gyro = gyro.rename(columns={ts_gyro: common_ts})

    accel_xyz = detect_xyz_columns(accel)
    gyro_xyz = detect_xyz_columns(gyro)

    accel_selected = accel[[common_ts, accel_xyz["x"], accel_xyz["y"], accel_xyz["z"]]].rename(
        columns={accel_xyz["x"]: "ax", accel_xyz["y"]: "ay", accel_xyz["z"]: "az"})
    gyro_selected = gyro[[common_ts, gyro_xyz["x"], gyro_xyz["y"], gyro_xyz["z"]]].rename(
        columns={gyro_xyz["x"]: "gx", gyro_xyz["y"]: "gy", gyro_xyz["z"]: "gz"})

    merged = pd.merge(accel_selected, gyro_selected, on=common_ts, how="inner")

    if normalize:
        for col in ["ax", "ay", "az", "gx", "gy", "gz"]:
            merged[col] = (merged[col] - merged[col].mean()) / merged[col].std()

    return merged


def sanitize_filename(name):
    """Return a safe string for filenames."""
    import re
    s = re.sub(r"[^a-zA-Z0-9_-]", "_", name.strip().lower())
    return re.sub(r"_+", "_", s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", "-s", default="data",
                        help="Root folder containing activity subfolders")
    parser.add_argument("--out", "-o", default=os.path.join("data", "merged"),
                        help="Output folder for merged CSVs")
    parser.add_argument("--name", "-n", help="Optional prefix for output filenames")
    parser.add_argument("--normalize", action="store_true",
                        help="Normalize sensor values before saving")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    prefix = sanitize_filename(args.name or "")

    activities = find_activity_folders(args.source_root)
    if not activities:
        print(f"No activity folders found under {args.source_root}")
        return

    for activity_dir in activities:
        activity_name = os.path.basename(activity_dir)
        accel_file, gyro_file = find_sensor_files(activity_dir)

        if not accel_file or not gyro_file:
            print(f"Skipping {activity_name}: missing Accelerometer.csv or Gyroscope.csv")
            continue

        try:
            merged = merge_session(accel_file, gyro_file, normalize=args.normalize)
            merged["activity_type"] = activity_name
            merged = merged.sort_values("time")
        except Exception as e:
            print(f"Error processing {activity_name}: {e}")
            continue

        out_name = f"{prefix}_{activity_name}_merged.csv" if prefix else f"{activity_name}_merged.csv"
        out_path = os.path.join(args.out, out_name)
        merged.to_csv(out_path, index=False, columns=[c for c in OUTPUT_COLS if c in merged.columns])
        print(f"Saved: {out_path} ({merged.shape[0]} rows)")

    print("All merged files saved under:", args.out)


if __name__ == "__main__":
    main()
