#!/usr/bin/env python3
"""
merge_activity_data.py

Merges Accelerometer.csv and Gyroscope.csv for all sessions,
then combines all sessions of the same activity into a single CSV.

Saves merged files in 'data/merged'.
"""

import os
import pandas as pd


# Columns to save in output CSV
OUTPUT_COLS = [
    "time", "ax", "ay", "az", "gx", "gy", "gz",
    "activity_type", "participant", "session"
]


def find_activity_folders(root_folder):
    """Return a list of activity directories under the root folder."""
    activity_dirs = []
    for participant in os.listdir(root_folder):
        participant_path = os.path.join(root_folder, participant)
        if not os.path.isdir(participant_path):
            continue
        for activity in os.listdir(participant_path):
            activity_path = os.path.join(participant_path, activity)
            if os.path.isdir(activity_path):
                activity_dirs.append(activity_path)
    return activity_dirs


def find_sensor_files(activity_folder):
    """Find sessions under activity_folder and return (accel, gyro, session_name) tuples."""
    sessions = []
    roots = {}
    for root, _, files in os.walk(activity_folder):
        for f in files:
            low = f.lower()
            if low == "accelerometer.csv" or low == "gyroscope.csv":
                roots.setdefault(root, {"accel": None, "gyro": None})
                if low == "accelerometer.csv":
                    roots[root]["accel"] = os.path.join(root, f)
                else:
                    roots[root]["gyro"] = os.path.join(root, f)
    for root, pair in roots.items():
        if pair["accel"] and pair["gyro"]:
            session_name = os.path.basename(root)
            sessions.append((pair["accel"], pair["gyro"], session_name))
    return sessions


def detect_timestamp(df):
    """Guess timestamp column."""
    for pattern in ("time", "timestamp", "ts", "seconds", "sec", "elapsed"):
        for col in df.columns:
            if pattern in col.lower():
                return col
    return df.columns[0]


def detect_xyz(df):
    """Detect x, y, z columns."""
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


def merge_session(accel_file, gyro_file):
    """Merge one accelerometer and gyroscope CSV."""
    accel = pd.read_csv(accel_file)
    gyro = pd.read_csv(gyro_file)

    ts_accel = detect_timestamp(accel)
    ts_gyro = detect_timestamp(gyro)

    accel = accel.rename(columns={ts_accel: "time"})
    gyro = gyro.rename(columns={ts_gyro: "time"})

    ax = detect_xyz(accel)
    gx = detect_xyz(gyro)

    accel_sel = accel[["time", ax["x"], ax["y"], ax["z"]]].rename(
        columns={ax["x"]: "ax", ax["y"]: "ay", ax["z"]: "az"})
    gyro_sel = gyro[["time", gx["x"], gx["y"], gx["z"]]].rename(
        columns={gx["x"]: "gx", gx["y"]: "gy", gx["z"]: "gz"})

    merged = pd.merge(accel_sel, gyro_sel, on="time", how="inner")
    return merged


def main():
    root_folder = "data"  # Change if your data is in a different folder
    output_folder = os.path.join(root_folder, "merged")
    os.makedirs(output_folder, exist_ok=True)

    activity_dirs = find_activity_folders(root_folder)
    activity_groups = {}

    # Organize by activity name
    for folder in activity_dirs:
        activity_name = os.path.basename(folder).lower()
        if activity_name not in activity_groups:
            activity_groups[activity_name] = []
        activity_groups[activity_name].append(folder)

    # Merge sessions per activity
    for activity, folders in activity_groups.items():
        merged_sessions = []
        for folder in folders:
            participant = os.path.basename(os.path.dirname(folder))
            sessions = find_sensor_files(folder)
            if not sessions:
                print(f"Skipping {folder}: missing sensor files")
                continue
            for accel, gyro, session_name in sessions:
                try:
                    merged = merge_session(accel, gyro)
                    merged["activity_type"] = activity
                    merged["participant"] = participant
                    merged["session"] = session_name
                    merged_sessions.append(merged)
                except Exception as e:
                    print(f"Error in {folder} ({session_name}): {e}")

        if merged_sessions:
            df_activity = pd.concat(merged_sessions, ignore_index=True)
            df_activity = df_activity.sort_values("time")
            out_file = os.path.join(output_folder, f"{activity}_merged.csv")
            df_activity.to_csv(out_file, index=False, columns=[c for c in OUTPUT_COLS if c in df_activity.columns])
            print(f"Saved: {out_file} ({df_activity.shape[0]} rows)")

    print(f"All merged files saved in '{output_folder}'")


if __name__ == "__main__":
    main()
