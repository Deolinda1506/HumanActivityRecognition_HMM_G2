import pandas as pd
import glob
import os

# Create processed folder if it doesn't exist
if not os.path.exists('data/processed'):
    os.makedirs('data/processed')

# List of activities
activities = ['Standing', 'Walking', 'Jumping', 'Still']

# Loop through each activity
for activity in activities:
    # Find all accelerometer files for this activity
    acc_files = glob.glob(f'data/raw/*{activity}*accelerometer*.csv')
    
    # List to hold merged recordings
    merged_dfs = []
    
    for acc_file in acc_files:
        # Load accelerometer
        acc_df = pd.read_csv(acc_file)
        
        # Try to find corresponding gyroscope file
        gyro_file = acc_file.replace('accelerometer', 'gyroscope')
        try:
            gyro_df = pd.read_csv(gyro_file)
        except FileNotFoundError:
            print(f"Gyroscope file not found for {acc_file}, skipping this recording.")
            continue
        
        # Merge accelerometer + gyroscope on timestamp
        merged_df = pd.merge(acc_df, gyro_df, on='timestamp', suffixes=('_acc', '_gyro'))
        
        # Add activity column if not present
        if 'activity' not in merged_df.columns:
            merged_df['activity'] = activity
        
        merged_dfs.append(merged_df)
    
    # Combine all recordings for this activity
    if merged_dfs:
        activity_combined = pd.concat(merged_dfs, ignore_index=True)
        # Save combined file
        activity_combined.to_csv(f'data/processed/{activity}_merged.csv', index=False)
        print(f"{activity} merged: {len(merged_dfs)} recordings, total rows = {activity_combined.shape[0]}")
    else:
        print(f"No recordings found for {activity}")

print("All activities merged successfully!")
