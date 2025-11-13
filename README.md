# Hidden Markov Model for Human Activity Recognition

## Project Overview

This project implements a complete Hidden Markov Model (HMM) system for recognizing human activities (Standing, Still, Walking) using smartphone accelerometer and gyroscope data.

## Project Structure

```
HumanActivityRecognition_HMM_G2/
│
├── 📓 Notebook/
│   └── HMM.ipynb                      # Main Jupyter notebook for training and evaluating the HMM model
│
├── 📜 hmm_metrics.csv                 # Summary of Hidden Markov Model metrics
│
├── 🧠 Scripts/
│   ├── extract_features_from_merged.py  # Script to extract statistical features from merged sensor data
│   └── merge_activity_data.py           # Script to merge individual activity CSVs into one dataset
│
├── 📂 data/
│   ├── Deolinda/
│   │   ├── Standing/                   # Contains 10 records of standing activity data
│   │   ├── Still/                      # Contains 10 records of still activity data
│   │   ├── Walking/                    # Contains 10 records of walking activity data
│   │   └── Jumping/                    # Contains 10 records of jumping activity data
│   │
│   └── Diana/
│       ├── Standing/                   # Contains 10 records of standing activity data
│       ├── Still/                      # Contains 10 records of still activity data
│       ├── Walking/                    # Contains 10 records of walking activity data
│       └── Jumping/                    # Contains 10 records of jumping activity data
│
├── 📊 activity_plots/
│   └── all_activities_sensor_plots.png # Visualization of sensor data for all activities
│
├── 📈 features/
│   └── features.csv                    # Extracted features ready for HMM model training
│
├── 📂 merged/
│   ├── jumping_merged.csv
│   ├── standing_merged.csv
│   ├── still_merged.csv
│   └── walking_merged.csv              # Merged sensor data per activity type
│
├── 🧾 results/
│   ├── classification_report.txt       # Model performance summary
│   ├── confusion_matrix_seaborn.png    # Confusion matrix visualization
│   ├── decoded_vs_true_labels.png      # Comparison between predicted and true activity labels
│   ├── hmm_emission_probabilities.png  # Emission probability heatmap
│   ├── hmm_training_convergence.png    # Log-likelihood convergence plot
│   ├── hmm_transition_matrix.png       # Transition matrix visualization
│   ├── metrics_table.csv               # Model performance metrics
│   └── overall_metrics.png             # Overall model accuracy plot
│
└── 📄 README.md                        # Project documentation (to be created)
```
