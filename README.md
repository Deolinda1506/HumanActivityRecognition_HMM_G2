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
│   ├── extract_features_from_merged.py  # Extracts statistical features from merged sensor data
│   └── merge_activity_data.py           # Merges raw activity data into combined datasets
│
├── 📂 data/
│   ├── Deolinda/
│   │   ├── Standing/                   # Contains 10 records (standing_1 to standing_10)
│   │   │   ├── standing_1/
│   │   │   │   ├── Accelerometer.csv
│   │   │   │   └── Gyroscope.csv
│   │   │   └── ... standing_10/
│   │   ├── Still/                      # Contains 10 records (still_1 to still_10)
│   │   │   ├── still_1/
│   │   │   │   ├── Accelerometer.csv
│   │   │   │   └── Gyroscope.csv
│   │   │   └── ... still_10/
│   │   ├── Walking/                    # Contains 10 records (walking_1 to walking_10)
│   │   │   ├── walking_1/
│   │   │   │   ├── Accelerometer.csv
│   │   │   │   └── Gyroscope.csv
│   │   │   └── ... walking_10/
│   │   └── Jumping/                    # Contains 10 records (jumping_1 to jumping_10)
│   │       ├── jumping_1/
│   │       │   ├── Accelerometer.csv
│   │       │   └── Gyroscope.csv
│   │       └── ... jumping_10/
│   │
│   └── Diana/
│       ├── Standing/                   # Contains 10 records (standing1–standing10)
│       │   ├── standing1/
│       │   │   ├── Accelerometer.csv
│       │   │   └── Gyroscope.csv
│       │   └── ... standing10/
│       ├── Still/                      # Contains 10 records (still1–still10)
│       │   ├── still1/
│       │   │   ├── Accelerometer.csv
│       │   │   └── Gyroscope.csv
│       │   └── ... still10/
│       ├── Walking/                    # Contains 10 records (walking1–walking10)
│       │   ├── walking1/
│       │   │   ├── Accelerometer.csv
│       │   │   └── Gyroscope.csv
│       │   └── ... walking10/
│       └── Jumping/                    # Contains 10 records (jumping1–jumping10)
│           ├── jumping1/
│           │   ├── Accelerometer.csv
│           │   └── Gyroscope.csv
│           └── ... jumping10/
│
├── 📊 activity_plots/
│   └── all_activities_sensor_plots.png # Visualization of all activity sensor data
│
├── 📈 features/
│   └── features.csv                    # Extracted features ready for HMM training
│
├── 📂 merged/
│   ├── jumping_merged.csv
│   ├── standing_merged.csv
│   ├── still_merged.csv
│   └── walking_merged.csv              # Merged datasets per activity
│
├── 🧾 results/
│   ├── classification_report.txt       # Model performance summary
│   ├── confusion_matrix_seaborn.png    # Confusion matrix visualization
│   ├── decoded_vs_true_labels.png      # Comparison between predicted and true labels
│   ├── hmm_emission_probabilities.png  # Emission probability heatmap
│   ├── hmm_training_convergence.png    # Log-likelihood convergence plot
│   ├── hmm_transition_matrix.png       # Transition matrix visualization
│   ├── metrics_table.csv               # Model performance metrics
│   └── overall_metrics.png             # Overall accuracy visualization
│
└── 📄 README.md                        # Project documentation
```

### 🚀 Features
### Data Collection

Sensors Used: Accelerometer (x, y, z) + Gyroscope (x, y, z)
Activities: Standing, Walking, Jumping, Still
Duration: 5-10 seconds per activity session
Total Samples: ~50 recordings across all activities
Sampling Rates: Harmonized across different devices

### Feature Engineering
- Time-Domain Features (per axis): Mean, Standard Deviation, Variance Mean Absolute Deviation (MAD), Signal Magnitude Area (SMA), Resultant acceleration magnitude
- Frequency-Domain Features (per axis): Dominant frequency (FFT) Spectral energy (Welch's method)
- Top-3 FFT components (magnitude + frequency)
- Total Features: 81 features per sliding window
- Window Size: 128 samples
- Overlap: 50%
- Normalization: Z-score normalization
- Time-domain features: Signal processing fundamentals
- FFT: Fast Fourier Transform for frequency analysis
- Spectral analysis: Power spectral density methods

### HMM Implementation
**Model Components:**

Hidden States (Z): {Standing, Walking, Jumping, Still}
Observations (X): 81-dimensional feature vectors
Transition Matrix (A): 4×4 probability matrix
Emission Probabilities (B): Gaussian distributions per state
Initial Probabilities (π): Starting state distribution

Algorithms:

Viterbi Algorithm: Decodes the most likely activity sequence
Baum-Welch Algorithm: Trains model parameters with convergence check (log-likelihood < ε)


### 🛠️ Installation
**Prerequisites**

Python 3.8 or higher
pip package manager

**Setup**

Clone the repository

bashgit clone https://github.com/yourusername/activity-recognition-hmm.git
cd activity-recognition-hmm

Create virtual environment

bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies

bashpip install -r requirements.txt
Required Packages
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
hmmlearn>=0.2.7
jupyter>=1.0.0

### 📊 Usage
**1. Data Collection**
Use a smartphone motion logging app:

Recommended Apps: Sensor Logger (iOS/Android), Physics Toolbox Accelerometer
Record 5-10 second sessions for each activity
Save as CSV files with timestamps

**2. Merge Sensor Data**
bashpython scripts/merge_activity_data.py
Output: Merged CSV files in data/merged/ with columns:
time, ax, ay, az, gx, gy, gz, activity_type
**3. Extract Features**
bashpython scripts/extract_features_from_merged.py
Output: data/features/features.csv with 81 features per window
**4. Train HMM Model**
bashpython scripts/train_hmm.py
Parameters:

Number of states: 4 (auto-detected from activities)
Convergence threshold: 1e-4
Maximum iterations: 100

**5. Evaluate Model**
bashpython scripts/evaluate_hmm.py
Outputs:

Confusion matrix
Sensitivity and specificity per activity
Overall accuracy
Visualizations saved to results/

**6. Run Jupyter Notebook**
bashjupyter notebook notebooks/hmm_activity_recognition.ipynb

### 📈 Results
### Model Performance
ActivitySamplesSensitivitySpecificityOverall AccuracyStandingXXXX.X%XX.X%XX.X%WalkingXXXX.X%XX.X%XX.X%JumpingXXXX.X%XX.X%XX.X%StillXXXX.X%XX.X%XX.X%
Visualizations
Transition Probability Matrix
Shows the likelihood of transitioning between different activities.
Emission Probabilities
Displays the distribution of feature observations for each activity state.
Confusion Matrix
Illustrates classification performance across all activities.

### 🧪 Data Collection Details
**Participants**

Number of Participants: 2
Phones Used:

Participant 1: [Phone Model] @ [XX Hz]
Participant 2: [Phone Model] @ [XX Hz]



**Windowing Strategy**

Window Size: 128 samples (2.56 seconds @ 50Hz)
Overlap: 50% (64 samples)
Rationale: Balances temporal resolution with computational efficiency while capturing complete activity cycles

Sampling Rate Harmonization
All recordings were resampled to 50 Hz to ensure consistency across different devices.

### 🔬 Methodology
### 1. Preprocessing

Timestamp normalization (handles milliseconds/nanoseconds)
Missing value interpolation
Outlier detection and removal

### 2. Feature Extraction

Sliding window approach with 50% overlap
Separate time-domain and frequency-domain computation
Z-score normalization across all windows

### 3. HMM Training

Train-test split: 80%-20%
Baum-Welch algorithm with convergence monitoring
Model validation on unseen test data

### 4. Evaluation

k-fold cross-validation (optional)
Per-class metrics: sensitivity, specificity, F1-score
Confusion matrix analysis


🎓 Key Findings
Activity Discrimination

Easiest to distinguish: Jumping (high amplitude, distinct frequency)
Most challenging: Standing vs. Still (subtle differences)

Transition Patterns

Walking → Standing: Most common transition
Still → Jumping: Rare but accurately detected
Model captures realistic activity sequences

## Additional Resources

### Understanding HMMs

- Forward-Backward algorithm: https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
- Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
- Baum-Welch algorithm: https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm

### Feature Importance

- Dominant frequency and spectral energy most discriminative
- Resultant acceleration magnitude critical for jumping detection
- Gyroscope features essential for rotation detection

🚧 Future Improvements

- Data Augmentation: Collect more samples for underrepresented activities
- Additional Features: Add time-series features (autocorrelation, entropy)
- Hybrid Models: Combine HMM with LSTM for improved accuracy
- Real-time Inference: Optimize for mobile deployment
- More Activities: Extend to running, climbing stairs, cycling

### License

Educational project for academic purposes.
