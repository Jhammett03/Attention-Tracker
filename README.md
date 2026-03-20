# Focus Detection via Computer Vision

A machine learning system that classifies human attention states (**focused** vs. **distracted**) using facial landmark analysis and temporal modeling.

This senior project explores how visual attention signals—such as eye aspect ratio, gaze position, and head movement patterns—can be leveraged to detect cognitive focus through video analysis.

---

## Overview

This project processes video recordings of individuals to extract facial landmarks, then trains multiple classification models to predict attention state. The system uses MediaPipe for facial feature extraction and implements several machine learning approaches ranging from classical logistic regression to deep learning architectures.

### Why This Matters

Attention detection has practical applications in:
- Driver drowsiness monitoring systems
- Student engagement analysis in educational settings
- Productivity tracking tools
- Human-computer interaction research
- Accessibility technologies

---

## Project Pipeline

```
Video Input
    ↓
Frame Extraction (3 fps, 10 seconds)
    ↓
Facial Landmark Detection (MediaPipe)
    ↓
Feature Engineering (EAR, eye centers, variance metrics)
    ↓
Dataset Generation (temporal sequences + aggregated features)
    ↓
Model Training (Logistic Regression / CNN / Transformer)
    ↓
Classification (focused / distracted)
```

---

## Features

### Core Capabilities

- **Video Processing**: Extracts frames at 3 fps for 10-second duration windows
- **Facial Landmark Extraction**: Uses MediaPipe Face Landmarker to detect 478 facial landmarks
- **Feature Engineering**: Computes Eye Aspect Ratio (EAR) and spatial variance metrics
- **Dual Dataset Generation**:
  - **Temporal sequences** (NPZ): 30 timesteps × 6 features per video
  - **Aggregated features** (CSV): Mean/std/variance statistics per video
- **Multiple Model Architectures**: Baseline, temporal, CNN, and transformer-based approaches
- **Cross-Validation**: 5-fold stratified cross-validation for robust evaluation

### Extracted Features

**Per-Frame Features (6 dimensions)**:
- `left_ear`: Eye Aspect Ratio for left eye (measures eye openness)
- `right_ear`: Eye Aspect Ratio for right eye
- `left_center_x`, `left_center_y`: Spatial coordinates of left eye center
- `right_center_x`, `right_center_y`: Spatial coordinates of right eye center

**Aggregated Features (8 dimensions)**:
- `mean_left_ear`, `mean_right_ear`: Average eye openness over video
- `std_left_ear`, `std_right_ear`: Variability in eye openness
- `left_x_variance`, `left_y_variance`: Spatial variance of left eye position
- `right_x_variance`, `right_y_variance`: Spatial variance of right eye position

---

## Project Structure

```
.
├── frame_extraction.py       # Video processing and feature extraction pipeline
├── baseline_model.py         # Logistic regression on aggregated features (CSV)
├── time_model.py            # Logistic regression on flattened temporal sequences
├── cnn_time_model.py        # 1D CNN for temporal pattern recognition
├── transformer_model.py     # Transformer with positional encoding
├── requirements.txt         # Python dependencies
├── dataset.csv             # Aggregated features per video (generated)
├── sequence_dataset.npz    # Temporal sequences (N, 30, 6) (generated)
├── face_landmarker.task    # MediaPipe model weights (required)
└── videos/                 # Video dataset (download separately)
    ├── focused/
    └── distracted/
```

**Note**: Large files (videos, model weights, generated datasets) are excluded from version control via `.gitignore`.

---

## Installation

### Prerequisites

- Python 3.8+
- pip
- virtualenv (recommended)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd <repo-name>
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the MediaPipe model**:
   Download `face_landmarker.task` from [MediaPipe Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) and place it in the project root.

---

## Dataset

The dataset consists of labeled video recordings of individuals in focused and distracted states.

### Download Dataset

The full dataset is hosted on Kaggle:

**[Focus Detection Dataset](https://www.kaggle.com/datasets/jaredhammett/focused-vs-distracted)**

### Setup Instructions

1. Download the dataset from Kaggle
2. Extract the `videos.zip` file
3. Place the `videos/` folder in the project root:
   ```
   videos/
   ├── focused/       # Videos of focused individuals
   └── distracted/    # Videos of distracted individuals
   ```

**Important**: The dataset is NOT included in this repository due to size constraints. Videos, frames, and generated datasets are excluded via `.gitignore`.

---

## Usage

### 1. Extract Features from Videos

Process all videos in the `videos/` directory to generate datasets:

```bash
python frame_extraction.py
```

**Outputs**:
- `sequence_dataset.npz`: Temporal sequences (shape: N × 30 × 6)
- `dataset.csv`: Aggregated features with labels

**What it does**:
- Extracts 30 frames per video (3 fps over 10 seconds)
- Detects facial landmarks using MediaPipe
- Computes Eye Aspect Ratio (EAR) for both eyes
- Calculates eye center positions
- Stores both temporal sequences and aggregated statistics

### 2. Train Models

#### Baseline Model (Logistic Regression on Aggregated Features)

```bash
python baseline_model.py
```

Uses aggregated features from `dataset.csv` (mean, std, variance of EAR and eye positions). Provides a simple baseline for comparison using logistic regression.

#### Temporal Model (Flattened Sequence Features)

```bash
python time_model.py
```

Flattens the temporal dimension (30 × 6 = 180 features) and applies logistic regression. Captures frame-level information without temporal modeling.

#### CNN Model (Temporal Convolutions)

```bash
python cnn_time_model.py
```

Uses 1D convolutional layers to learn temporal patterns across the 30-frame sequences. Detects local temporal dependencies.

#### Transformer Model (Self-Attention)

```bash
python transformer_model.py
```

Applies transformer architecture with positional encoding to model global temporal dependencies. Captures long-range patterns across the video.

### 3. Evaluation

All models use **5-fold stratified cross-validation** and report:
- Mean accuracy ± standard deviation
- Macro F1-score ± standard deviation
- Confusion matrix (transformer only)
- Per-class precision/recall (transformer only)

---

## Models

### 1. Baseline Model (`baseline_model.py`)

**Architecture**: Logistic Regression
**Input**: Aggregated features (8 dimensions)
**Approach**: Statistical summary of entire video

Simple baseline that reduces each video to 8 aggregate statistics. Fast to train but loses temporal information.

### 2. Temporal Flattened Model (`time_model.py`)

**Architecture**: Logistic Regression
**Input**: Flattened temporal sequences (180 dimensions)
**Approach**: Frame-level features without temporal modeling

Preserves all frame-level information by flattening the time dimension, but doesn't explicitly model temporal dependencies.

### 3. Temporal CNN (`cnn_time_model.py`)

**Architecture**: 1D Convolutional Neural Network
**Layers**:
- Conv1D (6 → 16 channels, kernel size 3)
- Conv1D (16 → 32 channels, kernel size 3)
- AdaptiveAvgPool1d (global pooling)
- Fully connected layer (32 → 2 classes)

**Input**: Temporal sequences (30 × 6)
**Approach**: Convolutional filters slide over time dimension to detect local patterns

Uses early stopping (patience=15) to prevent overfitting. Learns hierarchical temporal features through stacked convolutions.

### 4. Transformer Classifier (`transformer_model.py`)

**Architecture**: Transformer Encoder with Positional Encoding
**Components**:
- Input projection (6 → 64 dimensions)
- Learnable positional encodings (30 timesteps)
- 2-layer Transformer Encoder (4 attention heads)
- Global mean pooling
- Classification head (64 → 2 classes)

**Input**: Temporal sequences (30 × 6)
**Approach**: Self-attention mechanism captures global temporal dependencies

Most sophisticated model. Uses attention to weigh importance of different timesteps dynamically. Includes dropout (0.1) for regularization.

---

## Technical Details

### Eye Aspect Ratio (EAR)

The Eye Aspect Ratio is a geometric metric for measuring eye openness:

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```

Where p1-p6 are the 6 facial landmarks around each eye. Lower EAR values indicate closed or partially closed eyes, while higher values indicate wide-open eyes.

### Frame Extraction Parameters

- **Frame rate**: 3 fps
- **Duration**: 10 seconds
- **Total frames**: 30 per video
- **Padding**: Sequences shorter than 30 frames are zero-padded

### Model Training Configuration

- **Cross-validation**: 5-fold stratified
- **Normalization**: StandardScaler (per-feature z-score)
- **Early stopping**: Patience = 15 epochs
- **Optimizer**: Adam (learning rate = 1e-3)
- **Loss function**: CrossEntropyLoss
- **Device**: Auto-detects CUDA if available (CPU fallback)

---

## Technologies Used

- **Python 3.11**: Core programming language
- **OpenCV**: Video processing and frame extraction
- **MediaPipe**: Facial landmark detection (478-point face mesh)
- **NumPy**: Numerical operations and array manipulation
- **Pandas**: CSV data handling
- **scikit-learn**: Traditional ML models, preprocessing, evaluation metrics
- **PyTorch**: Deep learning framework for CNN and transformer models

---

## Future Work

### Potential Improvements

- **Real-time Inference**: Optimize for live video stream processing with sliding window approach
- **Enhanced Gaze Tracking**: Incorporate iris landmarks and head pose estimation for more accurate gaze direction
- **Larger Dataset**: Expand beyond 96 samples to improve model generalization
- **Data Augmentation**: Apply video transformations (rotation, brightness, noise) to increase training diversity

---

## Performance Notes

Model performance varies based on architecture:

- **Baseline models** (~75-80% accuracy): Fast training, good starting point
- **Deep learning models** (~80-95% accuracy): Require more data and hyperparameter tuning but can capture complex temporal patterns

Results depend heavily on dataset quality, recording conditions, and class balance. The current dataset contains 96 videos (50 focused, 46 distracted).

---

## Acknowledgments

- **MediaPipe** by Google for facial landmark detection
- **PyTorch** team for the deep learning framework
- **scikit-learn** contributors for machine learning utilities
