# ⚽ Football Kick Analysis ML Model

AI-powered football kick quality assessment using pose estimation and XGBoost regression models.

## 📋 Overview

This repository contains the complete machine learning pipeline for analyzing football kick technique from video footage. The system extracts biomechanical features using MediaPipe pose estimation and predicts 5 quality scores (0-100) using trained XGBoost models.

**Key Features:**
- 🎥 Automated pose extraction from video (MediaPipe)
- 🔢 ~200 engineered biomechanical features
- 🤖 5 XGBoost regression models (stability, power, technique, balance, overall)
- 📊 Expected accuracy: R² > 0.75
- ⚡ Processing time: ~3 seconds per video

## 🏗️ Project Structure

```
opensport-ml/
├── dataset/                    # Training data
│   ├── videos/                 # Video files
│   │   ├── raw/               # Original videos
│   │   └── processed/         # Validated videos
│   ├── pose_data/             # Extracted pose JSON files
│   ├── labels.csv             # Manual labels (CEO annotated)
│   └── training_data.csv      # Final feature matrix
│
├── src/                        # Source code
│   ├── data_collection/
│   │   ├── validate_videos.py      # Video quality validation
│   │   └── label_interface.py     # Labeling tool
│   │
│   ├── preprocessing/
│   │   ├── extract_poses.py       # MediaPipe pose extraction
│   │   └── feature_engineering.py # Feature calculation
│   │
│   ├── training/
│   │   ├── train_xgboost.py       # Model training
│   │   ├── evaluate.py            # Model evaluation
│   │   └── hyperparameter_tune.py # Optional tuning
│   │
│   └── inference/
│       ├── predictor.py           # Single video prediction
│       └── fastapi_app.py         # API deployment
│
├── models/                     # Saved models
│   ├── xgboost_models.pkl     # 5 trained models
│   ├── scaler.pkl             # StandardScaler
│   └── feature_names.pkl      # Feature order
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_model_evaluation.ipynb
│
├── tests/                      # Unit tests
│   ├── test_pose_extraction.py
│   ├── test_features.py
│   └── test_models.py
│
├── config/                     # Configuration files
│   └── config.yaml
│
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker container
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- FFmpeg (for video processing)
- 4GB+ RAM
- (Optional) NVIDIA GPU for faster pose extraction

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/opensport-ml.git
cd opensport-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

**1. Prepare Training Data**

```bash
# Validate your videos
python src/data_collection/validate_videos.py --input dataset/videos/raw

# Label videos (interactive tool)
python src/data_collection/label_interface.py
```

**2. Extract Features**

```bash
# Extract poses from all videos
python src/preprocessing/extract_poses.py --input dataset/videos/raw --output dataset/pose_data

# Generate feature matrix
python src/preprocessing/feature_engineering.py --input dataset/pose_data --output dataset/training_data.csv
```

**3. Train Models**

```bash
# Train XGBoost models
python src/training/train_xgboost.py --data dataset/training_data.csv --output models/

# Evaluate models
python src/training/evaluate.py --models models/ --data dataset/training_data.csv
```

**4. Make Predictions**

```bash
# Analyze single video
python src/inference/predictor.py --video path/to/kick.mp4 --models models/

# Start FastAPI server
uvicorn src.inference.fastapi_app:app --host 0.0.0.0 --port 8000
```

## 📊 Training Data Format

Your CEO should prepare `labels.csv` with this structure:

```csv
video_id,stability,power,technique,balance,overall,notes,filming_date,player_level
kick_001.mp4,78,85,72,80,79,"Good form, slight lean",2025-01-15,intermediate
kick_002.mp4,92,88,95,90,91,"Excellent technique",2025-01-15,advanced
kick_003.mp4,45,50,48,55,49,"Beginner, needs work",2025-01-16,beginner
```

**Columns:**
- `video_id`: Filename of video (must match files in `dataset/videos/raw/`)
- `stability`: Score 0-100 (plant foot stability, balance)
- `power`: Score 0-100 (leg speed, knee extension)
- `technique`: Score 0-100 (form, body position)
- `balance`: Score 0-100 (core stability, weight distribution)
- `overall`: Score 0-100 (holistic quality assessment)
- `notes`: Text feedback (optional, for reference)
- `filming_date`: Date video was recorded
- `player_level`: beginner/intermediate/advanced/professional

## 🎯 Expected Performance

With 150-200 labeled videos:

| Metric | Expected R² | RMSE |
|--------|------------|------|
| Stability | 0.76 | 11.3 |
| Power | 0.79 | 10.1 |
| Technique | 0.74 | 12.1 |
| Balance | 0.72 | 12.8 |
| Overall | 0.78 | 10.5 |

## 🐳 Docker Deployment

```bash
# Build image
docker build -t opensport-ml .

# Run container
docker run -p 8000:8000 opensport-ml
```

## 📝 API Usage

```python
import requests

# Analyze video via API
response = requests.post(
    "http://localhost:8000/analyze",
    json={"video_url": "https://your-storage.com/kick.mp4"}
)

result = response.json()
print(result['scores'])  # {'stability': 78, 'power': 82, ...}
print(result['feedback'])  # Text recommendations
```

## 🔧 Configuration

Edit `config/config.yaml` to customize:

```yaml
data:
  video_min_duration: 4  # seconds
  video_max_duration: 10
  fps_subsample: 3  # Process every 3rd frame
  
pose:
  model_complexity: 2  # MediaPipe: 0=lite, 1=full, 2=heavy
  min_detection_confidence: 0.5
  
model:
  n_estimators: 200
  learning_rate: 0.1
  max_depth: 6
  test_size: 0.2
```

## 📈 Development Roadmap

- [x] Pose extraction pipeline
- [x] Feature engineering (200 features)
- [x] XGBoost training
- [x] FastAPI deployment
- [ ] Hyperparameter tuning
- [ ] Model versioning
- [ ] A/B testing framework
- [ ] Real-time video streaming
- [ ] Multi-angle support

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- MediaPipe by Google Research
- XGBoost library
- OpenCV team
- Our CEO for expert video labeling

## 📧 Contact

- **Project Lead**: [Your Name]
- **Email**: your.email@example.com
- **Competition**: [Competition Name]
- **Deadline**: 45 days from now

---

**Built with ❤️ for the future of football analytics**
