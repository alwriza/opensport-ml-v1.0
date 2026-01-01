# 🚀 Quick Start Guide - OpenSport ML

Complete guide to get your ML model running in 45 days.

## 📅 Timeline Overview

**Week 1-2**: Data collection (150-200 videos)  
**Week 3**: Feature extraction  
**Week 4**: Model training  
**Week 5**: Deployment  
**Week 6**: Integration with website

---

## ⚡ Setup (30 minutes)

### 1. Clone and Install

```bash
# Clone repository
git clone https://github.com/yourusername/opensport-ml.git
cd opensport-ml

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import mediapipe; import xgboost; print('✓ All dependencies installed!')"
```

### 2. Create Directory Structure

```bash
# Create all necessary directories
mkdir -p dataset/videos/raw
mkdir -p dataset/videos/processed
mkdir -p dataset/pose_data
mkdir -p models
mkdir -p config

# Copy config file
# (config.yaml should already be in config/)
```

---

## 📹 Week 1-2: Data Collection (Your CEO's Task)

### Step 1: Film 150-200 Kick Videos

**Requirements** (give this list to your CEO):
- Side view camera angle (90° to kick direction)
- Full body visible (head to feet)
- 1080p, 30+ FPS
- 4-6 seconds per video
- Stationary ball kicks

**Distribution needed**:
- 20 terrible kicks (0-40 points) - beginners
- 45 poor kicks (40-60 points) - learning players
- 45 good kicks (60-80 points) - competent amateurs
- 40 excellent kicks (80-100 points) - advanced players

### Step 2: Validate Videos

```bash
# Validate all videos before proceeding
python validate_videos.py \
    --input dataset/videos/raw \
    --output dataset/videos/processed

# This will:
# - Check resolution (min 640x480)
# - Check FPS (min 24)
# - Check duration (4-10 seconds)
# - Copy valid videos to processed/
```

### Step 3: Label Videos

Your CEO needs to create `dataset/labels.csv`:

```csv
video_id,stability,power,technique,balance,overall,notes,filming_date,player_level
kick_001.mp4,78,85,72,80,79,"Good form, slight lean",2025-01-15,intermediate
kick_002.mp4,92,88,95,90,91,"Excellent technique",2025-01-15,advanced
kick_003.mp4,45,50,48,55,49,"Beginner, needs work",2025-01-16,beginner
```

**⏰ Time estimate**: 10-14 days (filming + labeling)

---

## 🤖 Week 3: Feature Extraction

### Step 1: Extract Poses

```bash
# Extract poses from all videos
python extract_poses.py \
    --input dataset/videos/processed \
    --output dataset/pose_data

# Takes ~30 seconds per video
# 150 videos = ~1.5 hours
```

This will create JSON files with pose data:
```
dataset/pose_data/
├── kick_001_poses.json
├── kick_002_poses.json
└── ...
```

### Step 2: Engineer Features

```bash
# Generate training dataset with ~200 features
python feature_engineering.py \
    --input dataset/pose_data \
    --labels dataset/labels.csv \
    --output dataset/training_data.csv

# Takes ~5 seconds per video
# 150 videos = ~12 minutes
```

This creates `dataset/training_data.csv` with ~200 feature columns.

**⏰ Time estimate**: 2-3 days (including validation)

---

## 🎓 Week 4: Model Training

### Train XGBoost Models

```bash
# Train all 5 models
python train_xgboost.py \
    --data dataset/training_data.csv \
    --output models/

# Training takes 3-5 minutes
```

**What this does**:
1. Trains 5 XGBoost models (stability, power, technique, balance, overall)
2. Evaluates on test set (20% of data)
3. Saves models to `models/`
4. Creates evaluation plots

**Expected results** (with 150-200 videos):
```
Stability  - R²: 0.76, RMSE: 11.3
Power      - R²: 0.79, RMSE: 10.1
Technique  - R²: 0.74, RMSE: 12.1
Balance    - R²: 0.72, RMSE: 12.8
Overall    - R²: 0.78, RMSE: 10.5
```

**⏰ Time estimate**: 1-2 days (including evaluation)

---

## 🚀 Week 5: Deployment

### Option A: Local Testing

```bash
# Start FastAPI server locally
uvicorn fastapi_app:app --reload --host 0.0.0.0 --port 8000

# Test in another terminal
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://your-storage.com/test-kick.mp4"}'
```

### Option B: Docker Deployment

```bash
# Build Docker image
docker build -t opensport-ml .

# Run container
docker run -p 8000:8000 opensport-ml

# Test
curl http://localhost:8000/
```

### Option C: Deploy to Render.com

1. Push code to GitHub
2. Go to Render.com → New Web Service
3. Connect your repo
4. Configure:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn fastapi_app:app --host 0.0.0.0 --port $PORT`
   - Add environment variables if needed

**⏰ Time estimate**: 2-3 days

---

## 🔗 Week 6: Website Integration

### Update Your Supabase Edge Function

```typescript
// supabase/functions/analyze-video/index.ts

const AI_WORKER_URL = 'https://your-ml-service.onrender.com';  // Your deployed URL

export const handler = async (req: Request) => {
  const { videoUrl } = await req.json();
  
  // Call your ML API
  const response = await fetch(`${AI_WORKER_URL}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video_url: videoUrl })
  });
  
  const analysis = await response.json();
  
  // Save to database
  const { data, error } = await supabase
    .from('reports')
    .insert({
      video_id: analysis.video_id,
      scores_json: analysis.scores,
      tags_json: analysis.tags,
      feedback_text: analysis.feedback,
      model_version: analysis.model_version
    });
  
  return new Response(JSON.stringify(analysis), {
    headers: { 'Content-Type': 'application/json' }
  });
};
```

**⏰ Time estimate**: 2-3 days

---

## 🧪 Testing Checklist

Before competition:

```bash
# ✓ Test pose extraction
python extract_poses.py --input test_videos/

# ✓ Test feature engineering
python feature_engineering.py --input test_poses/ --labels test_labels.csv --output test_features.csv

# ✓ Test model inference
curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" -d '{"video_url": "test-url"}'

# ✓ Test end-to-end website flow
# 1. Upload video through website
# 2. Wait for analysis
# 3. View results
```

---

## 🐛 Troubleshooting

### "MediaPipe not detecting poses"
```bash
# Check video quality
python validate_videos.py --input your_video_dir/

# Common issues:
# - Video too dark
# - Person not fully visible
# - Resolution too low
```

### "Model accuracy too low (R² < 0.70)"
```python
# Possible causes:
# 1. Not enough training data (need 150+ videos)
# 2. Labels inconsistent (CEO needs to re-check)
# 3. Video quality issues (re-validate videos)

# Solutions:
# - Collect more data
# - Get second opinion on labels
# - Tune hyperparameters (see train_xgboost.py)
```

### "FastAPI server crashes"
```bash
# Check logs
docker logs <container-id>

# Common issues:
# - Models not found (check models/ directory exists)
# - Config missing (check config/config.yaml)
# - Out of memory (reduce model_complexity in config)
```

---

## 📊 Monitoring

### Track Model Performance

```python
# After deploying, track these metrics:
# 1. Processing time (<5 seconds ideal)
# 2. Success rate (>95% of videos analyzed)
# 3. User feedback (are scores accurate?)
```

---

## 🎯 Success Criteria

Your system is competition-ready when:

- [x] 150+ labeled videos collected
- [x] Model R² > 0.75 on test set
- [x] FastAPI server deployed and accessible
- [x] Website integration complete
- [x] End-to-end flow tested (upload → analyze → display)
- [x] Demo video recorded
- [x] Presentation slides ready

---

## 🆘 Getting Help

If stuck:

1. **Check logs**: `tail -f /var/log/app.log`
2. **Test components individually**: Pose extraction → Features → Model
3. **Validate data**: Ensure videos and labels are correct
4. **Review documentation**: Check README.md

---

## 🎓 Next Steps After Competition

1. **Collect more data**: Aim for 500+ videos
2. **Retrain monthly**: As CEO adds more labels
3. **Add features**: Multi-angle support, real-time analysis
4. **Optimize**: Reduce processing time to <2 seconds
5. **Upgrade**: Consider LSTM/Transformer models

---

**Good luck! You've got this! 🚀⚽**
