"""
FastAPI Inference Service
Production API for kick analysis.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import joblib
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import requests
from pathlib import Path
import yaml
import logging
from typing import Optional, Dict, List

# Import our modules
import sys
sys.path.append('.')
from feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="OpenSport Kick Analysis API",
    description="AI-powered football kick quality assessment",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class AnalysisRequest(BaseModel):
    video_url: HttpUrl
    

class AnalysisResponse(BaseModel):
    video_id: str
    scores: Dict[str, float]
    tags: List[str]
    feedback: str
    processing_time_ms: int
    model_version: str
    features_json: Optional[Dict] = None


# Global variables for models
MODELS = None
SCALER = None
FEATURE_NAMES = None
FEATURE_ENGINEER = None
POSE_EXTRACTOR = None
CONFIG = None


@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    global MODELS, SCALER, FEATURE_NAMES, FEATURE_ENGINEER, POSE_EXTRACTOR, CONFIG
    
    logger.info("Loading models...")
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        CONFIG = yaml.safe_load(f)
    
    # Load ML models
    MODELS = joblib.load('models/xgboost_models.pkl')
    SCALER = joblib.load('models/scaler.pkl')
    FEATURE_NAMES = joblib.load('models/feature_names.pkl')
    
    # Initialize feature engineer
    FEATURE_ENGINEER = FeatureEngineer()
    
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    POSE_EXTRACTOR = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=CONFIG['pose_extraction']['model_complexity'],
        min_detection_confidence=CONFIG['pose_extraction']['min_detection_confidence'],
        min_tracking_confidence=CONFIG['pose_extraction']['min_tracking_confidence']
    )
    
    logger.info("Models loaded successfully!")


def download_video(video_url: str, output_path: str) -> bool:
    """Download video from URL"""
    try:
        response = requests.get(video_url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        logger.error(f"Error downloading video: {e}")
        return False


def extract_poses_from_video(video_path: str) -> Optional[Dict]:
    """Extract poses from video"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_subsample = CONFIG['pose_extraction']['fps_subsample']
    
    poses = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % fps_subsample == 0:
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process
            results = POSE_EXTRACTOR.process(frame_rgb)
            
            if results.pose_landmarks:
                landmarks = {}
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    landmarks[idx] = {
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    }
                
                poses.append({
                    'frame_number': frame_count,
                    'timestamp': frame_count / fps,
                    'landmarks': landmarks
                })
        
        frame_count += 1
    
    cap.release()
    
    if len(poses) < 5:
        return None
    
    return {
        'video_id': Path(video_path).name,
        'fps': fps,
        'total_frames': len(poses),
        'poses': poses
    }


def generate_feedback(scores: Dict[str, float], tags: List[str]) -> str:
    """Generate natural language feedback"""
    overall = scores['overall']
    
    # Opening
    if overall >= 80:
        feedback = "Excellent kick technique! "
    elif overall >= 60:
        feedback = "Good kick with room for improvement. "
    else:
        feedback = "Work on fundamentals to improve your kick. "
    
    # Strengths
    strengths = []
    for metric, score in scores.items():
        if metric != 'overall' and score > 75:
            strengths.append(f"{metric}")
    
    if strengths:
        feedback += f"Your {' and '.join(strengths)} {'is' if len(strengths) == 1 else 'are'} strong. "
    
    # Weaknesses with recommendations
    recommendations = []
    
    if scores['stability'] < 60:
        recommendations.append(
            f"Work on stability ({scores['stability']:.0f}/100). "
            "Plant your foot 30-40cm beside the ball for better balance."
        )
    
    if scores['power'] < 60:
        recommendations.append(
            f"Increase power ({scores['power']:.0f}/100). "
            "Focus on full knee extension and follow-through."
        )
    
    if scores['technique'] < 60:
        recommendations.append(
            f"Improve technique ({scores['technique']:.0f}/100). "
            "Strike with laces, keep ankle locked, and lean slightly over the ball."
        )
    
    if scores['balance'] < 60:
        recommendations.append(
            f"Enhance balance ({scores['balance']:.0f}/100). "
            "Engage your core and keep your body stable throughout the motion."
        )
    
    if recommendations:
        feedback += " ".join(recommendations)
    else:
        feedback += "Keep practicing to maintain consistency!"
    
    return feedback


def generate_tags(scores: Dict[str, float]) -> List[str]:
    """Generate tags based on scores"""
    tags = []
    
    for metric, score in scores.items():
        if metric == 'overall':
            continue
        
        if score < 50:
            tags.append(f"weak_{metric}")
        elif score > 80:
            tags.append(f"excellent_{metric}")
    
    return tags


@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "healthy",
        "service": "OpenSport Kick Analysis API",
        "version": "1.0.0"
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_video(request: AnalysisRequest):
    """
    Analyze football kick video
    
    Args:
        request: AnalysisRequest with video_url
        
    Returns:
        AnalysisResponse with scores and feedback
    """
    import time
    start_time = time.time()
    
    try:
        # Create temp file for video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            video_path = tmp.name
        
        # Download video
        logger.info(f"Downloading video from {request.video_url}")
        if not download_video(str(request.video_url), video_path):
            raise HTTPException(status_code=400, detail="Failed to download video")
        
        # Extract poses
        logger.info("Extracting poses...")
        pose_data = extract_poses_from_video(video_path)
        
        if not pose_data:
            raise HTTPException(status_code=400, detail="Failed to extract poses from video")
        
        # Extract features
        logger.info("Extracting features...")
        features = FEATURE_ENGINEER.extract_features_from_video(pose_data)
        
        if not features:
            raise HTTPException(status_code=400, detail="Failed to extract features")
        
        # Prepare feature vector
        feature_vector = []
        for feature_name in FEATURE_NAMES:
            feature_vector.append(features.get(feature_name, 0))
        
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        # Scale features
        feature_vector_scaled = SCALER.transform(feature_vector)
        
        # Predict with all 5 models
        scores = {}
        for target, model in MODELS.items():
            pred = model.predict(feature_vector_scaled)[0]
            pred_clipped = np.clip(pred, 0, 100)
            scores[target] = round(float(pred_clipped), 1)
        
        # Generate tags and feedback
        tags = generate_tags(scores)
        feedback = generate_feedback(scores, tags)
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Clean up
        Path(video_path).unlink(missing_ok=True)
        
        logger.info(f"Analysis complete in {processing_time_ms}ms")
        
        return AnalysisResponse(
            video_id=pose_data['video_id'],
            scores=scores,
            tags=tags,
            feedback=feedback,
            processing_time_ms=processing_time_ms,
            model_version="xgboost_v1.0",
            features_json=features if len(features) < 100 else None  # Include if not too large
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-upload")
async def analyze_upload(file: UploadFile = File(...)):
    """
    Analyze uploaded video file
    
    Args:
        file: Uploaded video file
        
    Returns:
        AnalysisResponse with scores and feedback
    """
    import time
    start_time = time.time()
    
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            video_path = tmp.name
            content = await file.read()
            tmp.write(content)
        
        # Extract poses
        logger.info("Extracting poses from uploaded file...")
        pose_data = extract_poses_from_video(video_path)
        
        if not pose_data:
            raise HTTPException(status_code=400, detail="Failed to extract poses")
        
        # Extract features
        logger.info("Extracting features...")
        features = FEATURE_ENGINEER.extract_features_from_video(pose_data)
        
        if not features:
            raise HTTPException(status_code=400, detail="Failed to extract features")
        
        # Prepare and scale features
        feature_vector = np.array([features.get(fn, 0) for fn in FEATURE_NAMES]).reshape(1, -1)
        feature_vector_scaled = SCALER.transform(feature_vector)
        
        # Predict
        scores = {}
        for target, model in MODELS.items():
            pred = np.clip(model.predict(feature_vector_scaled)[0], 0, 100)
            scores[target] = round(float(pred), 1)
        
        # Generate response
        tags = generate_tags(scores)
        feedback = generate_feedback(scores, tags)
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Clean up
        Path(video_path).unlink(missing_ok=True)
        
        return AnalysisResponse(
            video_id=file.filename,
            scores=scores,
            tags=tags,
            feedback=feedback,
            processing_time_ms=processing_time_ms,
            model_version="xgboost_v1.0"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
