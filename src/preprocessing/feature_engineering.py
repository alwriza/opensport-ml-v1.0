"""
Feature Engineering Module - ROBUST VERSION
Handles missing landmarks and incomplete pose data gracefully.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Extract biomechanical features from pose sequences"""
    
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # MediaPipe landmark indices
        self.LANDMARKS = {
            'nose': 0,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
            'left_heel': 29,
            'right_heel': 30,
            'left_foot': 31,
            'right_foot': 32
        }
    
    def get_landmark(self, landmarks, idx, default=None):
        """Safely get landmark with fallback"""
        try:
            if idx in landmarks:
                return landmarks[idx]
            elif str(idx) in landmarks:
                return landmarks[str(idx)]
            else:
                return default
        except:
            return default
    
    def calculate_angle(self, point1, point2, point3):
        """Calculate angle at point2 (handles None values)"""
        if point1 is None or point2 is None or point3 is None:
            return np.nan
        
        try:
            vector1 = np.array([point1['x'] - point2['x'], point1['y'] - point2['y']])
            vector2 = np.array([point3['x'] - point2['x'], point3['y'] - point2['y']])
            
            cosine = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2) + 1e-6)
            angle = np.arccos(np.clip(cosine, -1.0, 1.0))
            
            return np.degrees(angle)
        except:
            return np.nan
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance (handles None values)"""
        if point1 is None or point2 is None:
            return np.nan
        
        try:
            return np.sqrt(
                (point1['x'] - point2['x'])**2 + 
                (point1['y'] - point2['y'])**2 + 
                (point1.get('z', 0) - point2.get('z', 0))**2
            )
        except:
            return np.nan
    
    def extract_geometric_features(self, frame):
        """Extract geometric features from a single frame"""
        landmarks = frame['landmarks']
        features = {}
        
        try:
            # Get key points (with safe fallbacks)
            left_shoulder = self.get_landmark(landmarks, self.LANDMARKS['left_shoulder'])
            right_shoulder = self.get_landmark(landmarks, self.LANDMARKS['right_shoulder'])
            left_hip = self.get_landmark(landmarks, self.LANDMARKS['left_hip'])
            right_hip = self.get_landmark(landmarks, self.LANDMARKS['right_hip'])
            left_knee = self.get_landmark(landmarks, self.LANDMARKS['left_knee'])
            right_knee = self.get_landmark(landmarks, self.LANDMARKS['right_knee'])
            left_ankle = self.get_landmark(landmarks, self.LANDMARKS['left_ankle'])
            right_ankle = self.get_landmark(landmarks, self.LANDMARKS['right_ankle'])
            left_foot = self.get_landmark(landmarks, self.LANDMARKS['left_foot'])
            right_foot = self.get_landmark(landmarks, self.LANDMARKS['right_foot'])
            nose = self.get_landmark(landmarks, self.LANDMARKS['nose'])
            
            # Skip frame if critical landmarks missing
            critical_landmarks = [left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]
            if any(x is None for x in critical_landmarks):
                return None
            
            # Hip center
            hip_center_x = (left_hip['x'] + right_hip['x']) / 2
            hip_center_y = (left_hip['y'] + right_hip['y']) / 2
            hip_center = {'x': hip_center_x, 'y': hip_center_y, 'z': 0}
            
            # Angles
            features['left_knee_angle'] = self.calculate_angle(left_hip, left_knee, left_ankle)
            features['right_knee_angle'] = self.calculate_angle(right_hip, right_knee, right_ankle)
            features['left_hip_angle'] = self.calculate_angle(left_shoulder, left_hip, left_knee)
            features['right_hip_angle'] = self.calculate_angle(right_shoulder, right_hip, right_knee)
            
            if left_foot:
                features['left_ankle_angle'] = self.calculate_angle(left_knee, left_ankle, left_foot)
            if right_foot:
                features['right_ankle_angle'] = self.calculate_angle(right_knee, right_ankle, right_foot)
            
            # Body lean
            if nose:
                features['body_lean_forward'] = nose['y'] - hip_center_y
                features['body_lean_side'] = nose['x'] - hip_center_x
            
            # Distances (normalized)
            body_height = self.calculate_distance(
                {'x': (left_shoulder['x'] + right_shoulder['x'])/2 if left_shoulder and right_shoulder else hip_center_x,
                 'y': (left_shoulder['y'] + right_shoulder['y'])/2 if left_shoulder and right_shoulder else hip_center_y,
                 'z': 0},
                hip_center
            ) if left_shoulder and right_shoulder else 1.0
            
            features['plant_foot_distance_left'] = self.calculate_distance(left_ankle, hip_center) / (body_height + 1e-6)
            features['plant_foot_distance_right'] = self.calculate_distance(right_ankle, hip_center) / (body_height + 1e-6)
            features['feet_distance'] = self.calculate_distance(left_ankle, right_ankle) / (body_height + 1e-6)
            
            if left_shoulder and right_shoulder:
                features['shoulder_width'] = self.calculate_distance(left_shoulder, right_shoulder)
            features['hip_width'] = self.calculate_distance(left_hip, right_hip)
            
            # Remove any NaN features
            features = {k: v for k, v in features.items() if not (isinstance(v, float) and np.isnan(v))}
            
            return features if features else None
            
        except Exception as e:
            logger.warning(f"Error extracting geometric features: {e}")
            return None
    
    def extract_features_from_video(self, pose_data):
        """Extract all features from pose sequence"""
        poses = pose_data['poses']
        
        if len(poses) < 5:
            logger.warning("Too few frames detected")
            return None
        
        # Extract geometric features for each frame
        geometric_sequence = []
        for frame in poses:
            geo_features = self.extract_geometric_features(frame)
            if geo_features:
                geometric_sequence.append(geo_features)
        
        if len(geometric_sequence) < 5:
            logger.warning(f"Too few valid frames: {len(geometric_sequence)}")
            return None
        
        # Convert to DataFrame
        geo_df = pd.DataFrame(geometric_sequence)
        
        # Fill any remaining NaNs with column mean
        geo_df = geo_df.fillna(geo_df.mean())
        
        features = {}
        
        # TEMPORAL STATISTICS (mean, std, min, max, range)
        for col in geo_df.columns:
            features[f'{col}_mean'] = geo_df[col].mean()
            features[f'{col}_std'] = geo_df[col].std()
            features[f'{col}_min'] = geo_df[col].min()
            features[f'{col}_max'] = geo_df[col].max()
            features[f'{col}_range'] = geo_df[col].max() - geo_df[col].min()
        
        # VELOCITY FEATURES
        for col in geo_df.columns:
            velocities = np.diff(geo_df[col].values)
            if len(velocities) > 0:
                features[f'{col}_velocity_mean'] = np.mean(np.abs(velocities))
                features[f'{col}_velocity_max'] = np.max(np.abs(velocities))
        
        # IMPACT FRAME FEATURES
        try:
            # Find impact frame (max foot velocity)
            left_foot_positions = []
            right_foot_positions = []
            
            for frame in poses:
                left_foot = self.get_landmark(frame['landmarks'], self.LANDMARKS['left_foot'])
                right_foot = self.get_landmark(frame['landmarks'], self.LANDMARKS['right_foot'])
                
                if left_foot:
                    left_foot_positions.append(left_foot['y'])
                if right_foot:
                    right_foot_positions.append(right_foot['y'])
            
            if left_foot_positions and right_foot_positions:
                left_foot_velocities = np.abs(np.diff(left_foot_positions))
                right_foot_velocities = np.abs(np.diff(right_foot_positions))
                
                if len(left_foot_velocities) > 0 and len(right_foot_velocities) > 0:
                    if np.max(left_foot_velocities) > np.max(right_foot_velocities):
                        impact_frame_idx = np.argmax(left_foot_velocities)
                        kicking_foot = 'left'
                    else:
                        impact_frame_idx = np.argmax(right_foot_velocities)
                        kicking_foot = 'right'
                    
                    # Extract features at impact
                    impact_frame_idx = min(impact_frame_idx, len(geometric_sequence) - 1)
                    impact_features = geometric_sequence[impact_frame_idx]
                    
                    for key, value in impact_features.items():
                        features[f'{key}_at_impact'] = value
                    
                    features['impact_frame_number'] = impact_frame_idx
                    features['kicking_foot'] = 1 if kicking_foot == 'right' else 0
        except Exception as e:
            logger.warning(f"Error extracting impact features: {e}")
        
        # ADDITIONAL BIOMECHANICAL FEATURES
        if 'left_knee_angle_mean' in features and 'right_knee_angle_mean' in features:
            features['knee_angle_symmetry'] = 1 - abs(
                features['left_knee_angle_mean'] - features['right_knee_angle_mean']
            ) / 180.0
        
        # Remove any remaining NaN or inf values
        features = {k: v for k, v in features.items() 
                   if not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))}
        
        logger.info(f"Extracted {len(features)} features")
        
        return features if features else None
    
    def create_training_dataset(self, pose_dir, labels_path, output_path):
        """Create complete training dataset"""
        pose_dir = Path(pose_dir)
        
        # Load labels
        labels_df = pd.read_csv(labels_path)
        logger.info(f"Loaded {len(labels_df)} labels")
        
        # Process each pose file
        all_features = []
        failed_videos = []
        
        for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Extracting features"):
            video_id = row['video_id']
            
            # Find corresponding pose file
            pose_file = pose_dir / f"{Path(video_id).stem}_poses.json"
            
            if not pose_file.exists():
                logger.warning(f"Pose file not found for {video_id}")
                failed_videos.append((video_id, "Pose file not found"))
                continue
            
            # Load pose data
            try:
                with open(pose_file, 'r') as f:
                    pose_data = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading {video_id}: {e}")
                failed_videos.append((video_id, f"Load error: {e}"))
                continue
            
            # Extract features
            features = self.extract_features_from_video(pose_data)
            
            if features:
                # Add video ID and labels
                features['video_id'] = video_id
                features['stability'] = row['stability']
                features['power'] = row['power']
                features['technique'] = row['technique']
                features['balance'] = row['balance']
                features['overall'] = row['overall']
                
                all_features.append(features)
            else:
                logger.warning(f"Failed to extract features from {video_id}")
                failed_videos.append((video_id, "Feature extraction failed"))
        
        if not all_features:
            logger.error("No features extracted from any video!")
            logger.error("Failed videos:")
            for vid, reason in failed_videos:
                logger.error(f"  - {vid}: {reason}")
            raise ValueError("No valid features extracted. Check pose data quality.")
        
        # Create DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Reorder columns
        target_cols = ['video_id', 'stability', 'power', 'technique', 'balance', 'overall']
        feature_cols = [col for col in features_df.columns if col not in target_cols]
        features_df = features_df[target_cols + feature_cols]
        
        # Save
        features_df.to_csv(output_path, index=False)
        logger.info(f"Saved training dataset to {output_path}")
        logger.info(f"  Total samples: {len(features_df)}")
        logger.info(f"  Total features: {len(feature_cols)}")
        logger.info(f"  Failed videos: {len(failed_videos)}")
        
        if failed_videos:
            logger.warning(f"\nFailed to process {len(failed_videos)} videos:")
            for vid, reason in failed_videos[:10]:  # Show first 10
                logger.warning(f"  - {vid}: {reason}")
        
        return features_df


def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract features from pose data')
    parser.add_argument('--input', required=True, help='Input pose data directory')
    parser.add_argument('--labels', required=True, help='Path to labels.csv')
    parser.add_argument('--output', required=True, help='Output training_data.csv path')
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    engineer = FeatureEngineer(config_path=args.config)
    engineer.create_training_dataset(args.input, args.labels, args.output)
    
    logger.info("Feature extraction complete!")


if __name__ == "__main__":
    main()