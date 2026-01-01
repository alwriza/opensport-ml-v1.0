"""
Feature Engineering Module
Extracts ~200 biomechanical features from pose data.
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
            'left_eye_inner': 1,
            'left_eye': 2,
            'left_eye_outer': 3,
            'right_eye_inner': 4,
            'right_eye': 5,
            'right_eye_outer': 6,
            'left_ear': 7,
            'right_ear': 8,
            'mouth_left': 9,
            'mouth_right': 10,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_pinky': 17,
            'right_pinky': 18,
            'left_index': 19,
            'right_index': 20,
            'left_thumb': 21,
            'right_thumb': 22,
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
    
    def calculate_angle(self, point1, point2, point3):
        """
        Calculate angle at point2
        
        Args:
            point1, point2, point3: dicts with 'x', 'y' keys
            
        Returns:
            float: Angle in degrees
        """
        vector1 = np.array([point1['x'] - point2['x'], point1['y'] - point2['y']])
        vector2 = np.array([point3['x'] - point2['x'], point3['y'] - point2['y']])
        
        cosine = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2) + 1e-6)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt(
            (point1['x'] - point2['x'])**2 + 
            (point1['y'] - point2['y'])**2 + 
            (point1['z'] - point2['z'])**2
        )
    
    def extract_geometric_features(self, frame):
        """
        Extract geometric features from a single frame
        
        Returns:
            dict: Geometric features
        """
        landmarks = frame['landmarks']
        features = {}
        
        try:
            # Get key points
            left_shoulder = landmarks[self.LANDMARKS['left_shoulder']]
            right_shoulder = landmarks[self.LANDMARKS['right_shoulder']]
            left_hip = landmarks[self.LANDMARKS['left_hip']]
            right_hip = landmarks[self.LANDMARKS['right_hip']]
            left_knee = landmarks[self.LANDMARKS['left_knee']]
            right_knee = landmarks[self.LANDMARKS['right_knee']]
            left_ankle = landmarks[self.LANDMARKS['left_ankle']]
            right_ankle = landmarks[self.LANDMARKS['right_ankle']]
            left_foot = landmarks[self.LANDMARKS['left_foot']]
            right_foot = landmarks[self.LANDMARKS['right_foot']]
            nose = landmarks[self.LANDMARKS['nose']]
            
            # Hip center
            hip_center_x = (left_hip['x'] + right_hip['x']) / 2
            hip_center_y = (left_hip['y'] + right_hip['y']) / 2
            hip_center = {'x': hip_center_x, 'y': hip_center_y, 'z': 0}
            
            # Angles
            features['left_knee_angle'] = self.calculate_angle(left_hip, left_knee, left_ankle)
            features['right_knee_angle'] = self.calculate_angle(right_hip, right_knee, right_ankle)
            features['left_hip_angle'] = self.calculate_angle(left_shoulder, left_hip, left_knee)
            features['right_hip_angle'] = self.calculate_angle(right_shoulder, right_hip, right_knee)
            features['left_ankle_angle'] = self.calculate_angle(left_knee, left_ankle, left_foot)
            features['right_ankle_angle'] = self.calculate_angle(right_knee, right_ankle, right_foot)
            
            # Body lean
            features['body_lean_forward'] = nose['y'] - hip_center_y
            features['body_lean_side'] = nose['x'] - hip_center_x
            
            # Torso angle (shoulders to hips)
            shoulder_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
            torso_vector = np.array([shoulder_center_x - hip_center_x, 
                                    (left_shoulder['y'] + right_shoulder['y'])/2 - hip_center_y])
            features['torso_angle'] = np.degrees(np.arctan2(torso_vector[1], torso_vector[0]))
            
            # Distances (normalized by body size)
            body_height = self.calculate_distance(
                {'x': shoulder_center_x, 'y': (left_shoulder['y'] + right_shoulder['y'])/2, 'z': 0},
                hip_center
            )
            
            features['plant_foot_distance_left'] = self.calculate_distance(left_ankle, hip_center) / (body_height + 1e-6)
            features['plant_foot_distance_right'] = self.calculate_distance(right_ankle, hip_center) / (body_height + 1e-6)
            features['feet_distance'] = self.calculate_distance(left_ankle, right_ankle) / (body_height + 1e-6)
            features['shoulder_width'] = self.calculate_distance(left_shoulder, right_shoulder)
            features['hip_width'] = self.calculate_distance(left_hip, right_hip)
            
        except (KeyError, ZeroDivisionError) as e:
            logger.warning(f"Error extracting geometric features: {e}")
            return None
        
        return features
    
    def smooth_sequence(self, values, window=3):
        """Apply moving average smoothing"""
        if len(values) < window:
            return values
        return pd.Series(values).rolling(window=window, center=True, min_periods=1).mean().tolist()
    
    def extract_features_from_video(self, pose_data):
        """
        Extract all ~200 features from pose sequence
        
        Args:
            pose_data (dict): Loaded pose JSON data
            
        Returns:
            dict: Feature dictionary with ~200 features
        """
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
            logger.warning("Too few valid frames")
            return None
        
        # Convert to DataFrame for easier processing
        geo_df = pd.DataFrame(geometric_sequence)
        
        # Apply smoothing if configured
        if self.config['feature_engineering']['apply_smoothing']:
            window = self.config['feature_engineering']['smoothing_window']
            for col in geo_df.columns:
                geo_df[col] = self.smooth_sequence(geo_df[col].tolist(), window)
        
        features = {}
        
        # ===== CATEGORY 1: TEMPORAL STATISTICS (105 features) =====
        # For each geometric feature: mean, std, min, max, range
        for col in geo_df.columns:
            features[f'{col}_mean'] = geo_df[col].mean()
            features[f'{col}_std'] = geo_df[col].std()
            features[f'{col}_min'] = geo_df[col].min()
            features[f'{col}_max'] = geo_df[col].max()
            features[f'{col}_range'] = geo_df[col].max() - geo_df[col].min()
        
        # ===== CATEGORY 2: VELOCITY FEATURES (42 features) =====
        # Calculate velocities (first derivative)
        for col in geo_df.columns:
            velocities = np.diff(geo_df[col].values)
            features[f'{col}_velocity_mean'] = np.mean(np.abs(velocities))
            features[f'{col}_velocity_max'] = np.max(np.abs(velocities))
        
        # ===== CATEGORY 3: IMPACT FRAME FEATURES (21 features) =====
        # Find impact frame (max foot velocity)
        try:
            # Calculate foot velocities
            left_foot_positions = [frame['landmarks'][self.LANDMARKS['left_foot']]['y'] 
                                  for frame in poses]
            right_foot_positions = [frame['landmarks'][self.LANDMARKS['right_foot']]['y'] 
                                   for frame in poses]
            
            left_foot_velocities = np.abs(np.diff(left_foot_positions))
            right_foot_velocities = np.abs(np.diff(right_foot_positions))
            
            # Determine kicking foot (higher max velocity)
            if np.max(left_foot_velocities) > np.max(right_foot_velocities):
                impact_frame_idx = np.argmax(left_foot_velocities)
                kicking_foot = 'left'
            else:
                impact_frame_idx = np.argmax(right_foot_velocities)
                kicking_foot = 'right'
            
            # Ensure impact frame is within bounds
            impact_frame_idx = min(impact_frame_idx, len(geometric_sequence) - 1)
            
            # Extract features at impact
            impact_features = geometric_sequence[impact_frame_idx]
            for key, value in impact_features.items():
                features[f'{key}_at_impact'] = value
            
            # Additional impact features
            features['impact_frame_number'] = impact_frame_idx
            features['impact_timestamp'] = poses[impact_frame_idx]['timestamp']
            features['kicking_foot'] = 1 if kicking_foot == 'right' else 0
            features['foot_velocity_at_impact'] = max(
                left_foot_velocities[impact_frame_idx] if impact_frame_idx < len(left_foot_velocities) else 0,
                right_foot_velocities[impact_frame_idx] if impact_frame_idx < len(right_foot_velocities) else 0
            )
            
        except Exception as e:
            logger.warning(f"Error extracting impact features: {e}")
            # Fill with zeros if impact frame detection fails
            for col in geo_df.columns:
                features[f'{col}_at_impact'] = 0
            features['impact_frame_number'] = 0
            features['impact_timestamp'] = 0
            features['kicking_foot'] = 0
            features['foot_velocity_at_impact'] = 0
        
        # ===== CATEGORY 4: ADDITIONAL BIOMECHANICAL FEATURES (~20 features) =====
        
        # Symmetry score
        features['knee_angle_symmetry'] = 1 - abs(
            features['left_knee_angle_mean'] - features['right_knee_angle_mean']
        ) / 180.0
        
        # Range of motion
        features['total_knee_extension_left'] = features['left_knee_angle_range']
        features['total_knee_extension_right'] = features['right_knee_angle_range']
        
        # Movement smoothness (jerk - third derivative)
        for col in ['left_knee_angle', 'right_knee_angle']:
            if col in geo_df.columns:
                velocities = np.diff(geo_df[col].values)
                accelerations = np.diff(velocities)
                if len(accelerations) > 0:
                    features[f'{col}_smoothness'] = -np.mean(np.abs(np.diff(accelerations)))  # Negative jerk
                else:
                    features[f'{col}_smoothness'] = 0
        
        # Stability metrics
        features['hip_stability'] = -features['hip_width_std']  # Lower std = more stable
        features['shoulder_stability'] = -features['shoulder_width_std']
        
        # Phase analysis (backswing, impact, follow-through)
        total_frames = len(geometric_sequence)
        if features['impact_frame_number'] > 0:
            features['backswing_duration_ratio'] = features['impact_frame_number'] / total_frames
            features['follow_through_duration_ratio'] = (total_frames - features['impact_frame_number']) / total_frames
        else:
            features['backswing_duration_ratio'] = 0.5
            features['follow_through_duration_ratio'] = 0.5
        
        logger.info(f"Extracted {len(features)} features")
        
        return features
    
    def create_training_dataset(self, pose_dir, labels_path, output_path):
        """
        Create complete training dataset
        
        Args:
            pose_dir (str): Directory with pose JSON files
            labels_path (str): Path to labels.csv
            output_path (str): Output path for training_data.csv
        """
        pose_dir = Path(pose_dir)
        
        # Load labels
        labels_df = pd.read_csv(labels_path)
        logger.info(f"Loaded {len(labels_df)} labels")
        
        # Process each pose file
        all_features = []
        
        for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Extracting features"):
            video_id = row['video_id']
            
            # Find corresponding pose file
            pose_file = pose_dir / f"{Path(video_id).stem}_poses.json"
            
            if not pose_file.exists():
                logger.warning(f"Pose file not found for {video_id}")
                continue
            
            # Load pose data
            with open(pose_file, 'r') as f:
                pose_data = json.load(f)
            
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
        
        # Create DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Reorder columns (video_id and targets first)
        target_cols = ['video_id', 'stability', 'power', 'technique', 'balance', 'overall']
        feature_cols = [col for col in features_df.columns if col not in target_cols]
        features_df = features_df[target_cols + feature_cols]
        
        # Save
        features_df.to_csv(output_path, index=False)
        logger.info(f"Saved training dataset to {output_path}")
        logger.info(f"  Total samples: {len(features_df)}")
        logger.info(f"  Total features: {len(feature_cols)}")
        
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
