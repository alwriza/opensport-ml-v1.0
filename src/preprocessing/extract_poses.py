import cv2
import mediapipe as mp
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PoseExtractor:
    """Extract poses from video using MediaPipe"""
    
    def __init__(self, config_path="config/config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.config['pose_extraction']['static_image_mode'],
            model_complexity=self.config['pose_extraction']['model_complexity'],
            min_detection_confidence=self.config['pose_extraction']['min_detection_confidence'],
            min_tracking_confidence=self.config['pose_extraction']['min_tracking_confidence']
        )
        
        self.fps_subsample = self.config['pose_extraction']['fps_subsample']
    
    def extract_from_video(self, video_path):
        """
        Extract poses from a single video
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            list: List of pose data dictionaries (one per frame)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        logger.info(f"Processing {video_path}")
        logger.info(f"  FPS: {fps}, Duration: {duration:.2f}s, Frames: {total_frames}")
        
        poses = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Subsample frames
            if frame_count % self.fps_subsample == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = self.pose.process(frame_rgb)
                
                if results.pose_landmarks:
                    # Extract landmark data
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
                else:
                    logger.warning(f"  No pose detected in frame {frame_count}")
            
            frame_count += 1
        
        cap.release()
        
        logger.info(f"  Extracted poses from {len(poses)} frames")
        
        return poses
    
    def extract_batch(self, video_dir, output_dir):
        """
        Extract poses from all videos in directory
        
        Args:
            video_dir (str): Directory containing videos
            output_dir (str): Directory to save pose JSON files
        """
        video_dir = Path(video_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all video files
        video_files = []
        all_files = list(video_dir.iterdir())

        for file in all_files:
            if file.is_file():
                file_ext_lower = file.suffix.lower()
                config_exts_lower = [ext.lower() for ext in self.config['data']['video_formats']]
                
                if file_ext_lower in config_exts_lower:
                    video_files.append(file)
        
        logger.info(f"Found {len(video_files)} videos to process")
        
        # Process each video
        for video_path in tqdm(video_files, desc="Extracting poses"):
            output_path = output_dir / f"{video_path.stem}_poses.json"
            
            # Skip if already processed
            if output_path.exists():
                logger.info(f"Skipping {video_path.name} (already processed)")
                continue
            
            # Extract poses
            poses = self.extract_from_video(str(video_path))
            
            if poses:
                # Save to JSON
                with open(output_path, 'w') as f:
                    json.dump({
                        'video_id': video_path.name,
                        'fps': cv2.VideoCapture(str(video_path)).get(cv2.CAP_PROP_FPS),
                        'total_frames': len(poses),
                        'poses': poses
                    }, f, indent=2)
                
                logger.info(f"  Saved to {output_path}")
            else:
                logger.error(f"  Failed to extract poses from {video_path.name}")
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'pose'):
            self.pose.close()


def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract poses from football kick videos')
    parser.add_argument('--input', required=True, help='Input video directory')
    parser.add_argument('--output', required=True, help='Output pose data directory')
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    extractor = PoseExtractor(config_path=args.config)
    extractor.extract_batch(args.input, args.output)
    
    logger.info("Pose extraction complete!")


if __name__ == "__main__":
    main()
