import cv2
import numpy as np
from pathlib import Path
import yaml
import logging
from tqdm import tqdm
import mediapipe as mp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoValidator:
    """Validate video quality and camera angle for pose extraction"""
    
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.config = config['data']
            self.angle_config = config.get('angle_detection', {})
        
        # Initialize MediaPipe for angle detection
        self.mp_pose = mp.solutions.pose
        self.pose = None  # Lazy initialization
    
    def detect_camera_angle(self, video_path):
        """
        Detect camera angle from video
        
        Returns:
            str: 'side', 'diagonal', 'behind', 'front', or 'unknown'
        """
        if not self.angle_config.get('enabled', False):
            return 'unknown'
        
        # Lazy init MediaPipe (only if angle detection enabled)
        if self.pose is None:
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,  # Use lighter model for validation
                min_detection_confidence=0.5
            )
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return 'unknown'
        
        # Sample middle frame for analysis
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return 'unknown'
        
        # Process with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return 'unknown'
        
        # Calculate shoulder and hip widths
        landmarks = results.pose_landmarks.landmark
        
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        # Width in normalized coordinates (0-1)
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        hip_width = abs(left_hip.x - right_hip.x)
        
        # Calculate shoulder-hip symmetry
        shoulder_center = (left_shoulder.x + right_shoulder.x) / 2
        hip_center = (left_hip.x + right_hip.x) / 2
        body_alignment = abs(shoulder_center - hip_center)
        
        # Classify angle based on body width and alignment
        # Side view: Body appears narrow (profile)
        if shoulder_width < 0.15 and hip_width < 0.15:
            return 'side'
        
        # Behind/front: Body appears wide and centered
        elif shoulder_width > 0.25 and body_alignment < 0.05:
            # Check if left/right shoulders are roughly equal distance from edges
            left_edge_dist = left_shoulder.x
            right_edge_dist = 1 - right_shoulder.x
            
            if abs(left_edge_dist - right_edge_dist) < 0.1:
                return 'behind'  # Symmetrical = behind/front
            else:
                return 'diagonal'
        
        # Diagonal: In between
        elif 0.15 <= shoulder_width <= 0.25:
            return 'diagonal'
        
        # Front view: Similar to behind but person facing camera
        # (Hard to distinguish from behind without motion analysis)
        else:
            return 'unknown'
    
    def validate_video(self, video_path):
        """
        Validate a single video
        
        Returns:
            (bool, str, str): (is_valid, message, angle)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return False, "Cannot open video file", 'unknown'
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        file_size_mb = Path(video_path).stat().st_size / (1024 * 1024)
        
        cap.release()
        
        # Validate resolution (check if EITHER dimension meets minimum)
        # This handles both portrait and landscape
        min_width = min(width, height)
        min_height = max(width, height)
        
        if min_width < self.config['min_resolution_width']:
            return False, f"Resolution too low ({width}x{height}). Min: {self.config['min_resolution_width']}px", 'unknown'
        
        # Validate FPS
        if fps < self.config['min_fps']:
            return False, f"FPS too low ({fps:.1f}). Min: {self.config['min_fps']}", 'unknown'
        
        # Validate duration
        if duration < self.config['min_duration_sec']:
            return False, f"Video too short ({duration:.1f}s). Min: {self.config['min_duration_sec']}s", 'unknown'
        
        if duration > self.config['max_duration_sec']:
            return False, f"Video too long ({duration:.1f}s). Max: {self.config['max_duration_sec']}s", 'unknown'
        
        # Validate file size
        if file_size_mb > self.config['max_file_size_mb']:
            return False, f"File too large ({file_size_mb:.1f}MB). Max: {self.config['max_file_size_mb']}MB", 'unknown'
        
        # Detect camera angle
        angle = self.detect_camera_angle(video_path)
        
        # Check if angle is acceptable
        if self.angle_config.get('enabled', False):
            rejected_angles = self.angle_config.get('reject_angles', [])
            
            if angle in rejected_angles:
                return False, f"Camera angle '{angle}' not suitable. Please film from the side or at an angle.", angle
            
            # Warn about suboptimal angles
            warning_angles = self.angle_config.get('warning_angles', [])
            if angle in warning_angles:
                message = f"⚠️  ACCEPTED with warning ({width}x{height}, {fps:.1f}fps, {duration:.1f}s, angle: {angle})"
            else:
                message = f"✓ OK ({width}x{height}, {fps:.1f}fps, {duration:.1f}s, angle: {angle})"
        else:
            message = f"OK ({width}x{height}, {fps:.1f}fps, {duration:.1f}s)"
        
        return True, message, angle
    
    def validate_directory(self, video_dir, output_dir=None):
        """Validate all videos in directory"""
        video_dir = Path(video_dir)
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all videos - CASE INSENSITIVE
        video_files = []
        all_files = list(video_dir.iterdir())
        
        for file in all_files:
            if file.is_file():
                # Check if extension matches (case-insensitive)
                file_ext_lower = file.suffix.lower()
                config_exts_lower = [ext.lower() for ext in self.config['video_formats']]
                
                if file_ext_lower in config_exts_lower:
                    video_files.append(file)
        
        logger.info(f"Found {len(video_files)} videos to validate")
        
        valid_count = 0
        invalid_videos = []
        angle_stats = {'side': 0, 'diagonal': 0, 'behind': 0, 'front': 0, 'unknown': 0}
        
        for video_path in tqdm(video_files, desc="Validating"):
            is_valid, message, angle = self.validate_video(str(video_path))
            
            if is_valid:
                valid_count += 1
                logger.info(f"✓ {video_path.name}: {message}")
                
                # Track angle statistics
                angle_stats[angle] = angle_stats.get(angle, 0) + 1
                
                # Copy to output directory if specified
                if output_dir:
                    import shutil
                    # Add angle to filename for easy filtering later
                    if self.angle_config.get('enabled', False) and angle != 'unknown':
                        output_name = f"{video_path.stem}_angle_{angle}{video_path.suffix}"
                    else:
                        output_name = video_path.name
                    
                    shutil.copy2(video_path, output_dir / output_name)
            else:
                logger.warning(f"✗ {video_path.name}: {message}")
                invalid_videos.append((video_path.name, message))
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"Validation complete:")
        logger.info(f"  Valid: {valid_count}/{len(video_files)}")
        logger.info(f"  Invalid: {len(invalid_videos)}")
        
        if self.angle_config.get('enabled', False) and valid_count > 0:
            logger.info(f"\nCamera Angle Distribution:")
            for angle, count in sorted(angle_stats.items()):
                if count > 0:
                    percentage = (count / valid_count) * 100
                    logger.info(f"  {angle.capitalize()}: {count} ({percentage:.1f}%)")
        
        if invalid_videos:
            logger.info(f"\nInvalid videos:")
            for name, reason in invalid_videos:
                logger.info(f"  - {name}: {reason}")
        
        return valid_count, invalid_videos, angle_stats
    
    def __del__(self):
        """Cleanup MediaPipe"""
        if self.pose is not None:
            self.pose.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate video quality and camera angle')
    parser.add_argument('--input', required=True, help='Input directory with videos')
    parser.add_argument('--output', help='Output directory for valid videos (optional)')
    parser.add_argument('--config', default='config/config.yaml', help='Config file')
    
    args = parser.parse_args()
    
    validator = VideoValidator(config_path=args.config)
    validator.validate_directory(args.input, args.output)


if __name__ == "__main__":
    main()