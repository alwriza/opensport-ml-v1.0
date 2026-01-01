"""
Video Validation Script
Validates video quality before processing.
"""

import cv2
import numpy as np
from pathlib import Path
import yaml
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoValidator:
    """Validate video quality for pose extraction"""
    
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['data']
    
    def validate_video(self, video_path):
        """
        Validate a single video
        
        Returns:
            (bool, str): (is_valid, message)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return False, "Cannot open video file"
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        file_size_mb = Path(video_path).stat().st_size / (1024 * 1024)
        
        cap.release()
        
        # Validate resolution
        if width < self.config['min_resolution_width'] or height < self.config['min_resolution_height']:
            return False, f"Resolution too low ({width}x{height}). Min: {self.config['min_resolution_width']}x{self.config['min_resolution_height']}"
        
        # Validate FPS
        if fps < self.config['min_fps']:
            return False, f"FPS too low ({fps:.1f}). Min: {self.config['min_fps']}"
        
        # Validate duration
        if duration < self.config['min_duration_sec']:
            return False, f"Video too short ({duration:.1f}s). Min: {self.config['min_duration_sec']}s"
        
        if duration > self.config['max_duration_sec']:
            return False, f"Video too long ({duration:.1f}s). Max: {self.config['max_duration_sec']}s"
        
        # Validate file size
        if file_size_mb > self.config['max_file_size_mb']:
            return False, f"File too large ({file_size_mb:.1f}MB). Max: {self.config['max_file_size_mb']}MB"
        
        return True, f"OK ({width}x{height}, {fps:.1f}fps, {duration:.1f}s, {file_size_mb:.1f}MB)"
    
    def validate_directory(self, video_dir, output_dir=None):
        """Validate all videos in directory"""
        video_dir = Path(video_dir)
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all videos
        video_files = []
        for ext in self.config['video_formats']:
            video_files.extend(video_dir.glob(f"*{ext}"))
        
        logger.info(f"Found {len(video_files)} videos to validate")
        
        valid_count = 0
        invalid_videos = []
        
        for video_path in tqdm(video_files, desc="Validating"):
            is_valid, message = self.validate_video(str(video_path))
            
            if is_valid:
                valid_count += 1
                logger.info(f"✓ {video_path.name}: {message}")
                
                # Copy to output directory if specified
                if output_dir:
                    import shutil
                    shutil.copy2(video_path, output_dir / video_path.name)
            else:
                logger.warning(f"✗ {video_path.name}: {message}")
                invalid_videos.append((video_path.name, message))
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"Validation complete:")
        logger.info(f"  Valid: {valid_count}/{len(video_files)}")
        logger.info(f"  Invalid: {len(invalid_videos)}")
        
        if invalid_videos:
            logger.info(f"\nInvalid videos:")
            for name, reason in invalid_videos:
                logger.info(f"  - {name}: {reason}")
        
        return valid_count, invalid_videos


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate video quality')
    parser.add_argument('--input', required=True, help='Input directory with videos')
    parser.add_argument('--output', help='Output directory for valid videos (optional)')
    parser.add_argument('--config', default='config/config.yaml', help='Config file')
    
    args = parser.parse_args()
    
    validator = VideoValidator(config_path=args.config)
    validator.validate_directory(args.input, args.output)


if __name__ == "__main__":
    main()
