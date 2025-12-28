import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import pickle
from typing import List, Dict, Tuple
import json

# Import from project utils
from utils.hand_tracking import HandTracker

def extract_hand_landmarks_from_video(video_path: str, 
                                     hand_tracker: HandTracker,
                                     max_frames: int = 100,
                                     min_confidence: float = 0.5) -> List[np.ndarray]:
    """
    Extract hand landmarks from a video file
    
    Args:
        video_path: Path to video file
        hand_tracker: Initialized HandTracker instance
        max_frames: Maximum number of frames to process
        min_confidence: Minimum landmark confidence
        
    Returns:
        List of hand landmark features (42-dim vectors) or empty list
    """
    landmarks_sequence = []
    
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            hand_landmarks_list = hand_tracker.process_frame(frame)
            
            if hand_landmarks_list:
                # Extract features from first hand
                features = hand_tracker.extract_single_hand_features(hand_landmarks_list)
                if features is not None:
                    # Filter by confidence (simplified)
                    if np.mean(features) > 0:  # Basic check
                        landmarks_sequence.append(features)
            
            frame_count += 1
        
        cap.release()
        return landmarks_sequence
        
    except Exception as e:
        print(f"❌ Error processing video {video_path}: {str(e)}")
        return []

def preprocess_wlasl_dataset(wlasl_dir: str = 'data/wlasl', 
                            output_dir: str = 'data/processed_wlasl',
                            max_videos_per_class: int = 50,
                            max_frames_per_video: int = 30) -> Dict:
    """
    Preprocess WLASL dataset: extract hand landmarks from videos
    
    Args:
        wlasl_dir: Path to WLASL dataset directory
        output_dir: Directory to save processed features
        max_videos_per_class: Limit videos per gesture class
        max_frames_per_video: Max frames to extract per video
        
    Returns:
        Dictionary with processed data statistics
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize hand tracker
    hand_tracker = HandTracker(static_image_mode=False, 
                              max_num_hands=1,
                              min_detection_confidence=0.7,
                              min_tracking_confidence=0.7)
    
    processed_data = {
        'features': {},  # {gesture_name: [video_features_list]}
        'labels': [],
        'gesture_to_idx': {},
        'stats': {'total_videos': 0, 'total_frames': 0, 'classes': 0}
    }
    
    # WLASL metadata is in XML, but for simplicity, assume videos are organized by class folders
    # In practice, parse WLASL_v0.3.xml for mappings
    videos_dir = os.path.join(wlasl_dir, 'videos')
    
    if not os.path.exists(videos_dir):
        print(f"❌ Videos directory not found: {videos_dir}")
        print("Run download_dataset.py first and download videos.")
        return processed_data
    
    # Get all gesture classes (subdirectories)
    gesture_classes = [d for d in os.listdir(videos_dir) 
                      if os.path.isdir(os.path.join(videos_dir, d))]
    
    gesture_idx = 0
    for class_name in tqdm(gesture_classes[:20], desc="Processing classes"):  # Limit to 20 classes for demo
        class_dir = os.path.join(videos_dir, class_name)
        video_files = [f for f in os.listdir(class_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        
        # Limit videos per class
        video_files = video_files[:max_videos_per_class]
        
        class_features = []
        for video_file in tqdm(video_files, desc=f"Processing {class_name}", leave=False):
            video_path = os.path.join(class_dir, video_file)
            landmarks_seq = extract_hand_landmarks_from_video(
                video_path, hand_tracker, max_frames_per_video
            )
            
            if landmarks_seq:
                class_features.append(landmarks_seq)
                processed_data['stats']['total_frames'] += len(landmarks_seq)
        
        if class_features:
            processed_data['features'][class_name] = class_features
            processed_data['gesture_to_idx'][class_name] = gesture_idx
            processed_data['labels'].extend([gesture_idx] * len(class_features))
            gesture_idx += 1
        
        processed_data['stats']['total_videos'] += len(class_features)
    
    processed_data['stats']['classes'] = len(gesture_classes)
    
    # Save processed data
    with open(os.path.join(output_dir, 'processed_features.pkl'), 'wb') as f:
        pickle.dump(processed_data, f)
    
    # Save metadata
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump({
            'num_classes': processed_data['stats']['classes'],
            'total_videos': processed_data['stats']['total_videos'],
            'total_frames': processed_data['stats']['total_frames'],
            'gesture_to_idx': processed_data['gesture_to_idx']
        }, f, indent=2)
    
    print(f"✅ Preprocessing complete!")
    print(f"Processed {processed_data['stats']['total_videos']} videos from {processed_data['stats']['classes']} classes")
    print(f"Total frames extracted: {processed_data['stats']['total_frames']}")
    
    return processed_data

def preprocess_ms_asl_dataset(ms_asl_dir: str = 'data/ms_asl',
                            output_dir: str = 'data/processed_ms_asl',
                            max_videos_per_class: int = 50,
                            max_frames_per_video: int = 30) -> Dict:
    """
    Preprocess MS-ASL dataset (similar structure assumed)
    """
    # MS-ASL has different structure; adapt as needed
    print("MS-ASL preprocessing (placeholder - adapt based on actual dataset structure)")
    print("MS-ASL typically has CSV metadata with video paths.")
    # Implement similar to WLASL but parse CSV metadata
    return {}

if __name__ == "__main__":
    # Choose dataset to preprocess
    dataset_choice = input("Choose dataset to preprocess (1: WLASL, 2: MS-ASL): ")
    if dataset_choice == "1":
        data = preprocess_wlasl_dataset()
        print(f"Processed data keys: {list(data.keys())}")
    elif dataset_choice == "2":
        data = preprocess_ms_asl_dataset()
    else:
        print("Invalid choice.")
