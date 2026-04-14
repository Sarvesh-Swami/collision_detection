import cv2
import torch
import numpy as np
import os
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

# Load the model and processor
print("Loading model...")
model_name = "jatinmehra/Accident-Detection-using-Dashcam"
base_model = "MCG-NJU/videomae-base"

processor = VideoMAEImageProcessor.from_pretrained(base_model)
model = VideoMAEForVideoClassification.from_pretrained(model_name)

# Force CPU usage
device = torch.device("cpu")
model.to(device)
model.eval()

print(f"Model loaded on {device}")
print("="*60)

def extract_frames(video_path, num_frames=16):
    """Extract frames from video"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"  Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s duration")
    
    if total_frames == 0:
        cap.release()
        return None
    
    # Sample frames uniformly
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    
    if len(frames) != num_frames:
        return None
    
    return frames

def predict_collision(video_path):
    """Predict if video contains collision"""
    try:
        print(f"\n{'='*60}")
        print(f"Processing: {video_path}")
        print('='*60)
        
        # Check if file exists
        if not os.path.exists(video_path):
            print(f"  ❌ File not found: {video_path}")
            return None
        
        # Extract frames
        print("  Extracting frames...")
        frames = extract_frames(video_path)
        
        if frames is None:
            print(f"  ❌ Failed to extract frames from video")
            print(f"  Tip: Try converting to MP4 format or check if file is corrupted")
            return None
        
        print(f"  ✓ Successfully extracted {len(frames)} frames")
        
        # Preprocess frames
        print("  Running model inference...")
        inputs = processor(frames, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = logits.argmax(-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        print("\n" + "="*60)
        if predicted_class == 1:
            print("  🚨 RESULT: COLLISION DETECTED")
        else:
            print("  ✅ RESULT: NO COLLISION")
        print(f"  Confidence: {confidence:.2%}")
        print("="*60 + "\n")
        
        return predicted_class, confidence
    
    except Exception as e:
        print(f"\n  ❌ Error: {str(e)}")
        print(f"  Tip: Make sure the video file is valid and not corrupted\n")
        return None

# Interactive loop
if __name__ == "__main__":
    print("\n🎥 Video Collision Detection System")
    print("="*60)
    print("Enter video filename to analyze (or 'quit' to exit)")
    print("Examples: 5.mp4, 6.mkv, test/00001.mp4")
    print("="*60 + "\n")
    
    while True:
        try:
            video_file = input("Enter video filename: ").strip()
            
            if video_file.lower() in ['quit', 'exit', 'q']:
                print("\nExiting... Goodbye!")
                break
            
            if not video_file:
                print("Please enter a valid filename\n")
                continue
            
            predict_collision(video_file)
            
        except KeyboardInterrupt:
            print("\n\nExiting... Goodbye!")
            break
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}\n")
