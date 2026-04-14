import cv2
import torch
import numpy as np
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

# Load the model and processor
print("Loading model...")
model_name = "jatinmehra/Accident-Detection-using-Dashcam"
base_model = "MCG-NJU/videomae-base"

processor = VideoMAEImageProcessor.from_pretrained(base_model)
model = VideoMAEForVideoClassification.from_pretrained(model_name)

# Force CPU usage due to CUDA compatibility issues with RTX 5050
device = torch.device("cpu")
model.to(device)
model.eval()

print(f"Model loaded on {device}\n")

def extract_frames(video_path, num_frames=16):
    """Extract frames from video"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
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
        print(f"Processing: {video_path}")
        
        # Extract frames
        frames = extract_frames(video_path)
        
        if frames is None:
            print(f"  ❌ Failed to extract frames\n")
            return None
        
        print(f"  ✓ Extracted {len(frames)} frames")
        
        # Preprocess frames
        inputs = processor(frames, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = logits.argmax(-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        result = "COLLISION DETECTED" if predicted_class == 1 else "NO COLLISION"
        print(f"  Prediction: {result}")
        print(f"  Confidence: {confidence:.2%}\n")
        
        return predicted_class, confidence
    
    except Exception as e:
        print(f"  ❌ Error: {str(e)}\n")
        return None

# Test the videos
if __name__ == "__main__":
    videos = ["5.mp4", "6.mkv"]
    
    print("="*50)
    print("Testing Custom Videos")
    print("="*50 + "\n")
    
    for video in videos:
        result = predict_collision(video)
        if result:
            pred, conf = result
            print(f"Summary for {video}: {'Collision' if pred == 1 else 'No Collision'} ({conf:.2%})")
        else:
            print(f"Summary for {video}: Failed to process")
        print("-"*50 + "\n")
