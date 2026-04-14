import os
import cv2
import torch
import numpy as np
import pandas as pd
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from tqdm import tqdm

# Load the model and processor
print("Loading model...")
model_name = "jatinmehra/Accident-Detection-using-Dashcam"
base_model = "MCG-NJU/videomae-base"

# Use the base model's processor since the fine-tuned model doesn't have one
processor = VideoMAEImageProcessor.from_pretrained(base_model)
model = VideoMAEForVideoClassification.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print(f"Model loaded on {device}")

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
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    
    if len(frames) != num_frames:
        return None
    
    return frames

def predict_collision(video_path):
    """Predict if video contains collision"""
    try:
        # Extract frames
        frames = extract_frames(video_path)
        
        if frames is None:
            print(f"Failed to extract frames from {video_path}")
            return 0
        
        # Preprocess frames
        inputs = processor(frames, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = logits.argmax(-1).item()
        
        return predicted_class
    
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        return 0

# Main execution
if __name__ == "__main__":
    test_folder = "test"
    test_csv = "test.csv"
    
    # Read test IDs
    test_df = pd.read_csv(test_csv)
    
    results = []
    
    print(f"\nProcessing {len(test_df)} test videos...")
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        video_id = row['id']
        # Format video_id with leading zeros (5 digits)
        video_path = os.path.join(test_folder, f"{int(video_id):05d}.mp4")
        
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            prediction = 0
        else:
            prediction = predict_collision(video_path)
        
        results.append({
            'id': video_id,
            'target': prediction
        })
    
    # Create submission file
    submission_df = pd.DataFrame(results)
    submission_df.to_csv('submission.csv', index=False)
    
    print(f"\nPredictions saved to submission.csv")
    print(f"Total videos processed: {len(results)}")
    print(f"Collisions detected: {submission_df['target'].sum()}")
    print(f"No collisions: {(submission_df['target'] == 0).sum()}")
