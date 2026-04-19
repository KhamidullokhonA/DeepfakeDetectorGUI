import time
import csv 
import os
import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import argparse

# Load your trained model
def load_model(model_path="models/best_model.pt"):
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.4),
        torch.nn.Linear(in_features, 2)
    )
    # strict=False added just in case of minor state_dict mismatches
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    model.eval()
    return model

# Preprocess and classify image
def predict_image(image_path, model):
    # --- PREPROCESSING ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    # --- INFERENCE & TIMING ---
    with torch.no_grad():
        inference_start = time.time() # START EXACT CLOCK
        
        output = model(input_tensor)
        
        inference_time = time.time() - inference_start # STOP EXACT CLOCK

        probs = torch.softmax(output, dim=1)[0]
        pred = torch.argmax(probs).item()

    label = "FAKE" if pred == 1 else "REAL"
    confidence = probs[pred].item() # Get probability of the chosen label
    
    # Calculate FPS (1 divided by the time it took to process 1 frame)
    fps = 1.0 / inference_time if inference_time > 0 else float('inf')

    # --- CSV LOGGING FOR RESEARCH ---
    csv_file = "research_results.csv"
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write headers if this is the first time running the script
            writer.writerow(["Filename", "Prediction", "Confidence", "Inference_Time_Sec", "FPS"])
        
        # Log the specific run data
        writer.writerow([os.path.basename(image_path), label, f"{confidence:.4f}", f"{inference_time:.6f}", f"{fps:.2f}"])

    # --- CONSOLE OUTPUT ---
    print(f"\n🧠 Prediction: {label} (Confidence: {confidence:.2%})")
    print(f"Real: {probs[0]:.3f} | Fake: {probs[1]:.3f}")
    print("-" * 30)
    print(f"⚡ Inference Time: {inference_time:.4f} seconds")
    print(f"🎥 Model Speed:    {fps:.2f} FPS")
    print(f"💾 Logged to:      {csv_file}")

# Run from terminal
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to image file (.jpg/.png)")
    args = parser.parse_args()

    print("Loading model weights... (This cold start is not timed)")
    model = load_model()
    predict_image(args.image_path, model)
