"""
Contoh inference standalone (tanpa Streamlit)
untuk testing model secara langsung
"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def load_model(model_path, num_classes=37):
    """Load model Faster R-CNN"""
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, device

def preprocess_image(image_path):
    """Load dan preprocess gambar"""
    image = Image.open(image_path).convert("RGB")
    transform = T.ToTensor()
    image_tensor = transform(image)
    return image, image_tensor

def detect(model, image_tensor, device, confidence_threshold=0.5):
    """Jalankan deteksi"""
    with torch.no_grad():
        predictions = model([image_tensor.to(device)])
    
    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    
    # Filter
    mask = scores >= confidence_threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]
    
    return boxes, labels, scores

def label_to_char(label):
    """Convert label to character"""
    if label == 0:
        return "BG"
    elif 1 <= label <= 10:
        return str(label - 1)
    elif 11 <= label <= 36:
        return chr(ord('A') + label - 11)
    else:
        return str(label)

def visualize_results(image, boxes, labels, scores):
    """Visualisasi hasil deteksi"""
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        
        # Draw box
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
        
        # Draw label
        text = f"{label_to_char(label)}: {score:.2f}"
        text_bbox = draw.textbbox((x1, y1 - 25), text, font=font)
        draw.rectangle(text_bbox, fill="red")
        draw.text((x1, y1 - 25), text, fill="white", font=font)
    
    return image

def main():
    # Konfigurasi
    MODEL_PATH = "plate_detection_model.pth"
    IMAGE_PATH = "test_image.jpg"
    NUM_CLASSES = 37
    CONFIDENCE_THRESHOLD = 0.5
    
    print("Loading model...")
    model, device = load_model(MODEL_PATH, NUM_CLASSES)
    print(f"Model loaded on {device}")
    
    print(f"Processing image: {IMAGE_PATH}")
    image, image_tensor = preprocess_image(IMAGE_PATH)
    
    print("Running detection...")
    boxes, labels, scores = detect(model, image_tensor, device, CONFIDENCE_THRESHOLD)
    
    print(f"\nDetected {len(boxes)} object(s):")
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        print(f"  {i+1}. {label_to_char(label)} - Confidence: {score:.2%} - Box: {box}")
    
    # Visualize
    result_image = visualize_results(image.copy(), boxes, labels, scores)
    
    # Save result
    result_image.save("result.jpg")
    print("\nResult saved to 'result.jpg'")
    
    # Display with matplotlib
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(result_image)
    plt.title(f"Detection ({len(boxes)} objects)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("comparison.jpg")
    print("Comparison saved to 'comparison.jpg'")
    plt.show()

if __name__ == "__main__":
    main()
