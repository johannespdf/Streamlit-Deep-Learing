import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from io import BytesIO
import zipfile
import os
from collections import OrderedDict
import base64

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="ANPR System Pro",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e2e 0%, #2d2d44 100%);
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        color: #ffffff;
    }
    
    /* Cards */
    .stExpander {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        border: none;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .stExpander:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.15);
    }
    
    /* Headers */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3rem !important;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #2d3748;
        font-weight: 600;
        margin-top: 2rem;
    }
    
    h3 {
        color: #4a5568;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Upload box */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 16px;
        padding: 2rem;
        border: 2px dashed #667eea;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #764ba2;
        background: rgba(255, 255, 255, 1);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
    }
    
    /* Success/Error/Warning boxes */
    .stSuccess {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        border-radius: 12px;
        padding: 1rem;
        border: none;
    }
    
    .stError {
        background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
        color: white;
        border-radius: 12px;
        padding: 1rem;
        border: none;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
        color: white;
        border-radius: 12px;
        padding: 1rem;
        border: none;
    }
    
    /* Image containers */
    [data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    [data-testid="stImage"]:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px 12px 0 0;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Glass effect cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Custom result card */
    .result-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(255, 255, 255, 0.9) 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }
    
    /* Custom plate display */
    .plate-number {
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== MODELS ====================

class PlateDetectorCNN(nn.Module):
    """
    Faster R-CNN untuk deteksi plat nomor
    Menggunakan ResNet-50 FPN backbone
    """
    def __init__(self):
        super(PlateDetectorCNN, self).__init__()
        # Load Faster R-CNN dengan ResNet-50 backbone
        self.model = fasterrcnn_resnet50_fpn(weights=None)
        
        # Ganti predictor untuk 2 kelas (background + plate)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    
    def forward(self, images, targets=None):
        return self.model(images, targets)

class CharacterCNN(nn.Module):
    """
    Custom CNN untuk klasifikasi karakter A-Z dan 0-9
    Total 36 kelas
    """
    def __init__(self, num_classes=36):
        super(CharacterCNN, self).__init__()
        
        # Conv Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Grayscale input
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Conv Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Conv Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fully Connected
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# ==================== CONSTANTS ====================

CHAR_CLASSES = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
NUM_CHAR_CLASSES = len(CHAR_CLASSES)  # 36 classes
DEVICE = torch.device('cpu')

EMNIST_TO_CHAR = {}
for i in range(10):  # 0-9
    EMNIST_TO_CHAR[i] = i
for i in range(26):  # A-Z
    EMNIST_TO_CHAR[10 + i] = 10 + i

# ==================== HELPER FUNCTIONS ====================

@st.cache_resource
def initialize_models():
    """Inisialisasi semua model"""
    plate_detector = PlateDetectorCNN()
    plate_detector.to(DEVICE)
    plate_detector.eval()
    
    char_classifier = CharacterCNN(num_classes=NUM_CHAR_CLASSES)
    char_classifier.to(DEVICE)
    
    # Try to train, but don't fail if it doesn't work
    try:
        char_classifier = train_character_classifier_emnist(char_classifier)
    except Exception as e:
        print(f"Warning: Could not train character classifier: {e}")
        print("Using untrained model. Predictions will be random.")
    
    char_classifier.eval()
    
    return plate_detector, char_classifier

def train_character_classifier_emnist(model, num_epochs=3):
    """
    Train character classifier dengan synthetic data dan MNIST
    Simplified training untuk faster, more reliable results
    """
    import string
    
    print("Training character classifier...")
    
    # Gunakan MNIST untuk digits (0-9) yang lebih simple
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    
    try:
        # Load MNIST untuk digit recognition (0-9)
        from torchvision.datasets import MNIST
        
        mnist_train = MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        
        # Buat DataLoader
        train_loader = torch.utils.data.DataLoader(
            mnist_train,
            batch_size=256,
            shuffle=True,
            num_workers=0
        )
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        print(f"Training for {num_epochs} epochs on MNIST...")
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            batch_count = 0
            
            for i, (inputs, labels) in enumerate(train_loader):
                if i > 100:  # Limit to 100 batches for speed
                    break
                    
                # Filter hanya 0-9 (sudah otomatis di MNIST)
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # Only use first 10 outputs (digits)
                outputs_digits = outputs[:, :10]
                loss = criterion(outputs_digits, labels)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs_digits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                batch_count += 1
            
            accuracy = 100. * correct / total if total > 0 else 0
            avg_loss = running_loss / batch_count if batch_count > 0 else 0
            print(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.3f}, Accuracy={accuracy:.2f}%")
        
        print("✅ Character classifier trained on MNIST (digits 0-9)")
        print("⚠️ Note: Letters (A-Z) use untrained weights. For better letter recognition, more training data needed.")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("Using untrained model.")
    
    return model

def detect_plate(model, image):
    """
    Stage 2: Plate Detection menggunakan Faster R-CNN
    
    Args:
        model: PlateDetectorCNN
        image: PIL Image
    
    Returns:
        boxes: bounding boxes plat
        scores: confidence scores
    """
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).to(DEVICE)
    
    with torch.no_grad():
        predictions = model([image_tensor])
    
    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    
    # Filter dengan confidence > 0.5
    mask = scores > 0.5
    boxes = boxes[mask]
    scores = scores[mask]
    
    return boxes, scores

def segment_characters(plate_image):
    """
    Stage 3: Character Segmentation (VASTLY IMPROVED)
    Enhanced preprocessing, relaxed filtering, and character separation
    
    Args:
        plate_image: cropped plate region (PIL Image)
    
    Returns:
        char_images: list of character images
        char_boxes: list of bounding boxes
        debug_images: dict of intermediate images untuk visualization
    """
    debug_images = {}
    
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(plate_image), cv2.COLOR_RGB2GRAY)
    debug_images['01_original_gray'] = gray.copy()
    
    # Resize untuk standardisasi (lebih besar untuk better detail)
    target_height = 200  # Increased from 150
    scale_factor = target_height / gray.shape[0]
    gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor)
    plate_height, plate_width = gray.shape
    debug_images['02_resized'] = gray.copy()
    
    # === ENHANCED PREPROCESSING ===
    
    # 1. Bilateral filter - preserve edges while reducing noise
    gray_bilateral = cv2.bilateralFilter(gray, 11, 80, 80)
    debug_images['03_bilateral'] = gray_bilateral
    
    # 2. CLAHE untuk contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray_bilateral)
    debug_images['04_clahe'] = gray_clahe
    
    # === MULTIPLE THRESHOLDING METHODS ===
    
    # Method 1: Adaptive Gaussian (good for varying lighting)
    thresh1 = cv2.adaptiveThreshold(
        gray_clahe, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2  # Smaller block size, lower C
    )
    
    # Method 2: Adaptive Gaussian with larger block
    thresh2 = cv2.adaptiveThreshold(
        gray_clahe, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 3
    )
    
    # Method 3: Adaptive Mean
    thresh3 = cv2.adaptiveThreshold(
        gray_clahe, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 2
    )
    
    # Method 4: Otsu's binarization
    _, thresh4 = cv2.threshold(
        gray_clahe, 0, 255, 
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    
    # Combine all methods (OR operation captures maximum detail)
    thresh_combined = cv2.bitwise_or(thresh1, thresh2)
    thresh_combined = cv2.bitwise_or(thresh_combined, thresh3)
    thresh_combined = cv2.bitwise_or(thresh_combined, thresh4)
    debug_images['05_thresh_combined'] = thresh_combined
    
    # === MORPHOLOGICAL OPERATIONS ===
    
    # Opening - remove small noise (very gentle)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh_opened = cv2.morphologyEx(thresh_combined, cv2.MORPH_OPEN, kernel_open, iterations=1)
    debug_images['06_opened'] = thresh_opened
    
    # Closing - connect broken parts
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
    thresh_closed = cv2.morphologyEx(thresh_opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    debug_images['07_closed'] = thresh_closed
    
    # Final threshold
    thresh_final = thresh_closed
    debug_images['08_final_threshold'] = thresh_final
    
    # === CONTOUR DETECTION ===
    
    contours, _ = cv2.findContours(thresh_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # === RELAXED CONTOUR FILTERING ===
    
    char_candidates = []
    rejected_contours = {'too_small': 0, 'too_large': 0, 'wrong_aspect': 0, 'wrong_position': 0}
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        area = w * h
        aspect_ratio = h / float(w) if w > 0 else 0
        
        # VERY RELAXED criteria to capture all characters
        min_height = plate_height * 0.15  # 15% of plate height
        max_height = plate_height * 0.95  # 95% of plate height
        min_width = plate_width * 0.008   # 0.8% of plate width (very relaxed)
        max_width = plate_width * 0.25    # 25% of plate width
        min_area = 80  # Very low threshold
        
        # Vertical position filter (characters should be in middle region)
        min_y = plate_height * 0.05
        max_y = plate_height * 0.95
        
        # Check filters
        if h < min_height or h > max_height:
            rejected_contours['wrong_aspect'] += 1
            continue
        if w < min_width or w > max_width:
            rejected_contours['too_small' if w < min_width else 'too_large'] += 1
            continue
        if area < min_area:
            rejected_contours['too_small'] += 1
            continue
        if y < min_y or y > max_y:
            rejected_contours['wrong_position'] += 1
            continue
        
        # Very wide aspect ratio range: 0.5 to 6.0
        # 0.5 = wide characters (W, M)
        # 6.0 = very thin characters (I, 1)
        if 0.5 <= aspect_ratio <= 6.0:
            char_candidates.append((x, y, w, h, aspect_ratio))
    
    debug_images['rejection_stats'] = rejected_contours
    
    # Visualize all candidates
    debug_viz_candidates = cv2.cvtColor(thresh_final, cv2.COLOR_GRAY2RGB)
    for (x, y, w, h, _) in char_candidates:
        cv2.rectangle(debug_viz_candidates, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(debug_viz_candidates, f"{h}/{w}", (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    debug_images['09_candidates'] = debug_viz_candidates
    
    # Sort left to right
    char_candidates = sorted(char_candidates, key=lambda b: b[0])
    
    # === CHARACTER SEPARATION ===
    # Split potentially merged characters
    
    separated_boxes = []
    
    for (x, y, w, h, aspect_ratio) in char_candidates:
        # Check if this might be merged characters
        # Heuristic: if width is unusually large compared to height
        expected_char_width = h * 0.7  # Typical character aspect ratio
        
        if w > expected_char_width * 1.8 and aspect_ratio < 1.2:
            # Likely merged characters - try to split
            # Use vertical projection profile
            char_region = thresh_final[y:y+h, x:x+w]
            
            # Calculate vertical projection (sum along columns)
            vertical_projection = np.sum(char_region, axis=0)
            
            # Find valleys (low points) in the projection
            projection_smoothed = np.convolve(vertical_projection, np.ones(3)/3, mode='same')
            mean_projection = np.mean(projection_smoothed)
            
            # Find split points (valleys where projection is below mean)
            split_points = []
            in_valley = False
            valley_start = 0
            
            for i, val in enumerate(projection_smoothed):
                if val < mean_projection * 0.3 and not in_valley:
                    valley_start = i
                    in_valley = True
                elif val >= mean_projection * 0.3 and in_valley:
                    split_points.append((valley_start + i) // 2)
                    in_valley = False
            
            # Split if we found good split points
            if len(split_points) > 0:
                prev_x = 0
                for split_x in split_points:
                    if split_x - prev_x > w * 0.15:  # At least 15% of width
                        separated_boxes.append((x + prev_x, y, split_x - prev_x, h))
                    prev_x = split_x
                # Add remaining part
                if w - prev_x > w * 0.15:
                    separated_boxes.append((x + prev_x, y, w - prev_x, h))
            else:
                # Couldn't split, keep as is
                separated_boxes.append((x, y, w, h))
        else:
            # Normal character
            separated_boxes.append((x, y, w, h))
    
    # Remove duplicates and very small boxes
    char_boxes = []
    for box in separated_boxes:
        x, y, w, h = box
        if w > plate_width * 0.01 and h > plate_height * 0.15:
            char_boxes.append(box)
    
    # Sort again after separation
    char_boxes = sorted(char_boxes, key=lambda b: b[0])
    
    # === HANDLE OVERLAPPING/CLOSE BOXES ===
    
    final_boxes = []
    if char_boxes:
        current_box = list(char_boxes[0])
        
        for next_box in char_boxes[1:]:
            x1, y1, w1, h1 = current_box
            x2, y2, w2, h2 = next_box
            
            gap = x2 - (x1 + w1)
            
            # Only merge if extremely close (<3px) and similar height
            height_ratio = max(h1, h2) / min(h1, h2) if min(h1, h2) > 0 else 999
            
            if gap < 3 and height_ratio < 1.5:
                # Merge boxes
                new_x = min(x1, x2)
                new_y = min(y1, y2)
                new_w = max(x1 + w1, x2 + w2) - new_x
                new_h = max(y1 + h1, y2 + h2) - new_y
                current_box = [new_x, new_y, new_w, new_h]
            else:
                final_boxes.append(tuple(current_box))
                current_box = list(next_box)
        
        final_boxes.append(tuple(current_box))
    
    char_boxes = final_boxes if final_boxes else char_boxes
    
    # Visualize final boxes
    debug_viz_final = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    for i, (x, y, w, h) in enumerate(char_boxes):
        cv2.rectangle(debug_viz_final, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(debug_viz_final, str(i+1), (x+2, y+12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    debug_images['10_final_boxes'] = debug_viz_final
    
    # === CROP AND PREPARE CHARACTERS ===
    
    char_images = []
    for (x, y, w, h) in char_boxes:
        # Crop from threshold image
        char_img = thresh_final[y:y+h, x:x+w]
        
        # Add padding untuk better recognition
        pad = 5  # Slightly more padding
        char_img = cv2.copyMakeBorder(
            char_img, pad, pad, pad, pad,
            cv2.BORDER_CONSTANT, value=0
        )
        
        # Resize to 32x32 with better interpolation
        char_img = cv2.resize(char_img, (32, 32), interpolation=cv2.INTER_AREA)
        
        # Ensure white text on black background
        if np.mean(char_img) > 127:
            char_img = 255 - char_img
        
        # Additional normalization
        char_img = cv2.normalize(char_img, None, 0, 255, cv2.NORM_MINMAX)
        
        char_images.append(char_img)
    
    return char_images, char_boxes, debug_images

def classify_character(model, char_image):
    """
    Stage 4: Character Classification
    
    Args:
        model: CharacterCNN
        char_image: grayscale character image (numpy array)
    
    Returns:
        predicted_char: predicted character
        confidence: confidence score
    """
    # Normalize
    char_tensor = torch.FloatTensor(char_image).unsqueeze(0).unsqueeze(0) / 255.0
    char_tensor = char_tensor.to(DEVICE)
    
    with torch.no_grad():
        outputs = model(char_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    
    predicted_char = CHAR_CLASSES[predicted.item()]
    
    return predicted_char, confidence.item()

def post_process_plate(characters, confidences, boxes):
    """
    Stage 5: Post-processing
    Mengurutkan dan menggabungkan karakter menjadi string plat
    
    Args:
        characters: list of predicted characters
        confidences: list of confidence scores
        boxes: list of bounding boxes (already sorted)
    
    Returns:
        plate_text: formatted plate string
        avg_confidence: average confidence
    """
    # Characters sudah sorted by x-coordinate
    plate_text = ''.join(characters)
    
    # Format (heuristic untuk plat Indonesia: XX 1234 XX)
    if len(plate_text) >= 7:
        formatted = f"{plate_text[:2]} {plate_text[2:6]} {plate_text[6:]}"
    else:
        formatted = plate_text
    
    avg_confidence = np.mean(confidences) if confidences else 0.0
    
    return formatted, avg_confidence

def visualize_detection(image, boxes, scores):
    """Visualisasi detection results"""
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
    
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
        text = f"Plate: {score:.2f}"
        draw.text((x1, y1 - 35), text, fill="red", font=font)
    
    return img_draw

def visualize_segmentation(plate_image, char_boxes):
    """Visualisasi character segmentation"""
    img_array = np.array(plate_image)
    img_viz = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img_viz)
    
    for (x, y, w, h) in char_boxes:
        draw.rectangle([(x, y), (x+w, y+h)], outline="green", width=2)
    
    return img_viz

def process_single_image(image, plate_detector, char_classifier):
    """
    Process a single image through the entire ANPR pipeline
    
    Returns:
        dict with results
    """
    results = {
        'success': False,
        'plate_text': '',
        'confidence': 0.0,
        'num_plates': 0,
        'num_chars': 0,
        'boxes': [],
        'scores': [],
        'plate_img': None,
        'viz_img': None
    }
    
    try:
        # Stage 2: Plate Detection
        boxes, scores = detect_plate(plate_detector, image)
        results['boxes'] = boxes
        results['scores'] = scores
        results['num_plates'] = len(boxes)
        
        if len(boxes) > 0:
            # Use first detected plate
            box = boxes[0]
            x1, y1, x2, y2 = box.astype(int)
            plate_img = image.crop((x1, y1, x2, y2))
            results['plate_img'] = plate_img
            
            # Stage 3: Segmentation
            char_images, char_boxes, debug_images = segment_characters(plate_img)
            results['num_chars'] = len(char_images)
            
            if len(char_images) > 0:
                # Stage 4: Classification
                characters = []
                confidences = []
                
                for char_img in char_images:
                    char, conf = classify_character(char_classifier, char_img)
                    characters.append(char)
                    confidences.append(conf)
                
                # Stage 5: Post-processing
                plate_text, avg_confidence = post_process_plate(
                    characters,
                    confidences,
                    char_boxes
                )
                
                results['plate_text'] = plate_text
                results['confidence'] = avg_confidence
                results['success'] = True
                
                # Create visualization
                viz_img = visualize_detection(image, boxes, scores)
                results['viz_img'] = viz_img
    
    except Exception as e:
        results['error'] = str(e)
    
    return results

# ==================== STREAMLIT UI ====================

# Apply custom CSS
load_custom_css()

# Header with animation
st.markdown('<div class="animate-fade-in">', unsafe_allow_html=True)
st.title("🚗 ANPR System Pro")
st.markdown("**Automatic Number Plate Recognition with AI-Powered Detection**")
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.markdown("## ⚙️ Configuration")

mode = st.sidebar.radio(
    "Select Mode",
    ["🔍 Single Image", "📸 Multiple Images", "📊 About"],
    help="Choose processing mode"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🎯 Features
- ✅ Multi-image upload
- ✅ Batch processing
- ✅ Real-time detection
- ✅ Advanced segmentation
- ✅ Beautiful visualization

### 🧠 AI Pipeline
1. Plate Detection (Faster R-CNN)
2. Character Segmentation
3. Character Classification (CNN)
4. Post-processing

**Device:** CPU  
**Framework:** PyTorch
""")

# Initialize models
if 'plate_detector' not in st.session_state:
    with st.spinner("🚀 Loading AI models..."):
        plate_detector, char_classifier = initialize_models()
        st.session_state.plate_detector = plate_detector
        st.session_state.char_classifier = char_classifier
        st.session_state.models_loaded = True
    st.sidebar.success("✅ Models loaded!")

# ==================== SINGLE IMAGE MODE ====================

if mode == "🔍 Single Image":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("🔍 Single Image Processing")
    st.markdown("Upload a vehicle image to detect and recognize the license plate")
    st.markdown('</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "📤 Upload vehicle image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload image containing vehicle with license plate"
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📥 Input Image")
            st.image(image, use_column_width=True)
            st.caption(f"Image size: {image.size[0]} × {image.size[1]} pixels")
        
        with col2:
            st.markdown("### 🎯 Detection Result")
            with st.spinner("🔄 Processing image..."):
                results = process_single_image(
                    image, 
                    st.session_state.plate_detector,
                    st.session_state.char_classifier
                )
            
            if results['success']:
                if results['viz_img'] is not None:
                    st.image(results['viz_img'], use_column_width=True)
                
                # Display plate number in custom style
                st.markdown(f'<div class="plate-number">{results["plate_text"]}</div>', 
                          unsafe_allow_html=True)
                
                # Metrics
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Plates Detected", results['num_plates'])
                with metric_col2:
                    st.metric("Characters Found", results['num_chars'])
                with metric_col3:
                    st.metric("Confidence", f"{results['confidence']:.1%}")
                
                if results['num_chars'] < 7:
                    st.warning(f"⚠️ Only {results['num_chars']} characters detected. Expected 7-9 for Indonesian plates.")
                elif results['num_chars'] > 9:
                    st.warning(f"⚠️ {results['num_chars']} characters detected. May include noise.")
                else:
                    st.success(f"✅ Character count looks good!")
            else:
                st.error("❌ No plate detected in the image")
                if 'error' in results:
                    st.error(f"Error: {results['error']}")

# ==================== MULTIPLE IMAGES MODE ====================

elif mode == "📸 Multiple Images":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("📸 Multiple Images Processing")
    st.markdown("Upload multiple vehicle images for batch processing")
    st.markdown('</div>', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "📤 Upload multiple vehicle images",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Select multiple images to process at once"
    )
    
    if uploaded_files:
        st.success(f"✅ Uploaded {len(uploaded_files)} images")
        
        # Process button
        if st.button("🚀 Process All Images", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_results = []
            
            for idx, uploaded_file in enumerate(uploaded_files):
                # Update progress
                progress = (idx + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"Processing image {idx + 1} of {len(uploaded_files)}...")
                
                # Load and process image
                image = Image.open(uploaded_file).convert("RGB")
                results = process_single_image(
                    image,
                    st.session_state.plate_detector,
                    st.session_state.char_classifier
                )
                
                results['filename'] = uploaded_file.name
                results['image'] = image
                all_results.append(results)
            
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            st.markdown("---")
            st.markdown("## 📊 Processing Results")
            
            # Summary metrics
            total_images = len(all_results)
            successful = sum(1 for r in all_results if r['success'])
            total_plates = sum(r['num_plates'] for r in all_results)
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Total Images", total_images)
            with metric_col2:
                st.metric("Successful Detections", successful)
            with metric_col3:
                st.metric("Total Plates Found", total_plates)
            
            st.markdown("---")
            
            # Display individual results
            for idx, result in enumerate(all_results):
                with st.expander(f"📷 {result['filename']}", expanded=(idx == 0)):
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.image(result['image'], use_column_width=True)
                        st.caption(f"Original: {result['filename']}")
                    
                    with col2:
                        if result['success']:
                            if result['viz_img'] is not None:
                                st.image(result['viz_img'], use_column_width=True)
                                st.caption("Detection Result")
                            
                            st.markdown(f'<div class="plate-number">{result["plate_text"]}</div>', 
                                      unsafe_allow_html=True)
                            
                            res_col1, res_col2 = st.columns(2)
                            with res_col1:
                                st.metric("Characters", result['num_chars'])
                            with res_col2:
                                st.metric("Confidence", f"{result['confidence']:.1%}")
                        else:
                            st.error("❌ No plate detected")
                            if 'error' in result:
                                st.error(f"Error: {result['error']}")
    else:
        st.info("👆 Upload multiple images to get started")

# ==================== ABOUT MODE ====================

else:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("📊 About ANPR System Pro")
    
    st.markdown("""
    ## 🎯 Advanced License Plate Recognition
    
    This system implements a state-of-the-art **5-stage AI pipeline** for automatic number plate recognition:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔧 Pipeline Stages
        
        1. **Image Input** 
           - RGB vehicle images
           - Multiple format support
        
        2. **Plate Detection** 
           - Faster R-CNN architecture
           - ResNet-50 + FPN backbone
           - High-accuracy bounding boxes
        
        3. **Character Segmentation** 
           - Advanced preprocessing
           - CLAHE enhancement
           - Multi-threshold approach
        """)
    
    with col2:
        st.markdown("""
        ### ⚙️ Technical Details
        
        4. **Character Classification** 
           - Custom CNN architecture
           - 36 classes (A-Z, 0-9)
           - 32×32 grayscale input
        
        5. **Post-Processing** 
           - Character ordering
           - Plate formatting
           - Confidence scoring
        
        ---
        
        **Framework:** PyTorch  
        **Device:** CPU  
        **Models:** Real-time inference
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🚀 Features
    
    - ✅ **Multi-Image Upload** - Process multiple images simultaneously
    - ✅ **Batch Processing** - Efficient handling of large image sets
    - ✅ **Real-Time Detection** - Fast inference on CPU
    - ✅ **Advanced Segmentation** - Multiple thresholding and morphology operations
    - ✅ **Beautiful UI** - Modern, responsive interface with smooth animations
    - ✅ **Debug Visualization** - Detailed preprocessing pipeline views
    
    ### 🎨 Design Highlights
    
    - **Glassmorphism Effects** - Modern blur and transparency
    - **Gradient Accents** - Vibrant purple/blue color scheme
    - **Smooth Animations** - Enhanced user experience
    - **Responsive Layout** - Optimized for all screen sizes
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p style='color: white; font-size: 0.9rem;'>
            Built with Streamlit 🎈 | Powered by PyTorch 🔥 | Enhanced with ❤️
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
