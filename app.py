import streamlit as st
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

# Konfigurasi halaman
st.set_page_config(
    page_title="License Plate Detection",
    page_icon="🚗",
    layout="wide"
)

# Fungsi untuk load model dengan caching
@st.cache_resource
def load_model(model_path, num_classes=37):
    """
    Load model Faster R-CNN yang sudah dilatih
    
    Args:
        model_path: path ke file .pth
        num_classes: jumlah kelas (36 karakter + 1 background)
    
    Returns:
        model: model yang sudah di-load
    """
    try:
        # Rekonstruksi arsitektur model
        model = fasterrcnn_resnet50_fpn(weights=None)
        
        # Ganti predictor head sesuai num_classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Load weights (gunakan CPU)
        device = torch.device('cpu')
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle berbagai format checkpoint
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
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """
    Preprocessing gambar sesuai dengan training
    
    Args:
        image: PIL Image
    
    Returns:
        tensor: torch tensor
    """
    transform = T.Compose([
        T.ToTensor(),
    ])
    return transform(image)

def detect_plates(model, image, confidence_threshold=0.5):
    """
    Deteksi plat nomor dari gambar
    
    Args:
        model: model Faster R-CNN
        image: PIL Image
        confidence_threshold: threshold confidence untuk filtering
    
    Returns:
        boxes: list of bounding boxes
        labels: list of labels
        scores: list of confidence scores
    """
    device = torch.device('cpu')
    
    # Preprocess
    image_tensor = preprocess_image(image).to(device)
    
    # Inference
    with torch.no_grad():
        predictions = model([image_tensor])
    
    # Extract hasil
    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    
    # Filter berdasarkan confidence
    mask = scores >= confidence_threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]
    
    return boxes, labels, scores

def draw_boxes(image, boxes, labels, scores):
    """
    Gambar bounding boxes pada gambar
    
    Args:
        image: PIL Image
        boxes: numpy array of boxes
        labels: numpy array of labels
        scores: numpy array of scores
    
    Returns:
        image: PIL Image dengan bounding boxes
    """
    draw = ImageDraw.Draw(image)
    
    # Mapping label ke karakter (sesuaikan dengan dataset Anda)
    # 0 = background, 1-10 = 0-9, 11-36 = A-Z
    def label_to_char(label):
        if label == 0:
            return "BG"
        elif 1 <= label <= 10:
            return str(label - 1)
        elif 11 <= label <= 36:
            return chr(ord('A') + label - 11)
        else:
            return str(label)
    
    # Font untuk text (fallback ke default jika tidak ada)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Gambar setiap box
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        
        # Gambar rectangle
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
        
        # Text label
        text = f"{label_to_char(label)}: {score:.2f}"
        
        # Background untuk text
        text_bbox = draw.textbbox((x1, y1 - 25), text, font=font)
        draw.rectangle(text_bbox, fill="red")
        
        # Draw text
        draw.text((x1, y1 - 25), text, fill="white", font=font)
    
    return image

def main():
    st.title("🚗 License Plate Character Detection")
    st.markdown("---")
    
    # Sidebar untuk konfigurasi
    st.sidebar.header("⚙️ Configuration")
    
    # Upload model file
    model_file = st.sidebar.file_uploader(
        "Upload Model (.pth)",
        type=['pth', 'pt'],
        help="Upload file model PyTorch yang sudah dilatih"
    )
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Threshold untuk filtering deteksi"
    )
    
    # Number of classes
    num_classes = st.sidebar.number_input(
        "Number of Classes",
        min_value=2,
        max_value=100,
        value=37,
        help="Jumlah kelas (karakter + background)"
    )
    
    # Info
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **How to use:**
        1. Upload model file (.pth)
        2. Upload gambar plat nomor
        3. Adjust confidence threshold
        4. Lihat hasil deteksi
        """
    )
    
    # Main area
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload gambar plat nomor kendaraan"
        )
        
        if uploaded_file is not None:
            # Load dan tampilkan gambar asli
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Original Image", use_column_width=True)
            
            # Info gambar
            st.caption(f"Image size: {image.size[0]} x {image.size[1]}")
    
    with col2:
        st.subheader("🔍 Detection Result")
        
        if uploaded_file is not None and model_file is not None:
            # Save model file temporarily
            with open("/tmp/model.pth", "wb") as f:
                f.write(model_file.getbuffer())
            
            # Load model
            with st.spinner("Loading model..."):
                model = load_model("/tmp/model.pth", num_classes)
            
            if model is not None:
                # Detect
                with st.spinner("Detecting..."):
                    boxes, labels, scores = detect_plates(
                        model, 
                        image, 
                        confidence_threshold
                    )
                
                # Draw boxes
                result_image = image.copy()
                result_image = draw_boxes(result_image, boxes, labels, scores)
                
                # Tampilkan hasil
                st.image(result_image, caption="Detection Result", use_column_width=True)
                
                # Info deteksi
                st.success(f"✅ Found {len(boxes)} object(s)")
                
                # Detail deteksi
                if len(boxes) > 0:
                    st.markdown("### Detection Details")
                    
                    # Mapping label ke karakter
                    def label_to_char(label):
                        if label == 0:
                            return "BG"
                        elif 1 <= label <= 10:
                            return str(label - 1)
                        elif 11 <= label <= 36:
                            return chr(ord('A') + label - 11)
                        else:
                            return str(label)
                    
                    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
                        with st.expander(f"Object {i+1}: {label_to_char(label)} ({score:.2%})"):
                            st.write(f"**Label:** {label_to_char(label)}")
                            st.write(f"**Confidence:** {score:.2%}")
                            st.write(f"**Bounding Box:** [{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]")
                else:
                    st.warning("No objects detected. Try lowering the confidence threshold.")
        
        elif uploaded_file is not None:
            st.warning("⚠️ Please upload model file first")
        else:
            st.info("👆 Upload an image to start detection")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with Streamlit 🎈 | Powered by PyTorch 🔥</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
