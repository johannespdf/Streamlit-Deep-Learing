import streamlit as st
import easyocr
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from io import BytesIO
import os

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="ANPR System Pro - OCR",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
def load_custom_css():
    st.markdown("""
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Remove default top padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e9ecef;
    }
    
    /* Main area background */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Headers */
    h1 {
        color: #1a1a1a;
        font-weight: 700;
        font-size: 2.2rem !important;
        margin-bottom: 0.5rem;
    }
    
    h2, h3 {
        color: #2c3e50;
        font-weight: 600;
    }
    
    /* Primary Button Styling */
    .stButton > button {
        background-color: #0066cc;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.2s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #0052a3;
        box-shadow: 0 4px 12px rgba(0,102,204,0.2);
    }
    
    /* Secondary Button / File Uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #e0e0e0;
        border-radius: 10px;
        padding: 2rem;
        background-color: #fafafa;
        transition: border-color 0.3s;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #0066cc;
        background-color: #f0f7ff;
    }
    
    /* Cards/Containers */
    .result-container {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        background-color: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #0066cc !important;
        font-weight: 700;
        font-size: 1.8rem;
    }
    
    [data-testid="stMetricLabel"] {
        color: #666;
        font-size: 0.9rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 6px;
    }
    
    /* Plate Text Display */
    .plate-display {
        background-color: #1a1a1a;
        color: #ffffff;
        font-family: 'Consolas', monospace;
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        padding: 1rem;
        border-radius: 8px;
        border: 4px solid #333;
        letter-spacing: 2px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    /* Custom Badge */
    .status-badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        font-weight: 600;
    }
    
    .status-success {
        background-color: #d4edda;
        color: #155724;
    }
    
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== OCR FUNCTIONS ====================

@st.cache_resource
def initialize_ocr():
    """Initialize EasyOCR reader with English language"""
    try:
        # Initialize with English only for faster loading
        reader = easyocr.Reader(['en'], gpu=False)
        return reader
    except Exception as e:
        st.error(f"Error initializing OCR: {e}")
        return None

# Allowlist for Indonesian license plate characters
PLATE_ALLOWLIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

def preprocess_plate_image(image, method='auto'):
    """
    Preprocess plate image for better OCR accuracy on Indonesian plates
    Handles both white plates (black text) and black plates (white text)
    
    Args:
        image: PIL Image of the plate
        method: 'auto', 'white_plate', 'black_plate', or 'original'
        
    Returns:
        preprocessed: numpy array ready for OCR
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array.copy()
    
    # Resize to larger size for better OCR
    scale_factor = 3
    height, width = gray.shape
    if height < 100:  # Only upscale small images
        gray = cv2.resize(gray, (width * scale_factor, height * scale_factor), 
                         interpolation=cv2.INTER_CUBIC)
    
    # Determine if it's a dark or light plate based on mean value
    mean_val = np.mean(gray)
    
    if method == 'auto':
        # Auto-detect plate type
        if mean_val < 127:
            method = 'black_plate'
        else:
            method = 'white_plate'
    
    if method == 'original':
        # Just return grayscale without heavy processing
        return gray
    
    # Light denoise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    if method == 'black_plate':
        # Black background, white text - invert to get black text on white
        gray = cv2.bitwise_not(gray)
    
    # Simple Otsu thresholding works best
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Light morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary


def preprocess_multiple_methods(image):
    """
    Generate multiple preprocessed versions for OCR attempts
    
    Args:
        image: PIL Image
        
    Returns:
        list of (name, preprocessed_image) tuples
    """
    results = []
    
    # Method 1: Original grayscale (sometimes works best)
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array.copy()
    
    # Resize if small
    height, width = gray.shape
    if height < 100:
        scale = 3
        gray = cv2.resize(gray, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
    
    results.append(('original', gray))
    
    # Method 2: Auto-detect and preprocess
    results.append(('auto', preprocess_plate_image(image, 'auto')))
    
    # Method 3: Force white plate processing
    results.append(('white_plate', preprocess_plate_image(image, 'white_plate')))
    
    # Method 4: Force black plate processing
    results.append(('black_plate', preprocess_plate_image(image, 'black_plate')))
    
    return results

def detect_plate_region(image):
    """
    Simple plate detection using contours
    
    Args:
        image: PIL Image
        
    Returns:
        plate_img: cropped plate region or None
        box: bounding box coordinates
    """
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 100, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    plate_contour = None
    
    # Find rectangular contour (plate-like)
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
        
        if len(approx) == 4:  # Rectangle
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            
            # Indonesian plates are typically wider than tall
            if 2.0 <= aspect_ratio <= 5.0:
                plate_contour = approx
                break
    
    if plate_contour is not None:
        x, y, w, h = cv2.boundingRect(plate_contour)
        plate_img = image.crop((x, y, x + w, y + h))
        return plate_img, (x, y, x + w, y + h)
    
    # If no plate detected, return full image
    return image, None

def read_plate_text(reader, image, skip_detection=False):
    """
    Read text from plate image using EasyOCR with multiple attempts
    
    Args:
        reader: EasyOCR reader instance
        image: PIL Image or numpy array
        skip_detection: if True, treat image as already cropped plate
        
    Returns:
        text: detected text
        confidence: average confidence
        details: all detection details
        debug_info: debugging information
    """
    best_text = ""
    best_confidence = 0.0
    best_details = []
    debug_info = {}
    
    # Try multiple preprocessing methods
    methods = preprocess_multiple_methods(image)
    
    for method_name, preprocessed in methods:
        try:
            # Run OCR with allowlist for license plate characters only
            results = reader.readtext(
                preprocessed,
                allowlist=PLATE_ALLOWLIST,
                paragraph=False,
                contrast_ths=0.1,
                adjust_contrast=0.5,
                text_threshold=0.5,
                low_text=0.3,
                width_ths=0.7
            )
             
            if not results:
                continue
            
            # --- HEURISTIC FILTERING LOGIN ---
            valid_items = []
            
            # 1. Calculate height for each detection
            # bbox structure: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            items_with_metrics = []
            max_height = 0
            
            for (bbox, text, conf) in results:
                # Clean up text (keep only alphanumeric)
                cleaned_text = ''.join(c for c in text.upper() if c.isalnum())
                
                if cleaned_text:
                    # Calculate height: average of left and right vertical edges
                    h1 = bbox[3][1] - bbox[0][1]
                    h2 = bbox[2][1] - bbox[1][1]
                    height = (h1 + h2) / 2
                    
                    # Calculate center X for sorting
                    center_x = (bbox[0][0] + bbox[2][0]) / 2
                    
                    if height > max_height:
                        max_height = height
                        
                    items_with_metrics.append({
                        'text': cleaned_text,
                        'conf': conf,
                        'height': height,
                        'center_x': center_x,
                        'bbox': bbox
                    })
            
            # 2. Filter by height (remove small text like dates)
            # Threshold: Must be at least 60% of the tallest character
            HEIGHT_RATIO_THS = 0.6
            
            filtered_items = []
            for item in items_with_metrics:
                if item['height'] >= max_height * HEIGHT_RATIO_THS:
                    filtered_items.append(item)
            
            # 3. Sort left-to-right
            filtered_items.sort(key=lambda x: x['center_x'])
            
            # 4. Construct final text
            texts = [item['text'] for item in filtered_items]
            confidences = [item['conf'] for item in filtered_items]
            
            full_text = ''.join(texts)
            avg_confidence = np.mean(confidences) if confidences else 0.0
             
            # Store debug info
            debug_info[method_name] = {
                'text': full_text,
                'confidence': avg_confidence,
                'raw_results': len(results),
                'filtered_results': len(filtered_items)
            }
             
            # Check if this is better than previous results
            # Prefer results with reasonable length (7-10 chars for Indonesian plates)
            is_valid_length = 7 <= len(full_text) <= 12
            prev_valid_length = 7 <= len(best_text) <= 12
            
            # Logic to choose best result:
            # 1. If valid length, prefer high confidence
            # 2. If no valid length found yet, take best confidence
            # 3. Adjust threshold: valid length result beats invalid length result even if confidence is slightly lower (but reasonable)
            
            if is_valid_length:
                if not prev_valid_length:
                    # Found first valid length result, take it
                    best_text = full_text
                    best_confidence = avg_confidence
                    best_details = results # Keep original results for bbox visualization
                elif avg_confidence > best_confidence:
                    # Found better valid length result
                    best_text = full_text
                    best_confidence = avg_confidence
                    best_details = results
            elif not prev_valid_length:
                # Still no valid length, just maximize confidence
                if avg_confidence > best_confidence:
                    best_text = full_text
                    best_confidence = avg_confidence
                    best_details = results
                 
        except Exception as e:
            debug_info[f'{method_name}_error'] = str(e)
     
    return best_text, best_confidence, best_details, debug_info

def format_indonesian_plate(text):
    """
    Format text to match Indonesian plate pattern (XX 1234 XX)
    
    Args:
        text: raw OCR text
        
    Returns:
        formatted: formatted plate number
    """
    # Remove all spaces and special characters
    clean = ''.join(c for c in text if c.isalnum()).upper()
    
    # Try to match Indonesian pattern
    if len(clean) >= 7:
        # XX 1234 XX pattern
        return f"{clean[:2]} {clean[2:6]} {clean[6:]}"
    elif len(clean) >= 5:
        # Shorter plates
        return f"{clean[:2]} {clean[2:]}"
    else:
        return clean

def visualize_detection(image, box=None):
    """Draw bounding box on image"""
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    if box:
        x1, y1, x2, y2 = box
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=5)
        
        try:
            font = ImageFont.truetype("arial.ttf", 30)
        except:
            font = ImageFont.load_default()
        
        draw.text((x1, y1 - 40), "Plate Detected", fill="red", font=font)
    
    return img_draw

def process_single_image(image, reader, skip_plate_detection=False):
    """
    Process a single image through the ANPR pipeline
    
    Args:
        image: PIL Image
        reader: EasyOCR reader
        skip_plate_detection: if True, treat image as already cropped plate
    
    Returns:
        dict with results
    """
    results = {
        'success': False,
        'plate_text': '',
        'formatted_text': '',
        'confidence': 0.0,
        'plate_img': None,
        'viz_img': None,
        'details': [],
        'debug_info': {}
    }
    
    try:
        if skip_plate_detection:
            # Image is already a cropped plate
            plate_img = image
            box = None
            results['plate_img'] = plate_img
            results['viz_img'] = image
        else:
            # Step 1: Detect plate region
            plate_img, box = detect_plate_region(image)
            results['plate_img'] = plate_img
            
            if box:
                results['viz_img'] = visualize_detection(image, box)
            else:
                results['viz_img'] = image
        
        # Step 2: Read text using OCR with multiple methods
        text, confidence, details, debug_info = read_plate_text(reader, plate_img)
        
        results['plate_text'] = text
        results['formatted_text'] = format_indonesian_plate(text)
        results['confidence'] = confidence
        results['details'] = details
        results['debug_info'] = debug_info
        results['success'] = bool(text)
        
    except Exception as e:
        results['error'] = str(e)
    
    return results

# ==================== PROFESSIONAL DASHBOARD UI ====================
 
# Apply professional CSS
load_custom_css()
 
# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.markdown("## 🚗 ANPR Pro")
    st.caption("Advanced Number Plate Recognition")
    st.markdown("---")
    
    st.markdown("### ⚙️ Konfigurasi")
    
    # Input Source Selection
    input_source = st.radio(
        "Sumber Input",
        ["Upload Gambar", "Gunakan Kamera"],
        captions=["Upload file lokal", "Ambil foto langsung"]
    )
    
    st.markdown("---")
    
    # Confidence Threshold Slider
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.05,
        help="Minimum keyakinan untuk menampilkan hasil"
    )
    
    # Skip Detection Checkbox (Moved to sidebar for cleaner main UI)
    skip_detection = st.checkbox(
        "Mode Crop Otomatis",
        value=True,
        help="Centang jika input sudah berupa gambar plat yang di-crop"
    )
    
    st.markdown("---")
    st.info(
        """
        **Petunjuk:**
        1. Pilih sumber input
        2. Masukkan gambar
        3. Klik 'Deteksi Plat'
        """
    )
    
    # Status Indicators
    if 'ocr_loaded' in st.session_state and st.session_state.ocr_loaded:
        st.success("✅ System Ready")
    else:
        st.error("❌ System Offline")
 
# --- MAIN CONTENT AREA ---
 
# Header
st.title("Sistem Pengenalan Plat Nomor")
st.markdown("Dashboard deteksi dan pengenalan karakter plat nomor kendaraan Indonesia")
 
# Input Section Container
input_container = st.container()
input_images = [] # List of (name, image_object)

with input_container:
    if input_source == "Upload Gambar":
        uploaded_files = st.file_uploader(
            "Upload Gambar Kendaraan (Bisa banyak sekaligus)",
            type=['jpg', 'jpeg', 'png'],
            help="Drag & drop satu atau banyak gambar di sini",
            accept_multiple_files=True
        )
        if uploaded_files:
            for up_file in uploaded_files:
                input_images.append((up_file.name, Image.open(up_file).convert("RGB")))
            
    elif input_source == "Gunakan Kamera":
        st.info("💡 **Info:** Jika kamera tidak muncul, pastikan Anda telah klik **'Allow'** pada pop-up izin browser (biasanya di pojok kiri atas dekat URL).")
        camera_file = st.camera_input("Ambil Foto Kendaraan")
        if camera_file:
            input_images.append(("Foto Kamera", Image.open(camera_file).convert("RGB")))

# Processing Section
if input_images:
    count = len(input_images)
    label = f"🔍 Deteksi {count} Plat Nomor" if count > 1 else "🔍 Deteksi Plat Nomor"
    
    # Process Button
    start_process = st.button(label, type="primary")
    
    if start_process:
        if 'ocr_loaded' not in st.session_state or not st.session_state.ocr_loaded:
            st.error("Sistem OCR belum siap. Silakan refresh halaman.")
        else:
            # Container for all results
            results_container = st.container()
            
            with results_container:
                progress_bar = st.progress(0)
                
                for idx, (img_name, image) in enumerate(input_images):
                    st.markdown(f"### 📄 Memproses: {img_name}")
                    
                    with st.spinner(f"Sedang memproses {img_name}..."):
                        results = process_single_image(
                            image, 
                            st.session_state.ocr_reader,
                            skip_plate_detection=skip_detection
                        )
                    
                    # Update progress
                    progress_bar.progress((idx + 1) / count)
                    
                    # --- RESULT LAYOUT FOR EACH IMAGE ---
                    if results['success']:
                         # Success Badge
                        st.markdown(
                            f"""
                            <div style="background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                                <strong>✅ {img_name}: Berhasil!</strong> (Keyakinan: {results['confidence']:.1%})
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        
                        # 2-Column Layout
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.image(image, use_column_width=True, caption=f"Input: {img_name}")
                            
                        with col2:
                            if results['viz_img']:
                                st.image(results['viz_img'], use_column_width=True, caption="Visualisasi")
                            elif results['plate_img']:
                                 st.image(results['plate_img'], use_column_width=True, caption="Plat")
                        
                        # Metrics Section
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        with metric_col1:
                            st.metric("Plat Nomor", results['formatted_text'])
                        with metric_col2:
                            st.metric("Raw Text", results['plate_text'])
                        with metric_col3:
                            st.metric("Akurasi", f"{results['confidence']:.1%}")
                        
                        # Big Display
                        st.markdown(f'<div class="plate-display">{results["formatted_text"]}</div>', unsafe_allow_html=True)
                        
                        # Debug info in expander
                        with st.expander(f"🛠️ Detail Teknis ({img_name})"):
                            st.json({
                                "filename": img_name,
                                "processing_method": "EasyOCR",
                                "raw_output": results.get('details', []),
                                "debug_info": results.get('debug_info', {})
                            })
                            
                    else:
                        st.error(f"❌ {img_name}: Gagal mendeteksi teks.")
                        st.image(image, caption="Gambar Input", width=300)
                    
                    st.markdown("---") # Separator between images
                
                st.success(f"✅ Selesai memproses {count} gambar!")
    
else:
    # Placeholder / Empty State
    st.info("👆 Silakan upload satu atau lebih gambar untuk memulai.")
 
# Initialize OCR in background (if not loaded)
if 'ocr_reader' not in st.session_state:
    with st.spinner("Inisialisasi Sistem..."):
        ocr_reader = initialize_ocr()
        st.session_state.ocr_reader = ocr_reader
        st.session_state.ocr_loaded = True if ocr_reader else False
        if st.session_state.ocr_loaded:
            st.rerun() # Rerun to update sidebar status
