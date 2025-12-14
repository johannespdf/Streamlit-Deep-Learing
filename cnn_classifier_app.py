import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Konfigurasi halaman
st.set_page_config(
    page_title="CNN Image Classifier",
    page_icon="🖼️",
    layout="wide"
)

# ==================== MODEL CNN ====================
class CNNClassifier(nn.Module):
    """
    Custom CNN Model untuk klasifikasi gambar
    Arsitektur:
    - 3 Convolutional Blocks (Conv → ReLU → MaxPool)
    - 2 Fully Connected Layers
    - Softmax untuk output probabilitas
    """
    def __init__(self, num_classes=10):
        super(CNNClassifier, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Conv blocks
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# ==================== KONFIGURASI ====================
CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

NUM_CLASSES = len(CLASSES)
DEVICE = torch.device('cpu')

# ==================== FUNGSI HELPER ====================
@st.cache_resource
def initialize_model():
    """Inisialisasi model CNN"""
    model = CNNClassifier(num_classes=NUM_CLASSES)
    model.to(DEVICE)
    return model

def preprocess_image(image):
    """
    Preprocessing gambar untuk inference
    
    Args:
        image: PIL Image
    
    Returns:
        tensor: preprocessed image tensor
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def predict(model, image_tensor):
    """
    Jalankan inference
    
    Args:
        model: CNN model
        image_tensor: preprocessed image
    
    Returns:
        predicted_class: index kelas yang diprediksi
        confidence: confidence score
        probabilities: semua probabilitas kelas
    """
    model.eval()
    
    with torch.no_grad():
        outputs = model(image_tensor.to(DEVICE))
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    
    return predicted_class.item(), confidence.item(), probabilities[0].cpu().numpy()

def train_model(model, num_epochs=10, batch_size=64, learning_rate=0.001):
    """
    Training model dengan CIFAR-10 dataset
    
    Args:
        model: CNN model
        num_epochs: jumlah epoch
        batch_size: ukuran batch
        learning_rate: learning rate
    
    Returns:
        model: trained model
        history: training history
    """
    # Transform untuk training
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform_train
    )
    
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=0
    )
    
    # Loss dan optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    history = {'loss': [], 'accuracy': []}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress
            if i % 100 == 99:
                progress = (epoch * len(trainloader) + i + 1) / (num_epochs * len(trainloader))
                progress_bar.progress(progress)
                status_text.text(
                    f'Epoch [{epoch+1}/{num_epochs}], '
                    f'Step [{i+1}/{len(trainloader)}], '
                    f'Loss: {running_loss/100:.3f}, '
                    f'Acc: {100.*correct/total:.2f}%'
                )
                running_loss = 0.0
        
        # Epoch statistics
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100. * correct / total
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        
        st.write(f"**Epoch {epoch+1}/{num_epochs}** - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    
    progress_bar.empty()
    status_text.empty()
    
    return model, history

def plot_probabilities(probabilities, classes):
    """Plot bar chart untuk probabilitas kelas"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(classes))
    ax.barh(y_pos, probabilities * 100, color='skyblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes)
    ax.invert_yaxis()
    ax.set_xlabel('Confidence (%)')
    ax.set_title('Class Probabilities')
    ax.set_xlim(0, 100)
    
    # Add percentage labels
    for i, v in enumerate(probabilities * 100):
        ax.text(v + 1, i, f'{v:.1f}%', va='center')
    
    plt.tight_layout()
    return fig

# ==================== SIDEBAR ====================
st.sidebar.header("⚙️ Configuration")

# Mode selection
mode = st.sidebar.radio(
    "Select Mode",
    ["🔍 Inference", "🎓 Training"],
    help="Choose between inference (prediction) or training mode"
)

if mode == "🎓 Training":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Training Parameters")
    
    num_epochs = st.sidebar.slider("Number of Epochs", 1, 50, 10)
    batch_size = st.sidebar.selectbox("Batch Size", [32, 64, 128], index=1)
    learning_rate = st.sidebar.select_slider(
        "Learning Rate",
        options=[0.0001, 0.001, 0.01, 0.1],
        value=0.001
    )

# Info
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **CNN Classifier**
    - Custom PyTorch CNN
    - No pre-trained weights
    - CPU Inference
    - 10 Classes (CIFAR-10)
    """
)

# ==================== MAIN APP ====================
st.title("🖼️ CNN Image Classifier")
st.markdown("**Custom CNN Model untuk Klasifikasi Gambar**")
st.markdown("---")

# Initialize model
if 'model' not in st.session_state:
    st.session_state.model = initialize_model()
    st.session_state.trained = False

# ==================== TRAINING MODE ====================
if mode == "🎓 Training":
    st.header("🎓 Training Mode")
    
    st.write("""
    Train the CNN model using CIFAR-10 dataset. 
    This will download the dataset (~170MB) on first run.
    """)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.write("**Training Settings:**")
        st.write(f"- Epochs: {num_epochs}")
        st.write(f"- Batch Size: {batch_size}")
        st.write(f"- Learning Rate: {learning_rate}")
        st.write(f"- Device: CPU")
    
    with col2:
        if st.button("🚀 Start Training", type="primary"):
            st.info("⏳ Training started... This may take a while on CPU.")
            
            # Training
            trained_model, history = train_model(
                st.session_state.model,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            
            st.session_state.model = trained_model
            st.session_state.trained = True
            
            st.success("✅ Training completed!")
            
            # Plot training history
            if history['loss']:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                ax1.plot(history['loss'])
                ax1.set_title('Training Loss')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.grid(True)
                
                ax2.plot(history['accuracy'])
                ax2.set_title('Training Accuracy')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy (%)')
                ax2.grid(True)
                
                st.pyplot(fig)

# ==================== INFERENCE MODE ====================
else:
    st.header("🔍 Inference Mode")
    
    if not st.session_state.trained:
        st.warning(
            "⚠️ Model has random weights (not trained). "
            "Predictions will be random. Switch to Training mode to train the model first."
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image for classification"
        )
        
        if uploaded_file is not None:
            # Load image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.caption(f"Image size: {image.size[0]} x {image.size[1]}")
    
    with col2:
        st.subheader("🎯 Prediction Result")
        
        if uploaded_file is not None:
            # Preprocess
            image_tensor = preprocess_image(image)
            
            # Predict
            with st.spinner("Running inference..."):
                predicted_class, confidence, probabilities = predict(
                    st.session_state.model,
                    image_tensor
                )
            
            # Display results
            st.success(f"**Predicted Class:** {CLASSES[predicted_class]}")
            st.metric("Confidence Score", f"{confidence * 100:.2f}%")
            
            # Show top 3 predictions
            st.markdown("### Top 3 Predictions")
            top3_indices = np.argsort(probabilities)[::-1][:3]
            
            for i, idx in enumerate(top3_indices):
                st.write(
                    f"{i+1}. **{CLASSES[idx]}** - "
                    f"{probabilities[idx] * 100:.2f}%"
                )
            
            # Plot probabilities
            st.markdown("### All Class Probabilities")
            fig = plot_probabilities(probabilities, CLASSES)
            st.pyplot(fig)
        
        else:
            st.info("👆 Upload an image to start classification")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with Streamlit 🎈 | Custom CNN with PyTorch 🔥</p>
    </div>
    """,
    unsafe_allow_html=True
)
