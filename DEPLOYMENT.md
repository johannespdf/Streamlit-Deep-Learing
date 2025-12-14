# Deployment Guide - Streamlit Cloud

## 🚀 Deploy ke Streamlit Cloud (GRATIS)

### Prerequisites
1. Akun GitHub
2. Akun Streamlit Cloud (https://streamlit.io/cloud)
3. Repository GitHub untuk aplikasi ini

### Step 1: Persiapan Repository

1. **Buat repository baru di GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/username/plate-detection.git
   git push -u origin main
   ```

2. **Pastikan file-file penting ada:**
   - ✅ `app.py`
   - ✅ `requirements.txt`
   - ✅ `README.md`
   - ✅ `.gitignore`

### Step 2: Upload Model

**PENTING**: File .pth biasanya besar (>100MB). Streamlit Cloud memiliki batasan ukuran file.

**Solusi 1: GitHub LFS (untuk file <500MB)**
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
git add model.pth
git commit -m "Add model with LFS"
git push
```

**Solusi 2: Google Drive (RECOMMENDED)**

1. Upload model.pth ke Google Drive
2. Set sharing ke "Anyone with the link"
3. Get direct download link:
   - Share link: `https://drive.google.com/file/d/FILE_ID/view`
   - Direct link: `https://drive.google.com/uc?id=FILE_ID`

4. Modifikasi `app.py` untuk download model:

```python
import gdown

@st.cache_resource
def download_and_load_model():
    model_url = "https://drive.google.com/uc?id=YOUR_FILE_ID"
    model_path = "model.pth"
    
    if not os.path.exists(model_path):
        with st.spinner("Downloading model..."):
            gdown.download(model_url, model_path, quiet=False)
    
    return load_model(model_path)
```

Tambahkan ke `requirements.txt`:
```
gdown==4.7.1
```

**Solusi 3: Hugging Face Hub (RECOMMENDED untuk model besar)**

1. Upload model ke Hugging Face:
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="model.pth",
    path_in_repo="model.pth",
    repo_id="username/plate-detection",
    repo_type="model",
)
```

2. Download di app:
```python
from huggingface_hub import hf_hub_download

@st.cache_resource
def load_model_from_hf():
    model_path = hf_hub_download(
        repo_id="username/plate-detection",
        filename="model.pth"
    )
    return load_model(model_path)
```

Tambahkan ke `requirements.txt`:
```
huggingface-hub==0.20.0
```

### Step 3: Deploy ke Streamlit Cloud

1. **Login ke Streamlit Cloud**
   - Buka https://share.streamlit.io/
   - Sign in dengan GitHub

2. **Create New App**
   - Klik "New app"
   - Pilih repository Anda
   - Main file path: `app.py`
   - Klik "Deploy"

3. **Wait for deployment**
   - Streamlit akan install dependencies
   - Biasanya 2-5 menit
   - Jika error, cek logs

### Step 4: Configure Secrets (Optional)

Untuk API keys atau credentials:

1. Di Streamlit Cloud dashboard, buka app settings
2. Klik "Secrets"
3. Tambahkan secrets dalam format TOML:

```toml
[gdrive]
file_id = "YOUR_GOOGLE_DRIVE_FILE_ID"

[huggingface]
token = "YOUR_HF_TOKEN"
```

4. Akses di code:
```python
import streamlit as st

file_id = st.secrets["gdrive"]["file_id"]
```

## 🐳 Alternative: Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY app.py .
COPY model.pth .

# Expose port
EXPOSE 8501

# Run streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build dan Run

```bash
# Build
docker build -t plate-detection .

# Run
docker run -p 8501:8501 plate-detection
```

### Deploy ke Cloud Run / Heroku / Railway

Ikuti dokumentasi masing-masing platform untuk deploy Docker container.

## 📊 Optimasi Performa

### 1. Model Optimization

**Quantization** (reduce model size):
```python
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
torch.save(model_quantized.state_dict(), 'model_quantized.pth')
```

**ONNX Export** (faster inference):
```python
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "model.onnx")
```

### 2. Caching Strategy

```python
@st.cache_data(ttl=3600)  # Cache 1 hour
def process_image(image_bytes):
    # Processing logic
    return result
```

### 3. Resource Limits

Tambahkan di `.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 200
maxMessageSize = 200

[runner]
magicEnabled = false
```

## 🔍 Monitoring & Debugging

### Streamlit Cloud Logs

1. Buka app di dashboard
2. Klik "Manage app"
3. Lihat logs real-time

### Performance Monitoring

Tambahkan di app:
```python
import time

start = time.time()
# ... inference code ...
end = time.time()

st.metric("Inference Time", f"{(end-start)*1000:.2f} ms")
```

## 🛠️ Troubleshooting

### Error: "Module not found"
- Pastikan semua dependencies di `requirements.txt`
- Cek versi compatibility

### Error: "Out of memory"
- Gunakan model quantized
- Reduce batch size
- Use CPU instead of GPU

### Error: "Model loading failed"
- Cek path model
- Verify checkpoint format
- Test locally first

### Slow deployment
- Reduce dependencies
- Use lighter torch version
- Cache model download

## 📝 Best Practices

1. ✅ Test locally sebelum deploy
2. ✅ Use semantic versioning untuk dependencies
3. ✅ Add error handling
4. ✅ Include loading indicators
5. ✅ Document API clearly
6. ✅ Monitor performance
7. ✅ Regular updates

## 🔗 Resources

- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-community-cloud)
- [Docker Documentation](https://docs.docker.com/)
- [Hugging Face Hub](https://huggingface.co/docs/hub)
- [GitHub LFS](https://git-lfs.github.com/)

---

**Happy Deploying! 🚀**
