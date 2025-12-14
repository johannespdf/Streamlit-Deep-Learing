# 🚀 Quick Start Guide

## Instalasi & Menjalankan Aplikasi

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Simpan Model dari Colab

Tambahkan code ini di akhir notebook Colab Anda:

```python
# Simpan model
torch.save(model.state_dict(), 'plate_detection_model.pth')

# Download dari Colab
from google.colab import files
files.download('plate_detection_model.pth')
```

### 3. Jalankan Aplikasi
```bash
streamlit run app.py
```

### 4. Buka Browser
Aplikasi akan terbuka otomatis di `http://localhost:8501`

## 📋 Checklist

- [ ] Python 3.8+ terinstall
- [ ] Dependencies terinstall (`pip install -r requirements.txt`)
- [ ] File model (.pth) sudah di-download dari Colab
- [ ] Gambar test sudah disiapkan

## 🎯 Cara Menggunakan

1. **Upload Model** (di sidebar)
   - Klik "Browse files"
   - Pilih file `.pth` yang sudah di-download

2. **Upload Gambar**
   - Klik "Choose an image"
   - Pilih gambar plat nomor

3. **Adjust Settings** (optional)
   - Confidence Threshold: 0.5 (default)
   - Number of Classes: 37 (default)

4. **Lihat Hasil**
   - Bounding boxes akan muncul di gambar
   - Detail deteksi ada di bagian bawah

## ⚙️ Konfigurasi

### Jika model Anda berbeda:

**Number of Classes**
- Default: 37 (0-9 + A-Z + background)
- Sesuaikan dengan jumlah kelas di model Anda

**Label Mapping**
Edit fungsi `label_to_char()` di `app.py` jika mapping berbeda:
```python
def label_to_char(label):
    # Sesuaikan dengan label Anda
    if label == 0:
        return "BG"
    # ... dst
```

## 🐛 Troubleshooting

### Model tidak load
```
Error: Error loading model: ...
```
**Solusi:**
- Pastikan file .pth benar
- Cek num_classes di sidebar
- Pastikan format checkpoint sesuai

### Tidak ada deteksi
```
No objects detected
```
**Solusi:**
- Turunkan confidence threshold ke 0.3 atau 0.2
- Pastikan gambar jelas dan tidak blur
- Cek apakah model sudah trained dengan baik

### Aplikasi lambat
**Solusi:**
- Model inference di CPU memang lebih lambat
- Untuk production, pertimbangkan GPU atau model yang lebih ringan (YOLO)

## 📚 File Explanation

| File | Deskripsi |
|------|-----------|
| `app.py` | Main aplikasi Streamlit |
| `requirements.txt` | Dependencies Python |
| `save_model.py` | Helper untuk save model dari Colab |
| `inference_example.py` | Test inference tanpa Streamlit |
| `README.md` | Dokumentasi lengkap |
| `DEPLOYMENT.md` | Guide untuk deploy ke cloud |

## 🔄 Workflow

```
Training (Colab) → Save Model (.pth) → Download → Streamlit App → Inference
```

## 💡 Tips

1. **Test model dulu** dengan `inference_example.py` sebelum deploy
2. **Simpan checkpoint** selama training untuk monitoring
3. **Backup model** di cloud storage
4. **Document changes** setiap kali update model

## 📞 Support

Jika ada masalah:
1. Cek logs di terminal
2. Baca error message dengan teliti
3. Cek dokumentasi di `README.md`
4. Test dengan inference_example.py untuk isolasi masalah

## ✅ Next Steps

Setelah aplikasi jalan:
1. [ ] Test dengan berbagai gambar
2. [ ] Fine-tune confidence threshold
3. [ ] Deploy ke Streamlit Cloud (lihat DEPLOYMENT.md)
4. [ ] Share link dengan team!

---

**Selamat mencoba! 🎉**
