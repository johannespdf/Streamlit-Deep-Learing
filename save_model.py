"""
Script untuk menyimpan model yang sudah dilatih ke file .pth

Gunakan script ini setelah training selesai untuk menyimpan model.
"""

import torch

def save_model(model, save_path='plate_detection_model.pth'):
    """
    Simpan model ke file .pth
    
    Args:
        model: model PyTorch yang sudah dilatih
        save_path: path untuk menyimpan file
    """
    # Simpan hanya state_dict (lebih efisien)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def save_checkpoint(model, optimizer, epoch, loss, save_path='checkpoint.pth'):
    """
    Simpan checkpoint lengkap termasuk optimizer state
    
    Args:
        model: model PyTorch
        optimizer: optimizer
        epoch: epoch saat ini
        loss: loss saat ini
        save_path: path untuk menyimpan
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

# Contoh penggunaan setelah training:
# 
# Setelah training selesai di Colab, jalankan:
# save_model(model, 'plate_detection_model.pth')
# 
# Atau untuk save checkpoint lengkap:
# save_checkpoint(model, optimizer, epoch, loss, 'checkpoint.pth')
#
# Download file .pth dan gunakan di aplikasi Streamlit
