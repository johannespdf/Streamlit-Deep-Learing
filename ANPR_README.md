# ANPR System - Project Structure

## Overview
Complete end-to-end Automatic Number Plate Recognition system with 5-stage pipeline.

## Files Created

### Main Application
- **anpr_app.py** - Complete ANPR system with all pipeline stages
- **requirements_anpr.txt** - Dependencies for ANPR app

### Pipeline Stages
1. **Input**: Vehicle image upload
2. **Detection**: Faster R-CNN (ResNet-50 + FPN)  
3. **Segmentation**: Contour-based character localization
4. **Classification**: Custom CNN (36 classes: A-Z, 0-9)
5. **Post-processing**: Character ordering and formatting

## How to Run

```bash
# Install dependencies
pip install -r requirements_anpr.txt

# Run application
streamlit run anpr_app.py
```

## Application URLs

- **ANPR System**: http://localhost:8503
- **CNN Classifier**: http://localhost:8502  
- **Plate Detection**: http://localhost:8501

## Models

Both models are defined directly in code (no external .pth files):

1. **PlateDetectorCNN** - Faster R-CNN for plate detection
2. **CharacterCNN** - Custom CNN for character classification

## Important Notes

> Models have **random weights** (untrained). Predictions will be random until trained on actual datasets.

## Training (Future)

To get accurate results:
1. Collect plate detection dataset
2. Collect character image dataset  
3. Add training loops to application
4. Train models for 20-50 epochs

## Project Directory

```
DEEPLEARNINGGGGG/
├── anpr_app.py              ← New ANPR system
├── requirements_anpr.txt    ← ANPR dependencies
├── cnn_classifier_app.py
├── app.py
├── requirements.txt
├── inference_example.py
└── save_model.py
```
