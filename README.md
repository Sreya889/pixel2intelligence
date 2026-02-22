# pixel2intelligence
# COIL-20 Object Segmentation Using Classical Image Processing (Google Colab)

## Project Overview
This project implements an **object segmentation pipeline** for the COIL-20 dataset using **classical image processing techniques** in **Google Colab**.  

The pipeline segments objects from grayscale images without using machine learning, following these main steps:

1. **Noise Removal** – Apply Gaussian blur to reduce noise.  
2. **Edge Detection** – Detect object edges using **Canny** and **Sobel** operators for comparison.  
3. **Morphological Operations** – Refine edges and close gaps using morphological closing.  
4. **Contour Extraction & Masking** – Extract contours and generate object masks for clean segmentation.  

---

## Dataset
- **COIL-20 (Columbia Object Image Library)**  
- Grayscale images of 20 objects, each captured from 72 viewing angles (total 1440 images)  
- Image size: 128 × 128 pixels  
- Dataset used: `/MyDrive/coil-20-proc/coil-20-proc` in Google Drive  

**Download Links:**  
- [COIL-20 Repository](https://cave.cs.columbia.edu/repository/COIL-20)  
- [COIL-20 Software & Dataset](https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php)  

---

## Google Colab Setup
1. Open [Google Colab](https://colab.research.google.com/)  
2. Mount your Google Drive to access the dataset:
```python
from google.colab import drive
drive.mount('/content/drive')
