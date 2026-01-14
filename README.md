# 🎥 Unsupervised Video Anomaly Detection Using Motion-Based Autoencoders

## 📌 Overview
This project implements an end-to-end **unsupervised video anomaly detection pipeline** using **public-domain video data**. The objective is to learn normal motion patterns in video streams and identify anomalous segments by analyzing reconstruction error from an autoencoder model.

Instead of relying on labeled anomalies, the system models expected frame-to-frame motion and surfaces deviations as a **long-tail anomaly signal**, which is commonly observed in real-world monitoring and video analytics systems.

---

## 🧠 Core Idea
- Normal video behavior is repetitive and predictable  
- Anomalies disrupt learned motion patterns  
- These disruptions appear in the **long tail** of the reconstruction error distribution  

---

## 🛠️ Technical Approach

### 1. Video Processing
- Decode video frames using **OpenCV**
- Convert frames to grayscale
- Extract motion features via frame differencing

### 2. Feature Engineering
- Generate compact motion vectors from consecutive frames
- Normalize features for stable training

### 3. Modeling
- Train a **PyTorch Autoencoder** to learn normal motion representations
- Compress high-dimensional motion features into a low-dimensional latent space

### 4. Anomaly Scoring
- Compute **frame-level reconstruction error**
- Higher error indicates greater deviation from learned normal behavior

### 5. Evaluation & Visualization
- Plot reconstruction error across frames
- Identify a clear **peak (normal frames)** and **long tail (anomalies)**

---

## 📊 Results
- **Peak near zero**: expected, repetitive motion  
- **Long tail**: rare or abnormal motion patterns  
- Frames in the tail are flagged for further inspection

This approach emphasizes **signal generation rather than hard classification**, reflecting how production-scale anomaly detection systems operate.

---

## 📂 Dataset
- **Source:** *Big Buck Bunny* (public-domain video)
- **License:** Open and unrestricted use
- **Privacy:** Contains no user data or personal identifiers

---

## 🚀 How to Run
This project is implemented as a **single-cell Google Colab notebook**.

1. Open the Colab notebook  
2. Run the cell to install dependencies, download the video, train the model, and visualize results  
3. Inspect the reconstruction error plot for anomalies  

---

## 📦 Technologies Used
- Python  
- OpenCV  
- PyTorch  
- NumPy  
- Matplotlib  

---

## 🎯 Why This Project Matters
- Demonstrates video decoding and frame-level analysis  
- Applies machine learning to streaming-style video data  
- Highlights long-tail anomaly behavior  
- Uses open-source, privacy-safe data  

---

## 🔮 Future Enhancements
- Spatiotemporal models (CNNs, ConvLSTM)
- Codec-aware features (motion vectors, residuals)
- Online inference for real-time streams
- Adaptive thresholding strategies

---

## 👤 Author
**Deepthi Venmanasseril Jainlal**  
Graduate Student | Data Science & Machine Learning
