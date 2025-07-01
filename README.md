## 🧠 Real-Time Face Mask Detection

This project implements a real-time face mask detection system using a Convolutional Neural Network (CNN) trained on both a curated image dataset and real-world webcam-captured data. It uses OpenCV and MTCNN to detect faces and classify them into:

* `face_with_mask`
* `face_no_mask`
* `uncertain` (e.g., side profiles, occlusions)

---

### 📁 Folder Structure

```
.
├── dataset/                 # Webcam-collected images
│   ├── face_with_mask/
│   ├── face_no_mask/
│   └── uncertain/
├── combined_dataset/        # Merged dataset for training (from dataset/ + CSV source)
├── data/
│   └── face/
│       ├── train.csv
│       └── Medical mask/... # Original images
├── train_combined_model.py  # Training script
├── collect_faces.py         # Webcam face image collector
├── real_time_detector.py    # Real-time detection script
├── mask_classifier.h5       # Trained Keras model (output)
├── label_encoder.pkl        # Saved label encoder (output)
└── README.md
```

---

### 🚀 Features

* ✅ Real-time face detection using webcam
* ✅ Multi-class classification (`with mask`, `no mask`, `uncertain`)
* ✅ CNN training on both pre-annotated and webcam datasets
* ✅ MTCNN for robust face detection
* ✅ Confidence scores on predictions
* ✅ Model and label encoder saved for reuse

---

### 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/your_username/face-mask-detector.git
cd face-mask-detector

# Create and activate virtual environment (optional)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

> You’ll need `tensorflow`, `opencv-python`, `mtcnn`, `joblib`, `matplotlib`, `pandas`, and `scikit-learn`.

---

### 🎥 How to Use

#### 1. **Collect Real-World Face Data (Optional but Recommended)**

Run this to collect labeled face images:

```bash
python collect_faces.py
```

Change the label inside the script for each category:
`face_with_mask`, `face_no_mask`, `uncertain`

---

#### 2. **Train the CNN Model**

```bash
python train_combined_model.py
```

* Merges CSV and webcam datasets
* Trains a CNN
* Saves `mask_classifier.h5` and `label_encoder.pkl`

---

#### 3. **Run Real-Time Mask Detection**

```bash
python real_time_detector.py
```

A window will open with your webcam feed. Face predictions will be shown with bounding boxes and confidence scores.

---

### 📊 Example Output

* 🟩 Green box: Mask detected
* 🟥 Red box: No mask
* 🟨 Yellow box: Uncertain (e.g., side profile, hand covering face)

---

### 🧠 Model Architecture

* 2 Convolutional layers
* MaxPooling
* Dense + Dropout
* Output layer with softmax over 3 classes

---

### ✍️ Author

Gourav Sharma
Computer Science, Simon Fraser University
ghs4@sfu.ca


