# Image_Comparision

# Image Similarity Analyzer

> A powerful deep-learning–based web app that compares two images and calculates their visual similarity using **VGG16**, **ResNet50**, and **Gemini AI** for intelligent hybrid analysis.


## Overview

**Image Similarity Analyzer** is a Flask-based application that allows users to upload two images and instantly evaluate how similar they are.  
The app combines **convolutional neural networks (CNNs)** with **Gemini AI**’s advanced visual understanding to deliver both numerical and semantic similarity insights.  

---

##  Key Features

-  Upload any two images for comparison  
-  Feature extraction using **VGG16** and **ResNet50**  
-  **Gemini Wrapper Integration** for contextual and semantic image analysis  
-  Multiple similarity metrics — Cosine / Euclidean  
-  Minimal Flask web interface (only `app.py` + `utils.py`)  
-  Instant similarity score and analysis summary  

---

## Project Structure

image-similarity-analyzer/
├── app.py # Flask web app (frontend + API + Gemini integration)
├── utils.py # Helper functions for image preprocessing & similarity
├── requirements.txt
├── assets/ # Images, icons, or demo gifs
└── README.md


---

## Tech Stack

| Layer | Technology |
|-------|-------------|
| **Frontend** | HTML, CSS (Flask templates) |
| **Backend** | Python (Flask) |
| **Deep Learning** | TensorFlow / Keras (VGG16, ResNet50) |
| **AI Integration** | Gemini API Wrapper |
| **Image Processing** | OpenCV, NumPy, Pillow |
| **Similarity Metrics** | Cosine, Euclidean |

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/image-similarity-analyzer.git
cd image-similarity-analyzer

# Install dependencies
pip install -r requirements.txt

# Add your Gemini API key in a .env file
echo "GEMINI_API_KEY=your_api_key_here" > .env

# Run the Flask server
python app.py

Now open your browser and visit:
👉 http://127.0.0.1:5000/

