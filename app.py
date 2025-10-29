from flask import Flask, request
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import base64
from io import BytesIO
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)

# ---------------------------
# Utility Functions
# ---------------------------

def cv2_from_bytes(file_bytes):
    nparr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def pil_from_bytes(file_bytes):
    return Image.open(BytesIO(file_bytes)).convert("RGB")

def encode_image(image_bgr):
    _, buf = cv2.imencode('.jpg', image_bgr)
    return f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"

def resize_to_match(img1, img2):
    """Resize img2 to match img1's shape (for SSIM & histogram)."""
    h, w = img1.shape[:2]
    return cv2.resize(img2, (w, h))

def compute_ssim_visual(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, diff = ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")
    diff_colored = cv2.applyColorMap(255 - diff, cv2.COLORMAP_JET)
    return score, encode_image(diff_colored)

def compute_histogram_similarity(img1, img2):
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8],
                         [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8],
                         [0, 256, 0, 256, 0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def cosine_sim(vec1, vec2):
    """Compute cosine similarity safely between flattened tensors."""
    vec1 = vec1.view(-1)
    vec2 = vec2.view(-1)
    return torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()

def get_resnet_embedding(img):
    """Get deep feature embedding using pretrained ResNet18."""
    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # remove final FC layer
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    with torch.no_grad():
        emb = model(transform(img).unsqueeze(0))
    return emb.flatten()

# ---------------------------
# Routes
# ---------------------------

@app.route('/')
def home():
    return '''
    <h2>Upload two images to compare similarity</h2>
    <form method="POST" action="/compare" enctype="multipart/form-data">
        <label>Image 1:</label><br>
        <input type="file" name="img1" required><br><br>
        <label>Image 2:</label><br>
        <input type="file" name="img2" required><br><br>
        <button type="submit">Compare</button>
    </form>
    '''

@app.route('/compare', methods=['POST'])
def compare_images():
    try:
        img1_bytes = request.files['img1'].read()
        img2_bytes = request.files['img2'].read()

        img1_bgr = cv2_from_bytes(img1_bytes)
        img2_bgr = cv2_from_bytes(img2_bytes)

        # Resize second image to match first
        img2_bgr = resize_to_match(img1_bgr, img2_bgr)

        # SSIM
        ssim_score, ssim_vis = compute_ssim_visual(img1_bgr, img2_bgr)
        # Histogram
        hist_score = compute_histogram_similarity(img1_bgr, img2_bgr)
        # CNN Embedding Similarity
        emb1 = get_resnet_embedding(pil_from_bytes(img1_bytes))
        emb2 = get_resnet_embedding(pil_from_bytes(img2_bytes))
        resnet_score = cosine_sim(emb1, emb2)

        # Final Aggregated Score
        final_score = (ssim_score + hist_score + resnet_score) / 3

        explanation = (
            f"<b>SSIM (Structural Similarity):</b> {ssim_score:.2f}<br>"
            f"<b>Histogram Correlation:</b> {hist_score:.2f}<br>"
            f"<b>CNN Feature Similarity:</b> {resnet_score:.2f}<br><br>"
            f"<b>Overall Similarity Score:</b> {final_score:.2f}<br><br>"
            f"Interpretation: The images share {final_score*100:.1f}% visual similarity "
            f"based on structure, color, and deep feature patterns extracted by a CNN."
        )

        return f"""
        <h3>Similarity Analysis Results</h3>
        <p>{explanation}</p>
        <h4>SSIM Difference Visualization</h4>
        <img src="{ssim_vis}" width="400"><br><br>
        <a href="/">Compare Again</a>
        """

    except Exception as e:
        return f"<h3>Error occurred:</h3><pre>{str(e)}</pre>"

# ---------------------------
# Run Server
# ---------------------------
if __name__ == '__main__':
    print("Image Similarity Service is up at http://127.0.0.1:5000/")
    app.run(debug=True)
