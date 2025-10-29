# utils.py
import io
import base64
from PIL import Image
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from torchvision.models import resnet50
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --------- Helpers ----------
def pil_from_bytes(bts):
    return Image.open(io.BytesIO(bts)).convert("RGB")

def cv2_from_bytes(bts):
    arr = np.frombuffer(bts, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img  # BGR

def pil_to_base64(img_pil):
    buf = io.BytesIO()
    img_pil.save(buf, format='PNG')
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

def cv2_to_pil(cv2_img_bgr):
    cv2_rgb = cv2.cvtColor(cv2_img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_rgb)

# --------- SSIM using OpenCV ----------
def compute_ssim_visual(img1_bgr, img2_bgr):
    g1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)

    # ensure same size
    if g1.shape != g2.shape:
        g2 = cv2.resize(g2, (g1.shape[1], g1.shape[0]))

    # OpenCV’s SSIM (requires opencv-contrib-python)
    ssim_score = cv2.quality.QualitySSIM_compute(g1, g2)[0][0]

    # visualize difference
    diff = cv2.absdiff(g1, g2)
    diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    pil = cv2_to_pil(diff_colored)
    return float(ssim_score), pil_to_base64(pil)

# --------- Histogram similarity (HSV) ----------
def compute_histogram_similarity(img1_bgr, img2_bgr, bins=(8, 8, 8)):
    img1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2HSV)
    img2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2HSV)
    h1 = cv2.calcHist([img1], [0,1,2], None, bins, [0,180,0,256,0,256])
    h2 = cv2.calcHist([img2], [0,1,2], None, bins, [0,180,0,256,0,256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    sim = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    sim = max(min(sim, 1.0), -1.0)
    sim_norm = (sim + 1) / 2
    return float(sim_norm)

# --------- ORB keypoint matching visual ----------
def compute_orb_matches(img1_bgr, img2_bgr, max_matches=50):
    orb = cv2.ORB_create(2000)
    k1, d1 = orb.detectAndCompute(img1_bgr, None)
    k2, d2 = orb.detectAndCompute(img2_bgr, None)
    if d1 is None or d2 is None:
        return 0.0, None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(d1, d2, k=2)
    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
            good.append(m)
    score = min(len(good) / 200.0, 1.0)
    good = sorted(good, key=lambda x: x.distance)[:max_matches]
    match_img = cv2.drawMatches(img1_bgr, k1, img2_bgr, k2, good, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    pil = cv2_to_pil(match_img)
    return float(score), pil_to_base64(pil)

# --------- ResNet50 embedding (torch) ----------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
_resnet_model = None
_resnet_preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def load_resnet():
    global _resnet_model
    if _resnet_model is None:
        model = resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.to(device).eval()
        _resnet_model = model
    return _resnet_model

def get_resnet_embedding(pil_img):
    model = load_resnet()
    x = _resnet_preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(x).squeeze()
    feat = feat.cpu().numpy().reshape(-1)
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm
    return feat

# --------- CLIP embedding (transformers) ----------
_clip_model = None
_clip_processor = None
def load_clip(model_name="openai/clip-vit-base-patch32"):
    global _clip_model, _clip_processor
    if _clip_model is None:
        _clip_model = CLIPModel.from_pretrained(model_name)
        _clip_processor = CLIPProcessor.from_pretrained(model_name)
        _clip_model.to(device).eval()
    return _clip_model, _clip_processor

def get_clip_embedding(pil_img, model_name="openai/clip-vit-base-patch32"):
    model, processor = load_clip(model_name)
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        image_embeds = model.get_image_features(**inputs)
    emb = image_embeds.cpu().numpy().reshape(-1)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb

# --------- Cosine similarity ----------
def cosine_sim(a, b):
    a = np.array(a).reshape(1, -1)
    b = np.array(b).reshape(1, -1)
    return float(cosine_similarity(a, b)[0, 0])

# --------- Aggregate scores ----------
def aggregate_scores(scores: dict, weights=None):
    default = {'ssim':0.1, 'hist':0.1, 'orb':0.1, 'resnet':0.4, 'clip':0.3}
    if weights is None:
        weights = default
    present = {k: v for k,v in scores.items() if v is not None}
    w = {k: weights.get(k, 0.0) for k in present.keys()}
    s = sum(w.values()) or 1.0
    total = 0.0
    for k, val in present.items():
        total += (w.get(k, 0.0) / s) * float(val)
    return float(total)

# --------- Gemini explanation wrapper (stub) ----------
def generate_gemini_explanation_stub(caption1, caption2, final_score):
    if final_score > 0.75:
        return f"Both images appear visually similar. Detected: {caption1} and {caption2}. They seem to contain similar objects or scenes."
    elif final_score > 0.45:
        return f"The images share some visual characteristics (like color or shape), but differ in certain features. Detected: {caption1} vs {caption2}."
    else:
        return f"The images are different. Detected: {caption1} vs {caption2}."

# --------- Model Evaluation (Accuracy Metrics) ----------
def evaluate_model(pred_scores, true_labels, threshold=0.7):
    """
    pred_scores: list or np.array of similarity scores (0–1)
    true_labels: list or np.array of true labels (1=similar, 0=different)
    threshold: similarity threshold to classify as similar/dissimilar
    """
    pred_labels = [1 if s >= threshold else 0 for s in pred_scores]
    acc = accuracy_score(true_labels, pred_labels)
    prec = precision_score(true_labels, pred_labels)
    rec = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    print(f"Accuracy  : {acc:.3f}")
    print(f"Precision : {prec:.3f}")
    print(f"Recall    : {rec:.3f}")
    print(f"F1-score  : {f1:.3f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }
