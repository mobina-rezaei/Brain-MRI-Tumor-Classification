"""
Brain Tumor Diagnosis Web App
-----------------------------
- Upload an MRI image in browser
- Model predicts one of 4 classes
- Returns label + confidence + per-class probabilities
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import uuid
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, url_for, redirect, flash
from werkzeug.utils import secure_filename
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "outputs", "brain_tumor_cnn.keras")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
IMAGE_SIZE    = 224
CLASS_NAMES   = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
ALLOWED_EXT   = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}

# Human-friendly labels shown in the UI
LABEL_FA = {
    "glioma_tumor":     "تومور گلیوما",
    "meningioma_tumor": "تومور مننژیوما",
    "no_tumor":         "بدون تومور",
    "pituitary_tumor":  "تومور هیپوفیز",
}
LABEL_EN = {
    "glioma_tumor":     "Glioma tumor",
    "meningioma_tumor": "Meningioma tumor",
    "no_tumor":         "No tumor",
    "pituitary_tumor":  "Pituitary tumor",
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = "brain-mri-secret-2026"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# ---------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded:", MODEL_PATH)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def predict_image(filepath):
    """Run inference and return dict with label, confidence, probs."""
    img = tf.keras.preprocessing.image.load_img(
        filepath, target_size=(IMAGE_SIZE, IMAGE_SIZE)
    )
    arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = CLASS_NAMES[idx]
    return {
        "label":       label,
        "label_fa":    LABEL_FA[label],
        "label_en":    LABEL_EN[label],
        "confidence":  float(probs[idx]) * 100.0,
        "probabilities": [
            {
                "class":    c,
                "label_fa": LABEL_FA[c],
                "label_en": LABEL_EN[c],
                "prob":     float(p) * 100.0,
            }
            for c, p in zip(CLASS_NAMES, probs)
        ],
    }


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            flash("فایلی انتخاب نشده است.")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("نام فایل خالی است.")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("فرمت فایل پشتیبانی نمی‌شود. از PNG یا JPG استفاده کنید.")
            return redirect(request.url)

        # unique filename to avoid collisions
        ext = file.filename.rsplit(".", 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(save_path)

        try:
            result = predict_image(save_path)
        except Exception as e:
            flash(f"خطا در پردازش تصویر: {e}")
            return redirect(request.url)

        image_url = url_for("static", filename=f"uploads/{unique_name}")
        return render_template("index.html", result=result, image_url=image_url)

    return render_template("index.html", result=None, image_url=None)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)