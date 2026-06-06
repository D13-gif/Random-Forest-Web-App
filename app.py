import os
import cv2
import numpy as np
import pickle
from flask import (
    Flask, render_template, request, redirect, url_for, flash, send_from_directory
)
from werkzeug.utils import secure_filename
from pathlib import Path

app = Flask(__name__)
# SECRET_KEY: Set via environment variable di production
# Di development, key acak di-generate otomatis setiap startup
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(24).hex()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
MODEL_PATH = BASE_DIR / "model" / "rf_model.pkl"
ENCODER_PATH = BASE_DIR / "model" / "label_encoder.pkl"

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

CLASS_INFO = {
    "cardboard": {
        "id": "Kardus",
        "description": "Material kemasan terbuat dari serat pulp yang tebal.",
        "tip": "Tekankan untuk melunakkan kardus sebelum didaur ulang. Lepaskan semua selotip dan staples. Simpan di tempat kering agar tidak tercemar minyak atau makanan.",
        "icon": "bi-box-seam",
        "color": "#8B4513",
        "bg_color": "rgba(139, 69, 19, 0.1)",
    },
    "glass": {
        "id": "Kaca",
        "description": "Material transparan rapuh yang terbuat dari silika.",
        "tip": "Keluarkan tutup botol dan tutupnya (logam/plastik). Bilas sebentar untuk menghilangkan sisa minyak atau zat kimia. Pecahan kaca sebaiknya dibungkus sebelum dibuang.",
        "icon": "bi-cup-straw",
        "color": "#2E86AB",
        "bg_color": "rgba(46, 134, 171, 0.1)",
    },
    "metal": {
        "id": "Logam",
        "description": "Barang bekas terbuat dari besi, aluminium, atau campuran logam.",
        "tip": "Gunakan tempat pengelompokan khusus untuk logam. Kaleng minuman harus ditekan untuk menghemat ruang. Logam yang besar sebaiknya diserahkan ke bank sampah atau pusat daur ulang khusus.",
        "icon": "bi-tools",
        "color": "#6C757D",
        "bg_color": "rgba(108, 117, 125, 0.1)",
    },
    "paper": {
        "id": "Kertas",
        "description": "Material cetak atau tulis dari serat kayu yang telah diproses.",
        "tip": "Pisahkan kertas bersih dari yang sudah tercemar (minyak, makanan, cat). Kertas tisu bekas tidak bisa didaur ulang. Lipat karton bekas agar hemat ruang di tempat pengumpulan.",
        "icon": "bi-file-earmark-text",
        "color": "#F4A261",
        "bg_color": "rgba(244, 162, 97, 0.1)",
    },
    "plastic": {
        "id": "Plastik",
        "description": "Material polimer sintetis yang fleksibel dan tahan lama.",
        "tip": "Cek kode daur ulang (angka 1-7) pada kemasan. Cuci bersih sebelum didaur ulang untuk menghindari kontaminasi. Kantong plastik tipis sebaiknya dikumpulkan terpisah dari botol plastik tebal.",
        "icon": "bi-basket",
        "color": "#E76F51",
        "bg_color": "rgba(231, 111, 81, 0.1)",
    },
    "trash": {
        "id": "Sampah Residu",
        "description": "Sampah yang tidak bisa didaur ulang dan harus dibuang ke TPA.",
        "tip": "Sampah ini harus dibuang ke tempat pembuangan akhir (TPA). Kurangi produksi sampah residu dengan menghindari barang sekali pakai. Pilih produk dengan kemasan minimal.",
        "icon": "bi-trash3",
        "color": "#2B2D42",
        "bg_color": "rgba(43, 45, 66, 0.1)",
    },
}


def extract_features(img_path, img_size=64):
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    img_resized = cv2.resize(img, (img_size, img_size))

    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
    color_hist = np.concatenate([hist_h, hist_s, hist_v])
    color_hist /= (color_hist.sum() + 1e-7)
    cv2.normalize(color_hist, color_hist, 1.0, 0.0, cv2.NORM_L1)

    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor(
        _winSize=(64, 64),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    hog_features = hog.compute(gray).flatten()

    gray_32 = cv2.resize(gray, (32, 32)).flatten()

    features = np.concatenate([color_hist, hog_features, gray_32]).astype(np.float32)
    return features


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Load model sekali saat startup ──────────────────────────────
MODEL, ENCODER, MODEL_LOADED = None, None, False
try:
    with open(MODEL_PATH, "rb") as f:
        MODEL = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        ENCODER = pickle.load(f)
    MODEL_LOADED = True
    print("✅ Model berhasil dimuat!")
except FileNotFoundError:
    print("⚠️  WARNING: Model belum ada. Jalankan train_model.py terlebih dahulu.")
except Exception as e:
    print(f"❌ ERROR saat memuat model: {e}")


@app.errorhandler(413)
def too_large(e):
    flash("File terlalu besar. Maksimal ukuran file adalah 16MB.", "danger")
    return redirect(url_for("index"))


@app.errorhandler(500)
def internal_error(e):
    flash("Terjadi kesalahan internal server.", "danger")
    return redirect(url_for("index"))


@app.route("/")
def index():
    cm_exists = (BASE_DIR / "static" / "confusion_matrix.png").exists()
    return render_template(
        "index.html",
        confusion_matrix_exists=cm_exists,
        class_info=CLASS_INFO,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_LOADED:
        flash("⚠️ Model belum dilatih. Jalankan train_model.py terlebih dahulu.", "warning")
        return redirect(url_for("index"))

    if "file" not in request.files:
        flash("Tidak ada file yang diunggah.", "danger")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("Silakan pilih file gambar terlebih dahulu.", "warning")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash(
            "Format file tidak didukung. Gunakan JPG, PNG, GIF, BMP, atau WebP.",
            "danger",
        )
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    unique_name = f"{os.urandom(8).hex()}_{filename}"
    save_path = UPLOAD_FOLDER / unique_name
    file.save(str(save_path))

    try:
        features = extract_features(save_path)
        features = features.reshape(1, -1)

        prediction = MODEL.predict(features)[0]
        probabilities = MODEL.predict_proba(features)[0]
        # FIX BUG 3: konversi ke str agar match key dict Python
        predicted_class = str(ENCODER.inverse_transform([prediction])[0])
        confidence = float(probabilities[prediction]) * 100

        # FIX BUG 1: gunakan ENCODER.classes_ (nama kelas string), bukan MODEL.classes_ (int)
        all_probs = {}
        for idx, cls_name in enumerate(ENCODER.classes_):
            all_probs[str(cls_name)] = float(probabilities[idx]) * 100

        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        result_image = f"/static/uploads/{unique_name}"

        # FIX BUG 2: kirim class_info spesifik untuk kelas terprediksi, bukan seluruh dict
        info = CLASS_INFO.get(predicted_class, {
            "id": predicted_class.title(),
            "description": "Kategori sampah terdeteksi.",
            "tip": "Buang sampah pada tempatnya.",
            "icon": "bi-trash",
            "color": "#6C757D",
            "bg_color": "rgba(108,117,125,0.1)",
        })

        return render_template(
            "result.html",
            predicted_class=predicted_class,
            confidence=confidence,
            all_probs=sorted_probs,
            result_image=result_image,
            class_info=info,
        )

    except Exception as e:
        flash(f"Gagal memproses gambar: {str(e)}", "danger")
        return redirect(url_for("index"))


@app.route("/about")
def about():
    return render_template("about.html") if os.path.exists(
        BASE_DIR / "templates" / "about.html"
    ) else render_template("index.html")


@app.route("/static/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(str(UPLOAD_FOLDER), filename)


@app.context_processor
def inject_confusion_matrix():
    exists = (BASE_DIR / "static" / "confusion_matrix.png").exists()
    return dict(cm_exists=exists)


if __name__ == "__main__":
    if not MODEL_LOADED:
        print("WARNING: Model files not found. Please run 'python train_model.py' first.")
    print("Starting GarbageRF Classifier...")
    app.run(debug=True, host="0.0.0.0", port=5000)