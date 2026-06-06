# GarbageRF Classifier
### Sistem Klasifikasi Sampah Berbasis Random Forest Algorithm

Aplikasi web Flask untuk klasifikasi sampah menggunakan algoritma Random Forest dengan ekstraksi fitur gabungan (Color Histogram HSV, HOG, dan Flattened Pixels).

---

## 📁 Struktur Proyek

```
garbage_classifier/
├── app.py                          # Flask web application
├── train_model.py                  # Script training Random Forest
├── requirements.txt
├── Procfile
├── runtime.txt
├── README.md
├── model/
│   ├── rf_model.pkl                # Model Random Forest (hasil training)
│   └── label_encoder.pkl           # Label encoder (hasil training)
├── dataset/
│   └── Garbage classification/     # Dataset gambar sampah
│       ├── cardboard/
│       ├── glass/
│       ├── metal/
│       ├── paper/
│       ├── plastic/
│       └── trash/
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   ├── uploads/                    # Folder upload gambar user
│   └── confusion_matrix.png        # Hasil evaluasi model
└── templates/
    ├── base.html
    ├── index.html
    ├── result.html
    └── about.html
```

## 📥 Dataset

Dataset: [Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)

**Struktur folder dataset:**
```
dataset/Garbage classification/
├── cardboard/  # ~400 gambar
├── glass/      # ~400 gambar
├── metal/      # ~400 gambar
├── paper/      # ~500 gambar
├── plastic/    # ~500 gambar
└── trash/      # ~100 gambar
```

Simpan dataset di `dataset/Garbage classification/` sesuai struktur di atas.

## ⚙️ Instalasi

### 1. Clone / Download proyek
```bash
cd garbage_classifier
```

### 2. Buat virtual environment (disarankan)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Letakkan dataset
Pastikan folder dataset berada di:
```
dataset/Garbage classification/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

## 🎓 Training Model

Jalankan script training untuk melatih model Random Forest:

```bash
python train_model.py
```

Script akan:
1. Memuat dataset dari folder `dataset/Garbage classification/`
2. Mengekstrak fitur dari setiap gambar (HSV + HOG + Pixels = 2.884 fitur)
3. Membagi data (80% training, 20% testing, stratified)
4. Melatih Random Forest (200 estimators, max_depth=20)
5. Mengevaluasi dan menampilkan classification report
6. Menyimpan confusion matrix ke `static/confusion_matrix.png`
7. Menyimpan model ke `model/rf_model.pkl` dan encoder ke `model/label_encoder.pkl`

### Parameter Model
| Parameter         | Nilai  |
|-------------------|--------|
| Algoritma         | Random Forest |
| n_estimators      | 200    |
| max_depth         | 20     |
| n_jobs            | -1     |
| random_state      | 42     |

### Feature Extraction (Total: 2.884 fitur)
| Fitur                  | Jumlah | Deskripsi                               |
|------------------------|--------|-----------------------------------------|
| HSV Color Histogram    | 96     | 32 bins per channel (H, S, V)           |
| HOG                    | 1764   | winSize=(64,64), block=(16,16), nbins=9 |
| Grayscale Pixels       | 1024   | Resize 32x32, flatten                   |

## 🚀 Menjalankan Aplikasi

Setelah model dilatih:

```bash
python app.py
```

Buka browser dan akses: `http://localhost:5000`

## 🐳 Deployment

### Heroku

1. Buat file `Procfile` dan `runtime.txt` (sudah disediakan)
2. Push ke GitHub
3. Deploy di Heroku:
```bash
heroku create garbage-rf-classifier
git push heroku main
heroku open
```

### Railway

1. Connect repositori GitHub ke Railway
2. Railway akan otomatis mendeteksi `Procfile` dan `runtime.txt`
3. Deploy otomatis

### Buildpack untuk Heroku (opsional)

Tambahkan buildpack Python:
```
https://github.com/heroku/heroku-buildpack-python
```

## 📊 Evaluasi Model

Setelah training, lihat `static/confusion_matrix.png` untuk melihat:
- Confusion matrix visual
- Classification report di terminal

## 🔧 Troubleshooting

### Model belum dilatih
```
Model belum dilatih. File hilang: model/rf_model.pkl, model/label_encoder.pkl.
Jalankan 'python train_model.py' terlebih dahulu.
```
**Solusi:** Jalankan `python train_model.py` terlebih dahulu.

### File terlalu besar
```
File terlalu besar. Maksimal ukuran file adalah 16MB.
```
**Solusi:** Kompres gambar atau gunakan gambar dengan resolusi lebih kecil.

### OpenCV error
Pastikan `opencv-python-headless` terinstall:
```bash
pip install opencv-python-headless
```

## 📚 Referensi

- **Scikit-learn Random Forest:** https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- **OpenCV HOGDescriptor:** https://docs.opencv.org/4.x/d5/d33/structcv_1_1HOGDescriptor.html
- **Dataset:** Garbage Classification on Kaggle

## 👨‍💻 Tim

**Universitas Bale Bandung**
- Praktikum Kecerdasan Buatan - Bab 8: Random Forest Algorithm

## 📄 License

Proyek akademik untuk tugas praktikum Kecerdasan Buatan.
