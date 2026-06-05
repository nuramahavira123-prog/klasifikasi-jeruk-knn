from flask import Flask, render_template, request, url_for
import os
import cv2
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from skimage.feature import graycomatrix, graycoprops
from collections import defaultdict
import statistics

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


ORANYE_BAWAH = np.array([5, 80, 80])
ORANYE_ATAS = np.array([25, 255, 255])

HIJAU_BAWAH = np.array([30, 50, 50])
HIJAU_ATAS = np.array([90, 255, 255])

KUNING_BAWAH = np.array([20, 80, 80])
KUNING_ATAS = np.array([35, 255, 255])

# Segmentasi Jeruk
def buat_mask(hsv):

    m_oranye = cv2.inRange(hsv, ORANYE_BAWAH, ORANYE_ATAS)
    m_hijau = cv2.inRange(hsv, HIJAU_BAWAH, HIJAU_ATAS)
    m_kuning = cv2.inRange(hsv, KUNING_BAWAH, KUNING_ATAS)

    mask = cv2.bitwise_or(
        m_oranye,
        cv2.bitwise_or(m_hijau, m_kuning)
    )

    kernel = np.ones((7,7), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    mask = cv2.GaussianBlur(mask,(5,5),0)

    _, mask = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    return mask

# RGB
def ekstrak_rgb(img, mask):

    pixels = img[mask > 0]

    if len(pixels) == 0:
        return [0,0,0]

    b = np.mean(pixels[:,0])
    g = np.mean(pixels[:,1])
    r = np.mean(pixels[:,2])

    return [r,g,b]

# GLCM
def ekstrak_glcm(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(
        glcm,
        'contrast'
    )[0,0]

    homogeneity = graycoprops(
        glcm,
        'homogeneity'
    )[0,0]

    energy = graycoprops(
        glcm,
        'energy'
    )[0,0]

    correlation = graycoprops(
        glcm,
        'correlation'
    )[0,0]

    return [
        contrast,
        homogeneity,
        energy,
        correlation
    ]

# Fitur Lengkap
def ekstrak_fitur(path):

    img = cv2.imread(path)

    if img is None:
        return None

    img = cv2.resize(img,(400,300))

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    mask = buat_mask(hsv)

    # default: gunakan RGB + GLCM
    rgb = ekstrak_rgb(img, mask)
    glcm = ekstrak_glcm(img)

    fitur = rgb + glcm

    return fitur


def ekstrak_shape(path):
    img = cv2.imread(path)

    if img is None:
        return None

    img = cv2.resize(img, (400, 300))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = buat_mask(hsv)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # fallback: return zeros
        return [0.0, 0.0]

    # ambil kontur terbesar
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if perimeter == 0:
        circ = 0.0
    else:
        circ = 4 * np.pi * area / (perimeter ** 2)

    # normalisasi area terhadap ukuran citra
    area_norm = area / (img.shape[0] * img.shape[1])

    return [float(area_norm), float(circ)]

# Training Dataset
def build_dataset(mode="both"):

    X = []
    y = []

    classes = [
        "baik",
        "sedang",
        "buruk"
    ]

    for label in classes:

        folder = os.path.join(
            "dataset",
            label
        )

        if not os.path.exists(folder):
            continue

        for file in os.listdir(folder):

            path = os.path.join(
                folder,
                file
            )

            if mode == "shape":
                fitur = ekstrak_shape(path)
            else:
                fitur = ekstrak_fitur(path)

            if fitur is None:
                continue

            X.append(fitur)
            y.append(label)

    return np.array(X), np.array(y)

# Training KNN
print("Training dataset...")

# build default dataset (RGB+GLCM)
X, y = build_dataset(mode="both")

# default classifier (k=5)
knn = KNeighborsClassifier(n_neighbors=5)
if len(X) > 0:
    knn.fit(X, y)

print("Training selesai.")
print("Jumlah data:", len(X))


def evaluate_ks(X, y, ks=[1, 2, 3, 4, 5], n_iter=10):
    # Mengembalikan rata-rata akurasi overall dan per kelas untuk tiap k
    results = {}

    if len(X) == 0:
        return results

    sss = StratifiedShuffleSplit(n_splits=n_iter, test_size=0.3, random_state=42)

    labels = sorted(list(set(y)))

    for k in ks:
        overall_accs = []
        per_class_accs = {lab: [] for lab in labels}

        for train_idx, test_idx in sss.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf = KNeighborsClassifier(n_neighbors=k)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            overall_accs.append(accuracy_score(y_test, y_pred))

            # per-class
            for lab in labels:
                # select indices for this class in test
                inds = [i for i, yy in enumerate(y_test) if yy == lab]
                if not inds:
                    continue
                y_test_lab = [y_test[i] for i in inds]
                y_pred_lab = [y_pred[i] for i in inds]
                per_class_accs[lab].append(accuracy_score(y_test_lab, y_pred_lab))

        # rata-rata
        results[k] = {
            'overall': float(statistics.mean(overall_accs)) if overall_accs else 0.0,
            'per_class': {lab: float(statistics.mean(per_class_accs[lab])) if per_class_accs[lab] else 0.0 for lab in labels}
        }

    return results


# Precompute evaluation (k=1..5) with 10 iterations untuk kedua mode fitur
try:
    EVAL_BOTH = evaluate_ks(*build_dataset(mode="both"), ks=[1, 2, 3, 4, 5], n_iter=10)
except Exception:
    EVAL_BOTH = {}

try:
    EVAL_SHAPE = evaluate_ks(*build_dataset(mode="shape"), ks=[1, 2, 3, 4, 5], n_iter=10)
except Exception:
    EVAL_SHAPE = {}

DESKRIPSI_KELAS = {
    "baik": "Jeruk terdeteksi dengan warna oranye dominan dan fitur tekstur yang baik.",
    "sedang": "Jeruk terdeteksi dengan warna campuran atau tekstur sedang.",
    "buruk": "Jeruk terdeteksi dengan warna kurang merata atau kualitas rendah.",
}

# Flask
@app.route("/", methods=["GET","POST"])
def index():

    hasil = None
    confidence = None
    image_url = None
    deskripsi = None

    if request.method == "POST":

        file = request.files["gambar"]

        if file:

            save_path = os.path.join(
                UPLOAD_FOLDER,
                file.filename
            )

            file.save(save_path)

            # baca pilihan k dan mode fitur dari form
            k_choice = int(request.form.get("k", 5))
            shape_only = True if request.form.get("shape_only") is not None else False

            # ekstraksi fitur sesuai pilihan
            if shape_only:
                fitur = ekstrak_shape(save_path)
            else:
                fitur = ekstrak_fitur(save_path)

            # rebuild dataset & train dengan k terpilih agar prediksi konsisten
            X_train, y_train = build_dataset(mode=("shape" if shape_only else "both"))

            clf = KNeighborsClassifier(n_neighbors=k_choice)
            if len(X_train) > 0:
                clf.fit(X_train, y_train)

            # prediksi
            pred = clf.predict([fitur])[0]

            # confidence/probability jika tersedia
            confidence = None
            try:
                prob = clf.predict_proba([fitur])[0]
                idx = list(clf.classes_).index(pred)
                confidence = prob[idx] * 100
            except Exception:
                confidence = None

            eval_current = None
            if shape_only:
                eval_current = EVAL_SHAPE.get(k_choice)
            else:
                eval_current = EVAL_BOTH.get(k_choice)

            hasil = pred
            deskripsi = DESKRIPSI_KELAS.get(pred, "")
            image_url = url_for("static", filename=f"uploads/{file.filename}")
            selected_k = k_choice
            selected_mode = "shape" if shape_only else "both"

    return render_template(
        "index.html",
        hasil=hasil,
        confidence=confidence,
        image=image_url,
        deskripsi=deskripsi,
        eval_both=EVAL_BOTH,
        eval_shape=EVAL_SHAPE,
        selected_k=locals().get('selected_k'),
        selected_mode=locals().get('selected_mode'),
        eval_current=locals().get('eval_current'),
    )

# Run
if __name__ == "__main__":
    app.run(
        debug=True
    )