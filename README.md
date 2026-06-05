# Klasifikasi Kualitas Jeruk Menggunakan RGB, GLCM, dan KNN

## Anggota Kelompok

1. Resa Aulia Amanda (152024128)
2. Nura Mahavira (152024146)
3. Sindi Salwa Fauziah (152024153)

## Mata Kuliah

IFB 206 Komputasi Paralel

Dosen Pengampu: Lisa Kristiana, Ph.D.

## Deskripsi Proyek

Proyek ini merupakan sistem klasifikasi kualitas jeruk berbasis pengolahan citra digital menggunakan Python dan Flask. Sistem menganalisis warna dan tekstur kulit jeruk untuk menentukan kualitas buah ke dalam tiga kategori yaitu Baik, Sedang, dan Buruk.

## Latar Belakang

Penentuan kualitas jeruk secara manual sering kali bersifat subjektif dan membutuhkan waktu. Oleh karena itu dibuat sistem klasifikasi otomatis menggunakan metode pengolahan citra dan machine learning untuk membantu proses identifikasi kualitas jeruk.

## Tujuan

- Mengidentifikasi kualitas jeruk secara otomatis.
- Mengimplementasikan algoritma K-Nearest Neighbor (KNN).
- Menganalisis fitur warna RGB dan tekstur GLCM pada citra jeruk.

## Metode

### 1. Segmentasi Citra

Menggunakan ruang warna HSV untuk memisahkan objek jeruk dari latar belakang.

### 2. Ekstraksi Fitur RGB

Mengambil nilai rata-rata:
- Red
- Green
- Blue

### 3. Ekstraksi Fitur GLCM

Mengambil fitur tekstur:
- Contrast
- Correlation
- Energy
- Homogeneity

### 4. Klasifikasi KNN

Fitur yang diperoleh digunakan sebagai data masukan algoritma KNN untuk menentukan kategori kualitas jeruk.

## Alur Sistem

Upload Gambar
↓
Segmentasi HSV
↓
Ekstraksi RGB
↓
Ekstraksi GLCM
↓
Klasifikasi KNN
↓
Hasil Prediksi

## Dataset

Jumlah dataset:

- Baik : 50 gambar
- Sedang : 50 gambar
- Buruk : 50 gambar

Total: 150 gambar

## Teknologi

- Python
- Flask
- OpenCV
- NumPy
- Scikit-Learn
- Scikit-Image

## Hasil Pengujian

- Akurasi : (isi sesuai hasil program)
- Confusion Matrix : (tambahkan screenshot jika ada)

## Cara Menjalankan

bash
pip install flask opencv-python numpy scikit-image scikit-learn

python app.py

Buka browser:
text
http://127.0.0.1:5000

Upload gambar jeruk dan sistem akan menampilkan hasil klasifikasi beserta tingkat kepercayaannya (confidence).

## Struktur Folder

deteksi-jeruk/
├── dataset/
├── static/
├── templates/
├── app.py
├── evaluate.py
├── labels.csv
└── README.md

## Kesimpulan

Sistem berhasil mengklasifikasikan kualitas jeruk berdasarkan fitur warna RGB dan tekstur GLCM menggunakan algoritma KNN.
