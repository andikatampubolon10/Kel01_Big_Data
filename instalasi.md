# Panduan Instalasi dan Menjalankan Proyek

Selamat datang di proyek Analisis Retail & Segmentasi Pelanggan (Big Data). Dokumen ini dibuat untuk memandu pengguna awal agar dapat mengatur lingkungan kerja (*environment*) dan menjalankan proyek ini di komputer lokal dengan lancar.

---

## 📋 1. Prasyarat (Prerequisites)

Sebelum memulai, pastikan perangkat Anda sudah terinstal perangkat lunak berikut:

1. **Python (Versi 3.8 atau lebih baru)**
   - Anda bisa mengunduhnya di [python.org](https://www.python.org/downloads/).
   - **Penting untuk Windows**: Saat instalasi, pastikan Anda mencentang opsi **"Add Python to PATH"**.
2. **Java Development Kit (JDK) 8 atau 11**
   - Apache Spark (PySpark) membutuhkan Java untuk dapat berjalan.
   - Unduh dan instal JDK (disarankan versi 11). Pastikan Anda telah mengatur `JAVA_HOME` di *Environment Variables* Windows Anda.
3. **Git (Opsional namun disarankan)**
   - Untuk melakukan *clone* repositori.

---

## 🚀 2. Mengunduh Proyek (Clone Repository)

Buka terminal atau Command Prompt (CMD), lalu jalankan perintah berikut untuk mengunduh kode proyek ke komputer Anda:

```bash
git clone https://github.com/andikatampubolon10/Kel01_Big_Data.git
cd Kel01_Big_Data
```
*(Jika Anda mengunduh proyek dalam bentuk .zip, cukup ekstrak file zip tersebut dan buka foldernya di terminal).*

---

## 📦 3. Setup Virtual Environment & Instalasi Library

Sangat disarankan untuk menggunakan *Virtual Environment* agar instalasi pustaka (*library*) proyek ini tidak mengganggu proyek Python Anda yang lain.

**Membuat Virtual Environment:**
```bash
python -m venv venv
```

**Mengaktifkan Virtual Environment:**
- Di **Windows** (Command Prompt):
  ```cmd
  venv\Scripts\activate.bat
  ```
- Di **Windows** (PowerShell):
  ```powershell
  venv\Scripts\Activate.ps1
  ```
- Di **Mac/Linux**:
  ```bash
  source venv/bin/activate
  ```

**Menginstal Semua Library yang Dibutuhkan:**
Pastikan terminal Anda sudah berada di dalam folder proyek, lalu jalankan:
```bash
pip install pyspark streamlit pandas scikit-learn pymongo python-dotenv matplotlib seaborn psycopg2-binary
```

---

## ⚙️ 4. Konfigurasi Environment Variables (.env)

Proyek ini menggunakan koneksi ke database eksternal (PostgreSQL/Supabase dan MongoDB Atlas). Anda perlu membuat file konfigurasi `.env`.

1. Di dalam folder utama proyek (sejajar dengan `run_pipeline.py`), buat sebuah file baru bernama `.env`.
2. Buka file `.env` tersebut menggunakan teks editor (Notepad, VSCode, dsb) dan masukkan kredensial database Anda seperti format berikut:

```env
MONGO_URI=mongodb+srv://hris_admin:HRISAdmin2026@sotardok.phjkfd2.mongodb.net/?retryWrites=true&w=majority&appName=Sotardok
MONGO_DB=big_data

PG_HOST=db.shtiwpgrpkrqlgeocjes.supabase.co
PG_PORT=5432
PG_DB=postgres
PG_USER=postgres
PG_PASSWORD=parlinggoman10
```

*(Catatan: Ganti `<...>` dengan kredensial asli database milik kelompok/proyek Anda).*

> **⚠️ PENTING (Khusus MongoDB Atlas):**
> Jika Anda menggunakan MongoDB Atlas, pastikan IP Address internet Anda sudah masuk ke *Whitelist*. 
> Buka akun MongoDB Atlas Anda -> Menu **Security** -> **Network Access** -> Tambahkan IP Anda atau pilih **"Allow Access from Anywhere"** (`0.0.0.0/0`). Jika tidak, aplikasi akan mengalami *Timeout Error*.

---

## 🏃‍♂️ 5. Cara Menjalankan Pipeline Data (PySpark)

File `run_pipeline.py` berfungsi untuk mengeksekusi proses ETL (Extract, Transform, Load) menggunakan Apache Spark. Proses ini akan mengambil data mentah, membersihkannya, memproses *machine learning*, dan menyimpannya ke Database.

Jalankan perintah ini di terminal:
```bash
python run_pipeline.py
```

Tunggu hingga proses selesai. Anda akan melihat log proses seperti "Membaca data dari CSV...", "Melakukan data cleaning...", hingga "Menyimpan hasil data preprocessed ke CSV...".

---

## 📊 6. Cara Menjalankan Dashboard (Streamlit)

Setelah pipeline berjalan dan/atau Anda ingin melihat hasil visualisasi datanya, Anda dapat menjalankan Dashboard interaktif yang dibangun menggunakan Streamlit.

Jalankan perintah berikut di terminal:
```bash
streamlit run app_dashboard.py
```

Secara otomatis, browser web Anda akan terbuka dan mengarah ke alamat `http://localhost:8501`. Di sana, Anda bisa melihat visualisasi analisis tren penjualan, clustering segmentasi pelanggan, hingga tingkat retur pelanggan!

---

## 🛠️ Troubleshooting (Masalah yang Sering Terjadi)

1. **Error "Python was not found" atau "pip is not recognized"**:
   - Pastikan Anda sudah mencentang opsi "Add Python to PATH" saat menginstal Python. Anda mungkin perlu merestart terminal/komputer.
2. **Error "Java gateway process exited before sending its port number" (PySpark)**:
   - Ini berarti Java belum terinstal dengan benar atau `JAVA_HOME` belum diatur. Instal Java versi 11 dan atur *Environment Variables* di Windows Anda.
3. **Dashboard gagal memuat data MongoDB (Timeout)**:
   - Ingat untuk mendaftarkan (*whitelist*) IP Address koneksi internet Anda di dashboard MongoDB Atlas seperti pada Langkah 4.

Selamat bereksplorasi dengan Analisis Big Data! 🚀
