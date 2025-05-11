# Submission Akhir: Menyelesaikan Permasalahan Institusi Pendidikan ✨

## 📑 Deskripsi
Submission proyek menyelesaikan permasalahan Institusi Pendidikan ini adalah sebuah proyek akhir untuk penilaian praktik data science dari kelas Belajar Penerapan Data Science yang diberikan oleh Dicoding. Diharapkan dengan proyek ini dapat memberikan insight dan pemahaman mengenai data science.

## 🗂 Struktur Proyek
- `dataset/` : Folder yang menyimpan semua dataset yang digunakan dalam proyek.
  - `data_bersih.csv` : Berisi file CSV dari hasil proses cleaning.
  - `dataset_predict.csv` : Berisi file CSV yang ingin di prediksi oleh model.
  - `fitur_penting.csv` : Berisi file CSV berupa fitur-fitur penting yang memengaruhi Status Dropout.
  - `hasil_prediksi_model.csv` : Berisi file CSV dari hasil proses prediksi model.
- `model/` : Folder yang menyimpan semua model prediksi yang digunakan dalam proyek.
  - `joblib_model.pkl`: File model Random Forest Classifier yang disimpan oleh joblib dari hasil modeling di `notebook.ipynb`.
  - `pickle_model.pkl`: File model Random Forest Classifier yang disimpan oleh pickle dari hasil modeling di `notebook.ipynb`.
- `meakhelg-dashboard-1.jpg` : Halaman pertama dashboard.
- `meakhelg-dashboard-2.jpg` : Halaman kedua dashboard.
- `meakhelg-dashboard.pdf`: File dashboard yang telah saya buat untuk submission kali ini dalam bentuk PDF.
- `meakhelg-video.mkv`: Video penjelasan business dashboard yang telah dibuat dan kesimpulan dari dashboard tersebut.
- `metabase.db.mv.db`: File database dari Metabase.
- `app.py`: File yang digunakan untuk menjalankan prediksi machine learning berbasis Streamlit.
- `notebook.ipynb`: File yang digunakan untuk melakukan Data Understanding, EDA, hingga Modeling, Evaluasi, dan Konklusi.
- `README.md`: File dokumentasi.

## 📌 Business Understanding
### 🎯 Latar Belakang
**Jaya Jaya Institut** merupakan institusi pendidikan tinggi yang telah berdiri sejak tahun 2000 dan dikenal dengan reputasinya dalam mencetak lulusan berkualitas. Namun, dalam beberapa tahun terakhir, institusi ini menghadapi tantangan serius yaitu **tingginya angka siswa dropout** atau tidak menyelesaikan pendidikan mereka.    
Fenomena ini bukan hanya berdampak pada reputasi akademik institusi, tetapi juga pada efisiensi operasional, alokasi sumber daya pengajaran, dan keberhasilan program pendidikan jangka panjang. Jika tidak ditangani secara sistematis, masalah ini berpotensi merugikan baik dari sisi akademik maupun manajerial.

### ❗ Permasalahan Utama
Tingginya angka dropout mencerminkan kemungkinan adanya permasalahan struktural dan individual seperti:
* Ketidaksiapan akademik di perkuliahan yang ditunjukkan dari nilai rendah dan banyaknya mata kuliah tidak lulus,
* Masalah keuangan seperti keterlambatan pembayaran biaya pendidikan,
* Kurangnya motivasi atau ketidakcocokan dengan program studi yang dipilih,
* Faktor usia masuk dan latar belakang pendidikan sebelumnya,
* Tidak adanya sistem deteksi dini untuk memantau siswa yang berisiko dropout.
Permasalahan ini perlu ditangani dengan pendekatan berbasis data agar intervensi yang dilakukan dapat lebih cepat, tepat, dan berdampak nyata.

### 🎯 Tujuan Proyek Permasalahan Institusi Pendidikan
Proyek ini bertujuan untuk menyelesaikan permasalahan pendidikan yang dihadapi Jaya Jaya Institut dengan:
1. **Mengidentifikasi faktor-faktor utama** yang memengaruhi kemungkinan siswa mengalami dropout.
2. **Membangun model prediktif** berbasis machine learning untuk mendeteksi risiko dropout sedini mungkin.
3. **Menyediakan business dashboard** interaktif yang dapat digunakan oleh manajemen akademik untuk memantau performa siswa dan merancang strategi intervensi secara proaktif.

### 📌 Cakupan Analisis
Analisis difokuskan pada sejumlah aspek penting yang diyakini berkontribusi terhadap fenomena dropout, antara lain:
- **Demografis**: seperti Usia, Gender, Kewarganegaraan, Status Pernikahan, dan lain-lain yang dapat memengaruhi akademik.
- **Latar Belakang Pendidikan**: seperti Nilai-nilai sebelum masuk ke perguruan, Kualifikasi Orang Tua, Pekerjaan Orang Tua, Program Studi, hingga nilai-nilai lainnya.
- **Kinerja Akademik**: jumlah mata kuliah yang diselesaikan, nilai semester, IPK, hingga seorang yang pemegang beasiswa atau tidak.
- **Faktor Finansial dan Kompensasi**: keterlambatan atau ketidakpatuhan pembayaran biaya pendidikan, apakah seorang debtor atau tidak, hingga analisis Unemployment_rate, Inflation_rate, dan GDP.

### ⚙️ Strategi Analisis
- **Understanding and Exploratory Data Analysis (EDA)**: memahami pola data, melakukan visualisasi tren dropout, dan mengamati hubungan antara variabel.
- **Preprocessing Data**: meliputi pembersihan data, encoding fitur kategorikal, penyeimbangan data dengan SMOTE, serta pemilihan fitur relevan.
- **Pemodelan dan Evaluasi**: membandingkan sejumlah algoritma seperti Logistic Regression, Random Forest, XGBoost, SVM, dan KNN untuk menemukan model terbaik berdasarkan metrik klasifikasi seperti Accuracy, Precision, Recall, dan F1-Score.
- **Prediksi dan Visualisasi berbasis Streamlit**: memanfaatkan model terbaik untuk memprediksi risiko dropout dan membuat kerangka berbasis Streamlit yang bisa digunakan oleh semua orang.

## 🔧 Persiapan
### 💾 **Sumber Data**
Dataset yang digunakan dalam proyek ini adalah **[Dataset Mahasiswa Jaya Jaya Institut](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance)**, yang disediakan sesuai dengan instruksi pada submission proyek ini.

### 💻 **Menyiapkan Lingkungan (Environment Setup)**
Proyek ini memerlukan lingkungan yang sederhana untuk menjalankan **analisis data** dan **dashboard**. Berikut adalah langkah-langkah untuk mempersiapkan lingkungan kerja:

#### **1. Menjalankan notebook.ipynb**
- Pastikan **dependensi**, **paket**, dan **library** yang dibutuhkan telah tersedia. Lihat **file `requirements.txt`** untuk mengetahui daftar dependensi yang diperlukan.
- Jalankan seluruh isi **notebook.ipynb** di **Google Colab** atau IDE sejenis untuk melihat hasil dari **analisis data**, temuan, dan **insight** yang diperoleh.



#### **3. Menjalankan Dashboard**
Untuk melihat **dashboard** secara langsung, Anda dapat menggunakan **Metabase** dengan bantuan **Docker**. Pastikan **Docker** telah terpasang di sistem Anda.
**Langkah-langkah untuk menjalankan Metabase menggunakan Docker**:
1. **Tarik (pull) image Metabase dari Docker Hub** dengan perintah:
   ```bash
   docker pull metabase/metabase:latest
   ```

2. **Jalankan container Metabase** dengan perintah:
   ```bash
   docker run -p 3000:3000 --name metabase metabase/metabase
   ```

3. **Login ke Metabase** menggunakan kredensial berikut:
   - **Username**: `root@mail.com`
   - **Password**: `root123`

Dengan mengikuti langkah-langkah ini, Anda dapat memulai **analisis data** dan **dashboard**, serta melihat hasil visualisasi langsung di **Metabase**.

## 📊 Business Dashboard
### Ringkasan Dashboard
Dashboard ini menyajikan analisis terhadap **4.424 mahasiswa** dari sebuah institusi pendidikan untuk memahami dan memitigasi penyebab mahasiswa **dropout**. Sebanyak **1.421 mahasiswa** telah keluar (dropout), dan model prediktif mengidentifikasi **521 mahasiswa lain berisiko tinggi** dropout (dengan probabilitas > 0,8). Hasil ini mendukung **901 prediksi dropout** dari model machine learning yang dibangun.

### Faktor Utama Dropout Mahasiswa
Terdapat 10 fitur paling berpengaruh terhadap kemungkinan mahasiswa dropout. Faktor yang paling dominan adalah **nilai mata kuliah semester 2** (25,92%), disusul **nilai semester 1** (14,4%), dan **jumlah SKS semester 2** (12,18%). Faktor lainnya mencakup **nilai saat masuk (admission grade)**, **usia saat masuk**, dan **riwayat akademik sebelumnya**. Hal ini menunjukkan bahwa performa akademik di awal studi menjadi indikator kuat untuk mendeteksi potensi dropout.

### Distribusi dan Pola Dropout
* **49,9% mahasiswa lulus**, **32,1% dropout**, dan **17,9% masih aktif**.
* Mahasiswa dengan **nilai akademik rendah di semester awal** cenderung mengalami dropout.
* **88,1% mahasiswa membayar uang kuliah tepat waktu**, dan mereka memiliki kemungkinan dropout lebih rendah dibandingkan dengan yang tidak membayar.
* **Usia saat masuk kuliah** juga berpengaruh: mahasiswa yang lebih tua (di atas 23 tahun) menunjukkan tren dropout lebih tinggi.
* Nilai saat pendaftaran (admission grade) dan nilai kualifikasi sebelumnya juga memperlihatkan korelasi terhadap tingkat keberhasilan studi.

### Mode Aplikasi dan Risiko Dropout
Mahasiswa yang masuk melalui jalur **2nd Phase General Contingent** dan **Over 23 Years Old** memiliki tingkat dropout tertinggi, mengindikasikan bahwa mode aplikasi tertentu cenderung memiliki risiko lebih tinggi terhadap kegagalan studi. Sementara itu, jalur seperti **Technological Specialization Diploma Holders** dan **International Student (Bachelor)** memiliki risiko lebih rendah.

### Tren Risiko Akademik
Visualisasi tren risiko memperlihatkan bahwa semakin tinggi nilai pada **Curricular Units Semester 1 & 2**, maka **probabilitas dropout menurun**. Begitu pula dengan **admission grade**—mahasiswa dengan nilai awal yang tinggi memiliki risiko lebih rendah. Ini memperkuat pentingnya **intervensi dini di awal semester** untuk mencegah dropout.

### Prediksi Model
Model prediksi mampu mengidentifikasi mahasiswa dengan risiko tinggi dropout. Tabel prediksi menampilkan mahasiswa dengan kombinasi karakteristik seperti **usia lebih dari 23 tahun**, **nilai akademik rendah**, dan **nilai admission di bawah rata-rata**. Mereka menjadi prioritas utama untuk diberikan bimbingan, dukungan akademik, atau intervensi administratif agar dapat menyelesaikan studinya.

## Menjalankan Sistem Machine Learning
Jelaskan cara menjalankan protoype sistem machine learning yang telah dibuat. Selain itu, sertakan juga link untuk mengakses prototype tersebut.

```

```

## 🔍 Conclusion
### 🎯 Temuan Utama
- **Curricular Units (1st & 2nd) Semester Approved dan Grade** adalah fitur-fitur yang paling berpengaruh yang dimana mahasiswa yang **tidak menyelesaikan atau gagal** pada sejumlah besar mata kuliah serta dibarengi dengan **nilai akademik** yang rendah pada semester pertama dan kedua memiliki risiko dropout yang sangat tinggi. 
- **Admission Grade & Previous Qualification Grade** yang rendah juga cenderung lebih berisiko mengalami dropout yang menunjukkan bahwa **seleksi masuk** dan kesiapan awal mahasiswa penting untuk kesuksesan akademik jangka panjang.
- Faktor-faktor lain seperti **Age at Enrollment**, **Tuition Fees Up to Date**, bahkan **Debtor** juga berpengaruh meskipun tidak terlalu kuat.

### 🧠 Model Terbaik
Model **Random Forest** memberikan performa terbaik:
- Accuracy: **84.07%**
- Precision: **85.55%**
- Recall: **90.51%**
- F1-Score: **87.96%**
Model ini menunjukkan **akurasi dan sensitivitas yang sangat baik**, khususnya dalam mendeteksi mahasiswa yang berisiko dropout, menjadikannya alat prediksi yang handal untuk intervensi dini.

### 📊 Feature Importance (Top 8)
1. `Curricular_units_2nd_sem_approved` – 25.92%
2. `Curricular_units_1st_sem_approved` – 14.4%
3. `Curricular_units_2nd_sem_grade` – 12.18%
4. `Curricular_units_1st_sem_grade` – 8.6%
5. `Admission_grade` – 7.66%
6. `Age_at_enrollment` – 7.26% 
7. `Previous_qualification_grade` – 6.68%
8. `Tuition_fees_up_to_date` – 5.95%

### ✍️ Rekomendasi Action Items untuk Mengurangi Dropout Mahasiswa:
1. **Program Intervensi Dini untuk Mahasiswa Berisiko**    
   Gunakan model prediktif untuk **mengidentifikasi mahasiswa** dengan jumlah mata kuliah tidak lulus atau nilai rendah di semester awal, dan lakukan **pembinaan akademik atau mentoring personal** sejak dini.

2. **Pendampingan Akademik dan Tutor Sebaya**    
   Mahasiswa dengan nilai rendah dapat diberikan akses ke **tutor sebaya**, **kelas remedial**, atau **konseling belajar** untuk membantu mereka memperbaiki performa akademik.

3. **Perbaiki Proses Seleksi Masuk dan Orientasi**    
   Tingkatkan kualitas proses penerimaan mahasiswa agar hanya mahasiswa dengan kesiapan akademik yang baik diterima. Tambahkan **program orientasi dan pembekalan awal** agar mahasiswa lebih siap secara mental dan akademik.

4. **Bantuan Finansial dan Monitoring Pembayaran**    
   Mahasiswa dengan keterlambatan pembayaran perlu difasilitasi dengan **beasiswa darurat**, **konsultasi keuangan**, atau **penjadwalan ulang pembayaran** untuk mencegah dropout akibat kendala biaya.

5. **Perhatikan Profil Usia Masuk**    
   Sediakan program adaptasi khusus untuk mahasiswa dengan usia non-konvensional, misalnya yang masuk setelah bekerja atau baru lulus sekolah, agar mereka bisa menyesuaikan diri dengan lingkungan akademik.

### 🧾 **Kesimpulan**
Dengan memahami faktor-faktor utama yang menyebabkan **dropout mahasiswa** dan menerapkan **model prediktif berbasis Random Forest**, institusi pendidikan dapat melakukan **intervensi lebih awal dan tepat sasaran**. Tindakan strategis berdasarkan data ini dapat secara signifikan menurunkan tingkat dropout dan meningkatkan keberhasilan studi mahasiswa secara keseluruhan.
