# Prediction Response Customer for Marketing Campaign 
Pada repository ini merupakan pengembangan supervised machine learning yang berfungsi untuk memprediksi response customer ketika dilakukan campaign untuk berlangganan deposito berjangka. Tahapan yang dilakukan ketika membangun model ini adalah
1. Load Library
2. EDA
   - Data Exploration
   - Univariate Analysis
   - Bivariate Analysis
   - Multivariate Analysis
4. Data preprocessing dan Feature Engineering
5. Modelling
6. Evaluation Model
7. Business Recomendation

## Introduction
Suatu bank XYZ saat ini memiliki produk Depostio Berjangka. Deposito berjangka adalah suatu deposito yang ditawarkan oleh bank atau lembaga keuangan dengan tingkat bunga tetap (seringkali lebih baik daripada hanya membuka rekening deposito) di mana uang Anda akan dikembalikan pada waktu jatuh tempo tertentu. Bank tersebut telah melakukan marketing campaign untuk produk tersebut. Namun mendapatkan hasil yang kurang memuaskan, dimana terjadi sekitar 53 % dari hasil seluruh marketing campaign, customer menolak untuk berlangganan deposito berjangka. Angka ini tentunya tidak memuaskan untuk pihak management bank. Sehingga pihak bank menginginkan untuk mengetahui faktor-faktor apa aja yang berpengaruh besar dalam marketing campaign dan menginginkan meningkatkan Success rate of marketing campaign product depostio berjangka.

## Goals
1. Mengetahui faktor-faktor apa aja yang berpengaruh besar dalam marketing campaign
2. Meningkatkan Success rate of marketing campaign product depostio berjangka.

## Data Understanding
### Fiture Data Customer Bank:
1 - age (numeric) : Umur customer <br>
2 - job : Tipe dari pekerjaan Customer <br>
3 - marital : Status perkawinan Customer <br>
4 - education : Pendidikan Customer <br>
5 - default: Memiliki kredit secara default? <br>
6 - balance: Simpanan atau saldo yang dimiliki oleh customer. <br>
7 - housing: Memiliki pinjaman rumah? <br>
8 - loan: Memiliki pinjaman pribadi? <br>
### Fiture yang berhubungan dengan campaign terakhir dilakukan
9 - contact: Type contact komunikasi <br>
10 - month: Bulan terakhir dihubungi <br>
11 - day: Hari terakhir ketika dihubungi <br>
12 - duration: Durasi terakhir ketika dihubungi dalam detik. <br>

### Fiture lain:
13 - campaign: jumlah kontak yang dilakukan selama kampanye ini.
14 - pdays: jumlah hari yang berlalu setelah klien terakhir dihubungi dari kampanye sebelumnya
15 - previous: jumlah kontak yang dilakukan sebelum kampanye ini dan untuk klien ini
16 - poutcome: hasil dari kampanye pemasaran sebelumnya

### Target
17 response: target atau response customer dari marketing campaign

## EDA
1. Distribusi target cukup seimbang, namun hasil yang diperoleh adalah 53 % customer menolak berlangganan deposito berjangka, sisanya berlangganan deposito berjangka.
2. Karakteristik customer bank XYZ adalah seorang pekerja yang berumur 32 hingga 49 yang sudah menikah, dan memiliki tingkat pendidikan secondary, memiliki penghasilan dengan kategori tinggi dan secara umum tidak memiliki kredit, pinjaman rumah maupun hutang.
3. Terlihat ada kecenderungan semakin pendek call duration, maka akan semakin tinggi kemungkinan untuk menolak berlanggana deposito berjangka.
4. Campaign akan cenderung di terima pada seorang pelajar, status perkawinannya single, tingkat pendidikannya tertiary, secara default tidak memiliki kredit, tidak memiliki pinjaman rumah, tidak memiliki pinjaman, dihubungi melalui celluar dan hasil campaign sebelumnya sukses.

## Data Preprocessing dan Feature Engineering
Sebelum melakukan modeling machine learning, terdapat tahapan yang perlu dilakukan yaitu data preprocessing dan feature engineering. Tujuannya tentu untuk mendapatkan hasil model yang baik untuk melakukan prediksi. Berikut merupakan tahapan-tahapan yang dilakukan.

### Data Preprocessing
1. Handle invalid data values.
2. Handle invalid data types.
3. Melakukan transformasi pada feature-feature numerik yang distribusinya skew positif menggunakan log1p.
4. Masking outlier menggunakan nilai IQR (Interquartile Range).

### Feature Engineering
1. Feature Encoding pada feature defaults, housing, loan, education dan target Response.
2. One-Hot Encoding pada feature job, marital, contact, poutcome, month.

### Split Training dan Testing set
1. Melakukan split training dan testing dengan proporsi 70:30. 70 Untuk training set dan 30 untuk testing set.
2. Menggunakan random state bernilai 42.

## Modelling dan Evaluation Model.
### Modeling
Disini dilakukan percobaan modeling menggunakan beberapa jenis algoritma. Selanjutnya karena kita memiliki target seimbang, dan kita ingin meningkatkan kualitas prediksi, yaitu dengan meminimalisir false negatif yang terjadi dimana false positif terjadi ketika Actual "Yes" atau berlangganan namun model memprediksi "No" tidak berlangganan maka kita akan menggunakan evaluation score **Recall**. Berikut hasil modeling yang telah dilakukan.

1. Logistic Regression <br>
   Training       | Testing <br>
   ------------------------------- <br>
   Akurasi   82%  | Akurasi   81% <br>
   Precision 81%  | Precision 80% <br>
   Recall    83%  | Recall    82% <br>
   F1-Score  82%  | F1-Score  81% <br>

2. Random Forest Classifier <br>
   Training        | Testing <br>
   ------------------------------- <br>
   Akurasi   100%  | Akurasi   86% <br>
   Precision 100%  | Precision 82% <br>
   Recall    100%  | Recall    90% <br>
   F1-Score  100%  | F1-Score  86% <br>

3. Adaboost Classifier <br> 
   Training       | Testing <br>
   ------------------------------- <br>
   Akurasi   82%  | Akurasi   82% <br>
   Precision 83%  | Precision 82% <br>
   Recall    80%  | Recall    81% <br>
   F1-Score  82%  | F1-Score  82% <br>

4. Gradient Boosting Classifier <br>
   Training       | Testing <br>
   ------------------------------- <br>
   Akurasi   85%  | Akurasi   84% <br>
   Precision 84%  | Precision 82% <br>
   Recall    87%  | Recall    87% <br>
   F1-Score  85%  | F1-Score  85% <br>

### Evaluation Score
Berdasarkan hasil evaluation score di atas kita ketahui bahwa model Gradient Boosting yang memiliki tingkat Recall tertinggi sebesar 87% Selain itu, kalau dapat kita lihat model ini tidak memiliki rentang evaluation score recall pada training dan testing score, sehingga dapat kita simpulkan model ini sangat baik karena tidak terjadi underfitting maupun overfitting. Oleh karena itu, dipilih model Gradient Boosting Classifier untuk melakukan prediksi marketing campaign terhadap customer untuk berlangganan deposito berjangka

### Feature Importance
Berdasarkan feature importance menurut model Gradient Boosting feature-feature yang paling berperan besar untuk berlangganan deposito berjangka dapat kita bagi menjadi dua yaitu
1. Fitur yang berhubungan dengan campaign terakhir dilakukan
  a. Duration : durasi call yang dilakukan sales dengan customer 
  b. Success  : hasil campaign dari sebelumnya
  c. Pdays    : jumlah hari yang berlalu setelah klien terakhir dihubungi dari kampanye sebelumnya
2. Fitur yang berhubungan dengan data customer
  a. Housing  : Apakah mempunyai pinjaman rumah (Kontrak) atau tidak?
  b. Age      : Umur customer

## Business Recomendation
Bisnis recommendation yang disarankan adalah ada usaha pencegahan dan usaha penanggulangan
1. Usaha Pencegahan
   a. Melakukan improvement pelatihan untuk sales/agent bank XYZ dengan tim pelatihan yang tersedia agar dapat melakukan call duration di atas 4 menit dan menawarkan product secara lebih baik dan menarik.
   b. Jika hasil campaign secara umum yang telah dilakukan sukses, maka dapat dilanjutkan untuk melakukan campaign pada customer-customer berikutnya dengan metode campaign yang sama.
   c. Melakukan campaign pada customer yang memilki rumah (tidak kontrak) dan berumur pelajar kisaran 18 - 30 karena berdasarkan hasil EDA para pelajar lebih cenderung untuk berlangganan depisto.

2. Usaha penanggulangan
   Dengan machine learning tentunya kita dapat memprediksi hasil campaign yang akan terjadi, sehingga jika customer diprediksi menolak berlangganan deposito berjangka kita dapat mempersiapkan terlebih dahulu dengan penawaran yang lebih menarik dan memperkejakan senior sales untuk menangani customer tersebut.

Seluruh bisnis rekomendasi ini tidak memerlukan high cost untuk melakukan improvement, karena improvement bisa dilakukan dari pihak internal sendiri.


## Notes
1. Hasil analisa lebih lengkap dapat dilihat file Prediction Response Customer for Marketing Campaign.ipynb
2. Dataset yang digunakan diperoleh dari https://www.kaggle.com/janiobachmann/bank-marketing-campaign-opening-a-term-deposit.
3. gradient_boosting_model.pkl merupakan file pickle yang akan digunakan untuk proses deployment.
