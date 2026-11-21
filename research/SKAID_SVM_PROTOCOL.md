# SKAID: SVM linear dan RBF dengan seleksi dari ID sumber

Ditulis 11 November 2026, setelah hasil pilot dan seleksi Ridge diketahui,
sebelum menghitung kandidat SVM pada data SKAID. Eksplorasi lanjutan pada
kohort yang sama; bukan test baru yang belum diperiksa. Bekukan dokumen,
implementasi dan masukan dengan SHA256 sebelum validasi sumber dimulai.

Pertanyaan: apakah perubahan loss linear atau kernel nonlinear memperbaiki
generalisasi dengan fitur dan budget target tetap? Bandingkan dua kebijakan
terpisah, SVM linear dan SVM RBF, terhadap Ridge terpilih yang sudah selesai.
Jangan memilih pemenang kebijakan memakai query held ID untuk kemudian
mengklaim hasil sebagai prediksi kebijakan tunggal yang ditetapkan sebelumnya.

## Data, supervisi, dan unit pelaporan

Pakai persis cache delapan encoder, 17 ID, 12 huruf, crop 80 ms, 60 query
Phase 2 per ID, K=1,2,3,5,10 dukungan nested Phase 1, draw 101/102/103,
dan tiga seed ResNet dari pilot. Tidak mengubah cohort, anotasi, crop,
fitur, backbone, atau query. K0 lama hanya pembanding yang dipakai ulang.
Semua perhitungan memakai CPU; tidak mengekstrak fitur neural baru.

Label classifier target per model tetap 12×K. Pemilihan resep memakai
union dukungan tiga draw dan 960 query pada 16 ID lain, persis budget
studi Ridge terpilih (K10: 5.749–5.807 label sumber unik per fold).
Kedua kebijakan/kandidat memakai ulang label itu, tidak menambah label
unik. ResNet pralatih tetap memakai 720 label Harrison dan BEATs tetap
memiliki supervisi pralatih generik. Biaya log/timestamp/filter dan label
query evaluasi di luar budget classifier. K bukan biaya anotasi end-to-end.

Mean dihitung antar 17 ID setelah rata-rata draw dan model dalam ID;
bukan ensemble. Bootstrap deskriptif 10.000 ID, seed 7471, sama seperti
sebelumnya. Fold berbagi sumber, sehingga CI bersyarat pada cohort dan
prosedur. Laporkan delta berpasangan terhadap Ridge terpilih serta RBF
minus linear, semua K, kontrol, variasi ID, dan batas grid. Ambang 80/90%
hanya ringkasan kurva; tidak mengubah cohort/grid jika belum tercapai.

## Model dan grid yang ditetapkan

Gunakan dua transformasi support-only dari studi Ridge: row L2, atau
z-score memakai mean/population SD dukungan lalu row L2. SD ≤1e-12
diganti 1. Tidak memakai query untuk statistik transformasi atau bandwidth.

- C = 0,01; 0,1; 1; 10; 100.
- Linear: kernel dot product, dua normalisasi × lima C = 10 resep.
- RBF: exp(-gamma × jarak Euclidean kuadrat), dengan gamma = g/m,
  m median seluruh pasangan dukungan i<j setelah transformasi. Jika
  m≤1e-12, gamma = g (fallback yang harus dihitung/dilaporkan).
  g = 0,01; 0,1; 1; 10; 100. Dua normalisasi × lima g × lima C = 50 resep.
- Total 60 resep × 2.448 kondisi = 146.880 kandidat baru.

SVC sklearn 1.9.1, float64, kernel precomputed, class_weight=None,
probability nonaktif (default efektif; parameter deprecated tidak diberikan),
shrinking=False, tol=1e-6, max_iter=100000,
decision_function_shape='ovr', break_ties=True, cache_size=128 MiB.
Training multiclass memakai pasangan kelas; skor OvR memuat vote dan
koreksi confidence. Prediksi adalah argmax skor ini, bukan probabilitas.
Jika fit_status_ bukan 0 atau hasil tidak finite, hentikan dan catat;
jangan menghilangkan percobaan yang gagal. Tidak menyetel tol memakai
query. Simpan gamma aktual, median dukungan, status dan iterasi solver.

## Seleksi dan urutan eksekusi

Validasi seluruh kandidat pada tiap ID sebagai data pengembangan untuk
16 fold lainnya. Untuk ID h, buang seluruh baris h sebelum fungsi seleksi
menerima skor. Tidak pooling statistik fitur/dukungannya ke sumber.
Pilih satu resep per kebijakan/keluarga/K/held ID, bersama bagi seluruh
seed/draw. Kriteria integer jumlah benar pada 16 ID sumber yang masing-
masing memiliki jumlah predictor dan query sama. Normal dan kontrol
memiliki tabel seleksi sendiri. Seri: C lebih kecil; lalu g lebih kecil
untuk RBF; lalu row L2. Tidak memperluas grid setelah evaluasi luar.

Kontrol K10 memakai permutasi label dukungan persis dari pilot, seimbang
per draw/ID, dengan label query benar. Kontrol mendapat fleksibilitas
seleksi sama. Dua kebijakan × 408 = 816 pilihan harus dikunci beserta
hash validasi sebelum 4.896 refit luar dimulai. Refit hanya dukungan held
ID, menggunakan pilihan yang terkunci. Jangan memakai hasil query held
ID atau replay CPU untuk memperbarui resep.

## Audit dan artefak

Simpan skor/count seluruh kandidat, statistik transformasi/bandwidth,
jumlah iterasi, tabel seleksi dan biaya label, parameter dual final,
urutan support vector, skor query/dukungan dan prediksi per contoh.
Verifikasi hash masukan, urutan kelas, split, keseimbangan dan nesting
sebelum dan sesudah tahap. Implementasi lama tidak diubah.

Audit seluruh 146.880 kandidat dengan SVC kernel linear/RBF langsung,
independen dari jalur precomputed. Periksa scaler dan median secara
independen; bandingkan skor dan argmax serta laporkan setiap variasi
numerik, jangan menyembunyikan selisih. Toleransi skor atol=1e-5,
rtol=1e-5; perbedaan prediction dicatat meskipun toleransi lolos.
Audit semua 816 pilihan dengan penjumlahan/tie rule independen dan
counterfactual baris held. Replay 4.896 kepala dari parameter dual dan
kernel, tanpa memanggil SVC.predict/decision_function. Refit seluruh
kepala pada cache fitur CPU pilot dengan resep tetap dan laporkan
perubahan prediksi/akurasi. Audit ini tidak memilih ulang resep.

Laporan mencakup kurva, biaya label, tabel semua pilihan dan per-ID,
kontrol negatif, support fit versus query, CI bersyarat, perbandingan
dengan Ridge, audit dan snapshot. Hindari klaim SOTA, reproduksi literal,
mikrofon/sesi baru, 26 huruf, atau sistem tanpa timestamp.

Dokumentasi primer API yang diperiksa:
[SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
dan [kernel SVM](https://scikit-learn.org/stable/modules/svm.html#svm-kernels).
Median bandwidth di atas adalah aturan eksperimen ini.
