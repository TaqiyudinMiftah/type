# SKAID: pemeriksaan batas C pada SVM

Ditulis 14 November 2026 sebelum kandidat C tambahan dihitung. Hasil
SVM pertama telah diketahui: 241/340 pilihan RBF normal berada pada
C=100, batas atas grid. Studi lanjutan ini eksploratif pada cohort yang
sudah diperiksa, bukan konfirmasi pada test baru yang buta.

Pertanyaan: apakah keterbatasan grid C menjelaskan ketertinggalan SVM
terhadap Ridge, dan apakah kenaikan kecil MFCC RBF bertahan ketika C
dipilih dari rentang lebih luas? Laporkan linear dan RBF secara terpisah.
Tidak memilih kebijakan/family terbaik berdasarkan query held ID.

## Masukan dan biaya

Pakai seluruh masukan pilot/SVM terdahulu tanpa perubahan: 17 ID,
12 huruf, crop 80 ms dengan timestamp/filter log, query tetap 60/ID
Phase 2, dukungan Phase 1 nested K=1,2,3,5,10, tiga draw dan delapan
encoder beku. Kontrol K10 memakai permutasi support-label persis dari
pilot. Semua tahap memakai CPU; tidak ada ekstraksi fitur neural baru.

Label target tetap 12×K per model. Label sumber unik per fold persis
studi sebelumnya: union dukungan tiga draw pada 16 ID lain dan 960
query sumber. K10 memerlukan 5.749–5.807 label sumber, terpisah dari
120 label classifier target. Grid tambahan dan dua kebijakan memakai
ulang label yang sama. ResNet pralatih tetap memakai 720 label Harrison;
supervisi pralatih BEATs/log/filter/query evaluasi bukan nol biaya.

Pertahankan model, protokol dan hasil sebelumnya; pin seluruh hash.
Simpan hanya kolom kandidat baru dan referensi ke kolom lama agar tidak
menduplikasi cache. Union grid lama+baru adalah ruang pemilihan baru.

## Grid dan solver

Tambahkan C = 1.000, 10.000, 100.000. Dua normalisasi tetap: row L2,
atau support z-score lalu L2, memakai mean/population SD dukungan saja
dan SD≤1e-12 diganti 1. RBF tetap memiliki multiplier g = 0,01; 0,1;
1; 10; 100 dan gamma=g/median jarak kuadrat pasangan dukungan setelah
normalisasi, fallback gamma=g jika median≤1e-12. Tidak mengubah g/grid
normalisasi dari hasil target.

Grid lama 60 resep dipertahankan berurutan. Append enam resep linear
(normalisasi × C baru), lalu 30 resep RBF (normalisasi × g × C baru).
Total 96 resep: 16 linear dan 80 RBF. Ada 88.128 kandidat baru pada
2.448 kondisi; 146.880 kandidat lama dipakai ulang secara kanonik,
sehingga total ruang validasi 235.008 kandidat.

SVC sklearn 1.9.1 float64, kernel precomputed, class_weight=None,
probability nonaktif, shrinking=False, tol=1e-6, decision_function_shape
='ovr', break_ties=True, cache_size=128. max_iter=2.000.000 ditetapkan
sebelum fit baru, lebih tinggi dari batas 100.000 sebelumnya. Batas
iterasi tidak memperbarui fit lama yang telah konvergen. Replay dua
anchor per kondisi (row L2, C100, linear dan RBF g0,1) dengan batas
baru untuk memeriksa identitas skor, koefisien/status dan urutan kelas.
Replay cap lama pada anchor memeriksa parameter yang tidak disimpan di
cache kandidat lama. Total 4.896 pasangan anchor bukan kandidat grid
tambahan atau label tambahan.

Jika fit_status_≠0, parameter/skor tidak finite, atau urutan kelas
berubah, tulis identitas kegagalan dan hentikan tahap. Jangan menghapus
resep/kondisi gagal dari kriteria atau menyebut studi lengkap. Tidak
menaikkan cap secara adaptif atau menyetel tol dari query. Catat semua
n_iter_ dan fit_status_, termasuk yang mendekati cap.

## Seleksi dan evaluasi

Validasi setiap ID sebagai data pengembangan untuk 16 fold lainnya.
Satu resep per kebijakan/family/K/held ID bersama bagi semua seed/draw.
Buang seluruh baris held ID sebelum fungsi seleksi menerima kriteria.
Kriteria adalah integer jumlah benar, dengan kontribusi jumlah query
dan predictor sama per ID. Seri: C lebih kecil, lalu g lebih kecil,
lalu row L2, sama seperti sebelumnya. Kontrol memiliki tabel terpisah
dan fleksibilitas seleksi sama. Tidak memakai hasil target untuk
memilih antara resep lama dan baru.

Kunci seluruh 816 pilihan dan hash validasi sebelum 4.896 refit luar.
Refit memakai dukungan held ID sendiri saja. Simpan koefisien dual,
intercept, support-vector positions/counts, statistik scaler/bandwidth,
skor query/dukungan, label/index, dan metrik setiap percobaan. Jangan
mengubah pilihan dari hasil query, direct-kernel audit, atau fitur CPU.
Periksa identitas refit dengan kandidat sumber yang resepnya sama.

## Audit dan pelaporan

Audit 88.128 kandidat baru melalui kernel linear/RBF langsung; audit
146.880 kandidat lama dipakai ulang dengan verifikasi hash/coverage.
Toleransi perbandingan skor tetap atol=1e-5, rtol=1e-5. Laporkan semua
outlier serta perubahan prediction/support-count; jangan menghilangkan
outlier numerik yang diketahui pada model lama. Bandingkan scaler dan
median secara independen, hitungan sumber dan solver convergence.

Audit 816 pilihan dengan aturan seri/penjumlahan independen, union biaya
label dan counterfactual mengubah hanya baris held. Rekonstruksi semua
4.896 kepala final dari dual/kernel tanpa SVC.predict/decision_function.
Refit semua kepala terpilih dengan cache fitur CPU yang sudah diaudit,
resep tetap, dan laporkan selisih skor/prediksi/akurasi per kondisi.
Gabungkan count direct-kernel lama+baru untuk pemeriksaan sensitivitas
pilihan; alternatif diagnostik tidak diterapkan ke model utama.

Laporkan kurva seluruh K untuk kedua kebijakan dibandingkan SVM lama
dan Ridge. Mean antar 17 ID setelah rata-rata draw/model dalam ID,
bukan ensemble. Delta berpasangan dan bootstrap 10.000 ID seed 7471,
bersyarat pada cohort/prosedur; fold berbagi sumber. Laporkan CI, semua
ID/kontrol, fit dukungan, biaya sumber, pilihan batas C, berapa pilihan
berubah, dan threshold 80/90% pada grid tetap. Threshold bukan target
untuk menyaring data. Tidak memperluas grid lagi setelah hasil luar.

Jika C tertinggi tetap dominan, kesimpulan hanya berlaku pada grid ini,
bukan optimum universal. Penelitian masih 12 huruf, segmentasi berbasis
log dan cohort yang telah diperiksa; belum mikrofon/sesi baru, decoding
teks bebas, reproduksi literal paper atau klaim SOTA. Simpan laporan,
grafik, CSV per-ID/pilihan/prediksi dan snapshot ringan.

Dokumentasi primer: [SVC sklearn 1.9.1](https://scikit-learn.org/1.9/modules/generated/sklearn.svm.SVC.html).
Analisis numerik sebelumnya dipertahankan pada studi SVM awal.
