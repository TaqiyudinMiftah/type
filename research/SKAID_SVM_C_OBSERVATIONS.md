# SKAID: hasil pemeriksaan batas C pada SVM

14 November 2026. **MFCC+SVM linear naik dari 65,49% ke 76,63% pada
K10** setelah perluasan C; kenaikan terjadi pada semua 17 ID. Hasil
MFCC+RBF hanya berubah dari 76,73% ke 76,83%. Keduanya mendekati
MFCC+Ridge 75,78%, tanpa bukti keunggulan yang meyakinkan.

## Rancangan

[Protokol](SKAID_SVM_C_PROTOCOL.md) ditetapkan sebelum kandidat baru
dihitung, setelah studi awal diketahui. Tetap 17 ID, 12 huruf, crop 80 ms
dengan timestamp acuan, 60 query Phase 2 per ID, K=1,2,3,5,10 contoh
Phase 1, tiga draw dan delapan encoder beku. Semua tahap memakai CPU.
Tidak ada pengumpulan data atau ekstraksi neural baru.

C=1.000, 10.000, 100.000 menambah 36 resep pada grid 60 resep. Ada
88.128 kandidat baru, 146.880 kandidat lama dipakai ulang persis,
235.008 total. Dua replay C=100 per kondisi menghasilkan 4.896
pasangan anchor yang identik pada kedua cap iterasi, termasuk
koefisien, skor, support vectors dan hitungan iterasi.

Cap baru 2.000.000 iterasi ditetapkan sebelum fit. Empat kandidat
memerlukan lebih dari cap lama 100.000; maksimum 234.625 iterasi biner.
Seluruhnya konvergen. Semua 816 pilihan dikunci sebelum 4.896 refit
luar. Setiap pilihan hanya menerima count dari 16 ID selain ID yang
diuji, bersama untuk seed/draw, normal dan kontrol terpisah.

## Hasil K10

| Fitur | Ridge | SVM linear lama → baru | SVM RBF lama → baru |
|---|---:|---:|---:|
| MFCC | 75,78% | 65,49% → 76,63% | 76,73% → 76,83% |
| BEATs | 69,80% | 50,92% → 51,67% | 50,78% → 51,86% |
| ResNet pralatih | 69,96% | 55,34% → 55,84% | 55,23% → 55,12% |
| ResNet acak | 29,51% | 25,65% → 25,65% | 26,67% → 26,67% |

![Kurva seluruh K](../reports/skaid-svm-c/kurva-grid-c.png)

MFCC linear naik +11,14 pp dibanding grid lama (CI95% berpasangan
+8,37 hingga +14,15 pp); selisih terhadap Ridge +0,85 pp (−1,08
hingga +3,01). MFCC RBF naik +0,10 pp dari grid lama (0,00 hingga
+0,26), setara dua prediksi benar tambahan dari 3.060 prediksi
berulang, pada dua ID. RBF baru minus Ridge +1,05 pp (−1,18 hingga
+3,30), RBF minus linear baru +0,20 pp (−0,62 hingga +0,98).
Tidak ada rerata 80% atau 90% pada grid K1–10.

Mean memakai 17 ID setelah rata-rata draw/model di dalam ID, bukan
ensemble. Interval deskriptif dari 10.000 bootstrap ID seed 7471;
fold berbagi sumber. Fit dukungan tinggi untuk fitur neural tidak
menutup gap terhadap Ridge pada query. Query kontrol 6,73–9,02%,
dengan peluang uniform 8,33%.

## Seleksi dan biaya

114/340 pilihan linear normal dan 165/340 pilihan RBF normal beralih
ke C baru; dua pilihan kontrol RBF juga beralih. Semua 17 fold MFCC K10
memilih row L2, C=1.000. RBF menggunakan multiplier 0,1. Tidak ada
pilihan linear normal di C=100.000; hanya satu pilihan RBF normal
di batas tersebut. Profil skor sumber mendatar atau turun pada C
yang lebih tinggi untuk MFCC. Profil ini dipakai untuk pengembangan,
bukan ukuran akurasi target.

Budget K10 tetap **120 label target dan 5.749–5.807 label sumber
unik per fold**, termasuk 960 query sumber. Pemilihan resep memakai
ulang seluruh label yang sama. ResNet pralatih juga memakai 720 label
Harrison. Anotasi log, proses penyaringan dan label query berada di
luar penghitungan label classifier target.

## Audit

Audit mencakup seluruh 88.128 kandidat baru dengan kernel langsung,
146.880 audit lama yang dipatok, 816 pilihan, dan 4.896 kepala final.
Sebanyak 3.483 refit dengan resep yang tetap sama cocok persis dengan
parameter/skor lama; seluruh refit cocok dengan skor kandidat yang
sesuai. Rekonstruksi dual mengubah nol prediksi, dengan selisih skor
maksimum 5,19e-12.

**306 kandidat baru di luar toleransi skor**, semuanya linear dan 300
di antaranya memakai fitur ResNet acak. Ada 177 perubahan top-1 pada
5.287.680 prediksi kandidat baru dan 23 entri count sumber berubah;
tidak ada satu pun dari 816 pilihan yang berubah dalam pemeriksaan
diagnostik. Tidak ada outlier baru yang terpilih bagi ID-nya sendiri.
Temuan 65 outlier lama tetap tercatat dan tidak dihilangkan.

Pada kasus baru terbesar, skor OvR berbeda 1,2674 dan intercept
pasangan berbeda 1,3462, sementara koefisien dual pasangan berbeda
maksimal 1,14e-13. Kedua jalur memiliki koefisien sangat dekat C.
Ini konsisten dengan sensitivitas pemilihan intercept pada batas;
belum membuktikan penyebab tunggal seluruh perbedaan. Diagnosis
tidak mengubah koefisien, pilihan atau hasil utama.

Refit dengan cache fitur CPU mengubah dua prediksi normal dan dua
kontrol, seluruhnya pada ResNet acak RBF. Masing-masing kategori
memiliki satu perubahan benar/salah. MFCC, BEATs dan ResNet pralatih
tetap sama prediksinya. Seluruh 126 tes lulus.

## Implikasi

Rentang C lama memang membatasi baseline linear MFCC. Setelah batas
itu diperiksa, kenaikan dari kernel nonlinear kecil dan tidak konsisten
terhadap Ridge atau linear. Fitur neural tetap lebih baik dengan Ridge
pada grid ini. Tidak ada alasan memperluas C berulang pada query yang
sama. Pertahankan MFCC+Ridge sebagai baseline utama, dengan SVM linear
dan RBF sebagai pembanding yang parameternya sudah dievaluasi.

Studi masih bersyarat pada klasifikasi 12 huruf bertimestamp log di
cohort yang sudah dilihat. Belum ada verifikasi mikrofon, keyboard
atau sesi fisik baru, maupun pengenalan teks tanpa log. Berikutnya
prioritaskan protokol uji pada rekaman independen dengan metadata
jelas, sesuai [tujuan riset](RESEARCH_GOAL.md).

- [Laporan dan grafik](../reports/skaid-svm-c/laporan.md),
  [pilihan](../reports/skaid-svm-c/selected-recipes.csv),
  dan [per-ID](../reports/skaid-svm-c/per-id.csv).
- [Audit](../reports/skaid-svm-c-audit/audit.json),
  [prediksi](../reports/skaid-svm-c-audit/classifications.csv),
  dan [diagnosis](../reports/skaid-svm-c-numeric/summary.json).
- [Snapshot](results/skaid-svm-c-2026-11-14.json).
