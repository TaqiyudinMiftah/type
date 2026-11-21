# SKAID: hasil SVM linear dan RBF

11 November 2026. **MFCC+RBF mencapai 76,73% pada K10**, dibandingkan
75,78% Ridge terpilih. Selisih +0,95 poin persentase memiliki CI95%
berpasangan −1,37 hingga +3,27 pp; belum memberi bukti kenaikan yang
meyakinkan. SVM tidak memperbaiki fitur neural pada grid ini.

## Eksperimen yang dilakukan

[Protokol](SKAID_SVM_PROTOCOL.md) dibekukan sebelum kandidat SVM dihitung,
setelah hasil Ridge diketahui. Ini eksplorasi lanjutan pada cohort yang
sama, bukan konfirmasi pada test baru. Pertahankan seluruh 17 ID, 12
huruf, crop 80 ms, query Phase 2, dukungan nested Phase 1, tiga draw,
dan delapan encoder beku. Tidak mengubah query atau menyaring ID.

Sepuluh resep linear dan 50 resep RBF memakai dua normalisasi support-only.
C berada pada 0,01–100. Bandwidth RBF dihitung dari median jarak pasangan
dukungan; multiplier 0,01–100 dipilih dari ID sumber. Query tidak ikut
statistik normalisasi atau bandwidth. Tidak memakai kalibrasi probabilitas.

Seluruh 146.880 kandidat konvergen, tanpa fallback median nol; maksimum
19.474 iterasi pada satu masalah biner. Sebanyak 816 pilihan dikunci
sebelum 4.896 refit luar. Fungsi seleksi hanya menerima 16 ID selain ID
yang ditahan. Satu resep bersama untuk draw/seed per keluarga/K/ID;
kontrol permutasi label memiliki tabel seleksi sendiri.

Tahap ini memakai CPU dan cache yang sudah diaudit. Ekstraksi neural
sebelumnya memakai RX 6600 LE; tidak ada ekstraksi/backbone update baru.

## Hasil utama

Mean antar 17 ID, sesudah rata-rata draw/model dalam ID; bukan ensemble.
K10 berarti 10 contoh per huruf, atau 120 label classifier target.

| Fitur | Ridge terpilih | SVM linear | SVM RBF |
|---|---:|---:|---:|
| MFCC | 75,78% | 65,49% | 76,73% |
| BEATs | 69,80% | 50,92% | 50,78% |
| ResNet pralatih | 69,96% | 55,34% | 55,23% |
| ResNet acak | 29,51% | 25,65% | 26,67% |

![Kurva](../reports/skaid-svm/kurva-svm-vs-ridge.png)

| K / label target | MFCC Ridge | MFCC linear | MFCC RBF |
|---|---:|---:|---:|
| 1 / 12 | 43,40% | 41,96% | 41,96% |
| 2 / 24 | 55,78% | 46,93% | 54,74% |
| 3 / 36 | 61,60% | 50,65% | 61,08% |
| 5 / 60 | 69,02% | 55,10% | 68,66% |
| 10 / 120 | 75,78% | 65,49% | 76,73% |

Kenaikan MFCC RBF pada K10 terjadi pada 10 ID, dua tetap dan lima turun.
CI akurasi RBF 73,37–79,97%. Bootstrap memakai 10.000 resampling ID,
seed 7471, dengan draw/model tetap di dalam ID. Fold berbagi sumber;
interval ini deskriptif dan bersyarat pada cohort/prosedur.

RBF menurunkan BEATs 19,02 pp dan ResNet pralatih 14,73 pp terhadap
Ridge. Kedua fitur neural turun pada semua 17 ID. Fit dukungannya tetap
hampir 100%, sehingga fit tinggi saja bukan bukti transfer yang baik.
Tidak ada mean 80%/90% pada grid K1–10 untuk kedua kebijakan SVM.

Kontrol query K10 seluruh keluarga/kebijakan berada pada 6,73–9,02%,
dengan peluang uniform 8,33%. Kontrol MFCC RBF 6,73% berada di bawah
peluang; jangan menyebut semua kontrol tepat sama dengan peluang.
Tidak ada akurasi tinggi setelah support-label diacak. Interval bootstrap
kontrol bukan uji permutasi populasi yang independen.

## Biaya dan batas grid

Label sumber unik tetap sama dengan studi Ridge terpilih: pada K10,
5.749–5.807 per fold, termasuk 960 query dari 16 ID sumber. Kedua
kebijakan memakai ulang label itu; tambahan kandidat tidak menambah
label unik. ResNet pralatih juga memakai 720 label Harrison. Angka
120 tidak mencakup supervisi pralatih, log/filter, atau query evaluasi.

**241 dari 340 pilihan RBF normal berada di C=100, batas tertinggi.**
Pada MFCC K2–10, seluruh held ID memilih row L2, C=100 dan multiplier
0,1. Karena aturan seri memilih C lebih kecil, batas ini tidak bisa
dianggap optimum universal. Kesimpulan tentang SVM berlaku untuk grid
yang dicoba. Jangan memperluasnya diam-diam setelah evaluasi.

## Audit dan sensitivitas numerik

Seluruh 146.880 kandidat diperiksa melalui kernel langsung; seluruh 816
pilihan, union budget, dan 4.896 kepala final diaudit. Rekonstruksi dual
berbeda maksimal 1,03e-12 tanpa perubahan prediksi. Refit pada cache fitur
CPU tidak mengubah satu pun prediksi normal pada kedua kebijakan.
Ada dua perubahan prediksi kontrol RBF, satu mengubah salah menjadi benar.
Resep tidak diperbarui dari replay tersebut.

**65 kandidat melampaui toleransi skor precomputed vs kernel langsung.**
Ada tujuh perubahan top-1 dari 8.812.800 prediksi kandidat. Delapan
outlier merupakan kepala ResNet acak yang terpilih untuk ID-nya sendiri;
prediksi top-1 kedelapannya tetap sama. Count sumber agregat tidak
berubah, sehingga seluruh 816 pilihan juga sama bila dihitung secara
diagnostik menggunakan count kernel langsung. Satu kandidat berbeda
count fit dukungan. Jangan menyebut audit seluruh skor lolos toleransi.

Selisih OvR maksimal 1,07787; tidak semuanya berasal dari margin biner
yang sangat kecil. Pada kasus terbesar, intercept berbeda 0,26548 meski
koefisien pasangan hanya berbeda 1,42e-14. Satu jalur memiliki satu
koefisien sedikit di bawah C; jalur lain tepat di C. Keduanya berada
dalam interval intercept KKT diagnostik yang sama. Pada 55/65 kasus,
jumlah koefisien interior menurut pembandingan eksak berbeda.

Pola tersebut konsisten dengan sensitivitas pemilihan intercept di batas
koefisien. [Implementasi libsvm](https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.9.1/sklearn/svm/src/libsvm/svm.cpp)
memakai status bound eksak dan rumus rho berbeda bila ada koefisien free.
[Diagnosis batas](../reports/skaid-svm-bounds/summary.json) bersifat
post-hoc, bukan koreksi model atau bukti penyebab tunggal semua selisih.
Tidak melakukan snapping koefisien atau mengganti konfigurasi solver.

Seluruh 120 tes lulus, dengan 24 warning deprecation vendor lama.

## Implikasi

RBF memberikan peningkatan kecil pada MFCC di K10, dengan ketidakpastian
yang masih meliputi nol. Ridge tetap pembanding kuat dan lebih sederhana;
menambah kompleksitas kepala belum menghasilkan perbaikan yang konsisten.
Hasil ini tidak menetapkan bahwa seluruh model nonlinear gagal.

Langkah berikut yang langsung ditopang validasi sumber adalah studi batas
C: tetapkan grid tambahan sebelum menghitung kandidat, pilih lewat 16 ID
sumber, lalu evaluasi semua ID/query yang sama tanpa mengganti resep dari
hasil target. Setelah batas regularisasi lebih jelas, fitur temporal MFCC
layak dibandingkan dengan ringkasan mean/SD yang digunakan sekarang.
Setiap kelanjutan tetap eksploratif sampai ada data konfirmasi independen.

Hasil belum mengukur mikrofon/keyboard/sesi fisik baru, alfabet 26 huruf,
atau segmentasi tanpa log. Metadata dan anotasi onset masih membatasi
klaim. Ini bukan reproduksi literal paper atau penetapan SOTA.

- [Laporan lengkap](../reports/skaid-svm/laporan.md),
  [seluruh ID](../reports/skaid-svm/per-id.csv), dan
  [pilihan resep](../reports/skaid-svm/selected-recipes.csv).
- [Audit](../reports/skaid-svm-audit/audit.json),
  [prediksi](../reports/skaid-svm-audit/classifications.csv), dan
  [diagnosis numerik](../reports/skaid-svm-numeric/summary.json).
- [Snapshot](results/skaid-svm-2026-11-11.json).
