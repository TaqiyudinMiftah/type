# SKAID: hasil pilot kalibrasi pada rekaman terpisah

Tanggal lokal: 9 November 2026. **MFCC + Ridge mencapai 52,29% pada
K=10**, dibanding ResNet pralatih 43,12%, BEATs beku 26,31% dan ResNet
acak 13,44% pada setup ini. Semua keluarga membaik ketika label ditambah,
tetapi belum mencapai rerata 80% atau 90%. Pilot ini belum menetapkan
jumlah label minimum untuk adaptasi yang andal.

## Apa yang diuji

[Protokol](SKAID_PILOT_PROTOCOL.md) dikunci sebelum fitur dan akurasi
SKAID dihitung. Ada 17 ID dengan dua M4A terpisah: Phase 1 memasok
dukungan, Phase 2 memasok query tetap. Subset alfabetis adalah
`a,d,e,h,i,l,n,o,r,s,t,u`. Setiap ID memiliki 60 query, lima per huruf,
yang sama untuk seluruh K, draw dan encoder.

Tiga draw 101/102/103 memilih dukungan nested dari pool Phase 1, dengan
K=1/2/3/5/10 atau 12/24/36/60/120 label unik per percobaan. Ekstraksi
memakai 6.137 crop unik: 5.117 union dukungan dan 1.020 query. Tidak ada
duplikasi PCM persis atau raw file yang dibagi antara dukungan/query.
Ada 28 pasangan crop terpilih yang bertumpang tindih dalam file yang
sama; seluruhnya di dalam satu peran, sehingga bootstrap memakai ID.

Crop 80 ms/pre 15 ms harus lengkap dan tidak memuat press/release tombol
lain menurut log. MFCC memakai 80 ms asli; neural mendapat buffer nol
400 ms dengan audio pada 35–115 ms dan press acuan pada 50 ms. Padding
menjaga kebutuhan frontend lama, tetapi konteksnya berbeda dari sumber
400 ms penuh. Tidak melakukan augmentasi atau fine-tuning backbone.

Semua fitur memakai L2 per baris dan Ridge alpha 1, tanpa scaler yang
dipelajari dari query atau target validation tambahan. Tiga ResNet
pralatih Harrison dan tiga ResNet acak disilangkan dengan ketiga draw.
ResNet pralatih memakai ulang 720 label sumber Harrison (648 fit +72
validasi); random memakai arsitektur dan statistik sumber yang sama,
tanpa pelatihan bobot. Ini kontrol encoder, bukan scratch end-to-end.

## Seluruh kurva

Mean berikut dihitung per ID terlebih dahulu, termasuk semua draw/model,
lalu dirata-ratakan pada 17 ID. Rerata ResNet bukan ensemble logits.
Peluang uniform pada query seimbang 12 kelas adalah 8,33%.

| K / label per ID | MFCC | BEATs beku | ResNet pralatih | ResNet acak |
|---|---:|---:|---:|---:|
| 1 / 12 | 35,29% | 14,22% | 23,31% | 10,52% |
| 2 / 24 | 40,85% | 17,84% | 28,41% | 11,00% |
| 3 / 36 | 43,59% | 18,79% | 31,82% | 12,25% |
| 5 / 60 | 47,32% | 21,47% | 37,41% | 12,73% |
| 10 / 120 | 52,29% | 26,31% | 43,12% | 13,44% |

![Kurva label](../reports/skaid-pilot-reviewed/kurva-k.png)

Pada K10, CI95% bootstrap 10.000 resample ID:

| Fitur | Mean query | CI95% bersyarat pada 17 ID | Dukungan fit-label | Kontrol query |
|---|---:|---:|---:|---:|
| MFCC | 52,29% | 48,01–57,03% | 69,07% | 8,24% |
| BEATs | 26,31% | 23,37–29,58% | 40,11% | 7,45% |
| ResNet pralatih | 43,12% | 40,26–45,97% | 63,34% | 8,71% |
| ResNet acak | 13,44% | 12,56–14,26% | 16,03% | 8,43% |

Kontrol mengacak label dukungan pada tingkat contoh, menjaga jumlah tiap
kelas, dan mempertahankan query. Seluruh 408 kontrol dilaporkan. K0 head
sumber ResNet, dengan logits dibatasi ke 12 kelas, mencapai 9,31%
(CI95% 8,37–10,23%), tanpa fit target.

Selisih berpasangan MFCC−ResNet pralatih pada K10 adalah +9,17 pp,
CI95% +6,02 hingga +12,71 pp. Rerata MFCC per ID berkisar 37,78–75,56%.
CI mempertahankan draw dan seed di dalam ID; tidak menganggap 1.020
crop, pengulangan draw atau model sumber sebagai rekaman independen.
Ini studi eksploratif, belum peringkat universal atau pembandingan SOTA.

Confusion MFCC K10 memperlihatkan n→o (34,90% baris n), i→o (29,02%)
dan e→o (26,27%). Ini 3.060 prediksi dari 1.020 query yang sama pada
tiga draw, bukan 3.060 audio unik. Penyebab fisik kesalahan belum
ditetapkan. Dukungan juga belum dikuasai pada resep Ridge ini; regularisasi,
representasi dan ketidakpastian crop masih perlu dibedakan.

## Apa yang sudah diverifikasi

Ekstraksi neural memakai **AMD Radeon RX 6600 LE**. Semua bobot dan
buffer tetap beku. Kontrol sumber BEATs 36 clip dan ResNet 180 contoh cocok
dengan artefak lama, termasuk prediksi sumber.

Auditor mendekode ulang 34 audio dan seluruh 6.137 crop: PCM native,
resample, padding dan array identik. Semua embedding/MFCC dihitung ulang
di CPU dan cocok dalam toleransi 1e-3. Seluruh 2.448 kepala dan 51 K0
di-replay; koefisien/skor cache identik. Penyelesaian dual ridge
independen berbeda paling besar 1,11e-15. Semua kepala juga di-fit ulang
dengan fitur CPU: **tidak ada prediksi query yang berubah**, termasuk K0.
Seluruh 105 tes proyek, Ruff lint/format, lock dan build wheel/sdist lulus.

## Batas dan kelanjutan

Budget 12×K menghitung label **kepala klasifikasi**. Segmentasi,
stratifikasi pool/query dan penyaringan tetangga memakai anotasi log
tambahan. Onset fisik belum dianotasi manual dan ketelitian fitur clock
masih 10 ms. Tanpa event lain di log bukan berarti crop bebas noise atau
ekor suara. Karena itu hasil belum mengukur biaya label end-to-end atau
deteksi bebas timestamp.

Seleksi ID/kosakata dilakukan sebelum akurasi dari kelayakan dataset yang
sudah diperiksa. Fase adalah file terpisah, tetapi metadata belum
menetapkan pergantian mikrofon, keyboard fisik atau sesi. Hasil tidak
berlaku otomatis pada alfabet penuh atau mereproduksi angka EchoStrike.

Langkah berikutnya: uji regularisasi/normalisasi melalui validasi pada
ID sumber yang berbeda; kunci pilihan sebelum evaluasi ID yang ditahan.
Hitung label sumber tambahan secara terpisah dan pertahankan baseline
ini. Jangan memilih resep dari query ID yang sedang dievaluasi.

- [Laporan lengkap dan grafik](../reports/skaid-pilot-reviewed/laporan.md).
- [Snapshot angka dan hash](results/skaid-pilot-2026-11-09.json).
- [Audit terstruktur](../reports/skaid-pilot-audit/audit.json) dan
  [prediksi seluruh kondisi](../reports/skaid-pilot-audit/classifications.csv).
- [CSV per ID](../reports/skaid-pilot-reviewed/per-id.csv),
  [per kondisi](../reports/skaid-pilot-reviewed/per-condition.csv), dan
  [kesalahan MFCC K10](../reports/skaid-pilot-reviewed/errors-mfcc-k10.csv).
