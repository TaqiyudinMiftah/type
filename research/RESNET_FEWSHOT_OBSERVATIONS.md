# Efisiensi label CNN residual: hasil 8 November 2026

Pretraining Zoom membantu adaptasi ke ponsel pada semua 15 pasangan
blok/K yang diuji. Dengan CNN residual yang sudah diperbaiki, keuntungan
adaptasi tetap terlihat saat kedua metode menguasai seluruh contoh latihan.
Ini hasil pilot Harrison, dengan 36 kelas dan uji 360 potongan ketukan tetap.

| Contoh target/tombol (K) | Label target/fit | Scratch: top-1 ± SD | Adaptasi: top-1 ± SD | Selisih berpasangan ± SD |
|---:|---:|---:|---:|---:|
| 1 | 36 | 27,22% ± 0,48 pp | 44,72% ± 1,94 pp | +17,50 ± 2,42 pp |
| 2 | 72 | 43,52% ± 2,05 pp | 63,33% ± 2,37 pp | +19,81 ± 0,58 pp |
| 3 | 108 | 57,78% ± 1,94 pp | 76,76% ± 1,40 pp | +18,98 ± 1,37 pp |
| 5 | 180 | 72,50% ± 0,73 pp | 83,52% ± 2,05 pp | +11,02 ± 2,78 pp |
| 10 | 360 | 87,22% ± 2,55 pp | 94,07% ± 1,67 pp | +6,85 ± 1,76 pp |

SD adalah sample SD dari tiga blok (inisialisasi, pilihan dukungan):
(42,101), (43,102), (44,103). Kedua faktor berubah bersama, sehingga SD
ini tidak memisahkan variasi inisialisasi dari pemilihan contoh. Semua
blok memakai event uji yang sama; bukan 1.080 event uji independen.

## Kesimpulan yang didukung hasil

Pada grid K=1/2/3/5/10, adaptasi pertama kali melewati 80% pada K=5,
baik menurut rerata maupun seluruh blok. Scratch pertama kali melewati
80% pada K=10. Jika sumber sudah tersedia, anggaran kalibrasi target
untuk ambang ini turun dari 360 menjadi 180 label, atau 50% pada grid
yang diuji. K=4 dan titik di antara grid belum diuji, sehingga K=5
bukan estimasi jumlah minimum yang presisi.

Untuk ambang 90%, adaptasi mencapainya pada K=10 di ketiga blok:
95,83%, 93,89%, dan 92,50%. Scratch K=10 menghasilkan 90,00%, 85,00%,
dan 86,67%; reratanya belum mencapai 90%. Satu blok scratch yang mencapai
90% tidak mewakili ketiga pilihan kalibrasi.

Tanpa label target, model sumber hanya mencapai 4,91% ± 1,05 pp pada
ponsel, walaupun test Zoom masing-masing 95,00%, 96,67%, dan 96,11%.
Pretraining tetap berguna setelah fine-tuning, tetapi penggunaan langsung
model sumber belum berhasil pada perpindahan kanal ini. Perubahan dari
K=0 ke K>0 juga mencakup penggantian statistik normalisasi sumber dengan
statistik dukungan target. Selisih adaptasi−scratch memakai normalisasi
target yang sama, sehingga lebih tepat untuk menilai manfaat inisialisasi
sumber pada resep ini.

Seluruh 30 fit target mencapai akurasi dukungan akhir 100%. Keunggulan
adaptasi pada hasil ini bertahan setelah scratch menguasai dukungan;
perbandingan ini sudah mengatasi kelemahan kontrol scratch pilot lama
yang masih underfit pada 1.000 update. Belum memisahkan kontribusi kepala,
fitur, BatchNorm, atau strategi fine-tuning lain.

## Perbandingan dan biaya

Kedua metode memakai compact-resnet 311.892 parameter, frontend
16 kHz/400 ms/mel 64×64, dukungan nested yang sama dalam setiap blok,
Adam baru 5e-4, batch 16 replacement, augmentasi yang sama, serta tepat
10.000 update dengan LR linear ke nol. Adaptasi menyalin seluruh bobot
dan buffer model sumber; semua parameter dan BatchNorm dilatih kembali.
Target tidak menggunakan validasi, early stopping, atau pilihan checkpoint
berdasarkan test. Model target terakhir dipakai pada setiap fit.

Biaya sumber adalah 648 label train +72 validasi Zoom, total **720 label
unik**, digunakan ulang pada tiga seed. Arsitektur bersama sebelumnya
dipilih dengan 432 label Zoom yang tercakup dalam 720 ini. Biaya tersebut
tetap harus dihitung; hasil tidak menyatakan penghematan 50% dari seluruh
label sumber dan target. Sepanjang kampanye, tiga pilihan dukungan
mencakup 518 label ponsel unik, tetapi satu fit hanya menerima 36K labelnya
sendiri. Label uji tidak masuk anggaran fitting.

Tiga pretraining memerlukan total 5,24 menit dan 30 fit target 52,11 menit.
Seluruh fit memakai **RX 6600 LE**, tanpa Radeon 780M. SD scratch K=10
sebelumnya, 0,85 pp, berasal dari dukungan tetap seed 101; SD 2,55 pp di
sini mencakup perubahan dukungan juga. Perbedaan ini tidak membuktikan
optimisasi kembali tidak stabil.

## Audit dan batas interpretasi

Replay CPU cocok untuk seluruh 33 checkpoint baru (3 sumber +30 target)
dan tiga evaluasi K=0: prediksi individual, confusion matrix, top-1/top-5,
macro-F1, serta penguasaan dukungan. Selisih skor test absolut maksimum
CPU/GPU adalah sekitar 2,29e-5. Audit memeriksa ID dan anggaran, statistik
input hanya dari fitting, inisialisasi, counter Adam/BatchNorm, jadwal LR,
dan fingerprint pasangan pada enam update yang ditetapkan. Seluruh 15
pasangan cocok. Checkpoint pemulihan menyimpan optimizer/scheduler/RNG.

Masih memakai WAV Harrison yang sama, dan test ponsel telah dilihat dalam
tahap penelitian sebelumnya. Ini klasifikasi potongan, belum deteksi event
dan pemulihan teks beruntun. Isolasi berbantuan jumlah ketukan serta
ketidakpastian phone/J tetap berlaku. Belum membuktikan generalisasi sesi,
keyboard, pengguna, atau mikrofon baru; hasil ini juga tidak menetapkan
SOTA atau keunggulan atas semua strategi adaptasi/encoder lain.

Langkah berikutnya yang paling berguna adalah mengunci resep ini lalu
memverifikasi manfaatnya pada rekaman eksternal lain dengan kelompok
rekaman yang jelas. [Audit kelayakan dataset eksternal](EXTERNAL_VALIDATION_FEASIBILITY.md)
menyimpan pemeriksaan MKA/SKAID/HEAR; audit hitungan log SKAID sudah
dilakukan, tanpa pelatihan model pada dataset tersebut. Kecukupan kelas
dan sinkronisasi audio perlu dipastikan sebelum menentukan protokol baru.

Artefak:

- [Protokol sebelum fit](RESNET_FEWSHOT_PROTOCOL.md).
- [Laporan lengkap dan semua blok](../reports/resnet-fewshot/laporan.md),
  [grafik PNG](../reports/resnet-fewshot/kurva-label.png),
  [grafik PDF](../reports/resnet-fewshot/kurva-label.pdf).
- [Snapshot hasil, biaya dan audit](results/resnet-fewshot-2026-11-08.json).
- `reports/resnet-fewshot-audit/audit.json` dan `classifications.csv`.
- `runs/resnet-fewshot/sources-locked.json`, checkpoint dan history setiap fit.
