# Audit asal rekaman pada pilot Harrison

15 November 2026. **Skor 83,52% pada lima contoh per tombol mengukur
kalibrasi dan uji ponsel yang berbagi WAV induk.** Audit ini memeriksa
pembagian data dan apakah arsip publik cukup untuk menyatakan bahwa WAV
Zoom dan ponsel berisi ketukan fisik yang sama. Tidak ada fit atau prediksi
baru.

## Apa yang benar-benar terpisah?

Kami cocokkan 72 hash WAV, 36 konfigurasi yang tercantum di audit lama,
dan keterkaitan manifest serta hasil yang sudah dibekukan. Ada satu WAV
per kelas per kanal. Set uji ponsel berisi 360 potongan yang sama pada
semua kondisi ponsel. Untuk seluruh 30 fit target, tidak ada ID potongan
yang dipakai ulang sebagai dukungan dan uji dalam fit yang sama. Namun
seluruh 36 WAV induk dukungan juga muncul pada uji. Tiga fit sumber Zoom
juga memakai WAV induk yang sama di sisi pengembangan dan uji Zoom.

| Evaluasi | Fit | WAV induk di sisi fit/validasi | WAV induk di sisi uji | WAV induk di kedua sisi |
|---|---:|---:|---:|---:|
| Kalibrasi ponsel, scratch/adaptasi | 30 | 36 | 36 | 36 |
| Model sumber Zoom | 3 | 36 | 36 | 36 |

Karena hanya ada satu WAV per kelas pada rute ponsel, membagi ulang
potongan dari rilis yang sama tidak dapat memberi dua WAV independen
untuk setiap kelas. Tiga blok dan urutan awal/akhir di dalam file tidak
memberi hari uji tambahan. [Hasil awal](RESNET_FEWSHOT_OBSERVATIONS.md)
tetap berlaku untuk pilot dalam batas ini.

## Apakah ketukan Zoom dan ponsel direkam berpasangan?

[Harrison dkk., §3.1 dan §4.1](https://arxiv.org/html/2308.01074) menjelaskan
dua cara perekaman dan 25 ketukan per tombol. Bagian itu tidak memberikan
pemetaan per ketukan, tanggal sesi, atau jam bersama. Pemeriksaan
tag format/stream pada 72 WAV dengan `ffprobe` tidak menemukan tag yang
dapat dipakai untuk memulihkan waktu perekaman. Jam modifikasi file di
workspace adalah waktu penyimpanan salinan, sehingga tidak dipakai.

Sebagai diagnostik tambahan, kami bandingkan **penanda waktu yang sudah ada**
di manifest potongan. Selisih mutlak antara durasi ketukan pertama dan
terakhir pada dua kanal memiliki median **1,15 detik** (0,09–4,73 detik).
Saringan berikut memeriksa kecocokan urutan 25 penanda setelah memberi
kelonggaran pada pergeseran awal, laju waktu relatif, dan kesalahan
penanda per kanal. Pasangan file dari tombol berbeda menjadi kontrol
kemiripan ritme. Tidak ada pencocokan label dari suara yang belum dikenal.

| Toleransi waktu per penanda | Perbedaan laju maksimum | Tombol sama lolos / 36 | Tombol berbeda lolos / 1.260 |
|---:|---:|---:|---:|
| 25 ms | 2% | 0 | 0 |
| 50 ms | 2% | 2 | 118 |
| 100 ms | 2% | 9 | 389 |
| 200 ms | 2% | 17 | 580 |
| 200 ms | 5% | 21 | 737 |

Pada baris terakhir, **58,33% vs 58,49%** menunjukkan bahwa kecocokan pada
toleransi itu sering muncul juga pada kontrol. Perbandingan menggunakan
ulang file sehingga 1.260 pasangan kontrol tidak independen. Ketika satu
penanda boleh dikeluarkan dari tiap kanal, 29 dari 36 pasangan lolos pada
toleransi terlonggar; mengeluarkan file dengan isolasi yang meragukan tidak
mengubah jumlah yang lolos. Lolos saringan berarti waktu relatif masih
mungkin konsisten, bukan bahwa ketukannya identik. Gagal saringan juga
bergantung pada asumsi urutan, galat penanda, dan laju waktu yang tetap;
pengeditan atau pergeseran yang tidak tetap tidak tercakup. Jadi audit ini
tidak memastikan apakah sebagian ketukan dibagi antar kanal.

Untuk menghitung saringan, setiap jarak antara dua penanda pada ponsel
dibandingkan dengan jarak yang bersesuaian pada Zoom. Semua jarak harus
mengizinkan satu faktor laju dalam rentang yang sama. Kelonggaran dihitung
dari batas kesalahan tiap penanda di kedua kanal, memakai laju maksimum
sebagai batas konservatif. Ini hanya syarat perlu untuk model jam yang
dipakai. Hitungan 6.480 pemeriksaan dikonfirmasi lewat perhitungan vektor
terpisah, dengan hash input tercatat.

## Implikasi bagi pengukuran mikrofon

Harrison tetap bukti bahwa kalibrasi memperbaiki skor pada perpindahan
**jalur** Zoom→ponsel yang diuji. Tanpa identitas ketukan bersama dan
rekaman target pada hari uji terpisah, arsip ini tidak memberi estimasi
biaya label saat hanya mikrofon yang berubah. Perbandingan antarjalur
juga mencakup pemrosesan Zoom. Ini memperjelas keterbatasan eksperimen
lama tanpa mengubah skornya.

[Hasil dan hash audit](results/harrison-recording-provenance-2026-11-15.json)
· [skrip audit](audit_harrison_recording_provenance.py)
· [kajian bukti mikrofon](MICROPHONE_EVIDENCE.md).
