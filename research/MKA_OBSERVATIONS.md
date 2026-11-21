# Audit MKA: hasil 8 November 2026

MKA v3 sudah diunduh, diverifikasi dan diaudit. Dataset ini menyediakan
seluruh 36 tombol Harrison pada enam folder platform, tetapi struktur
yang tersedia **belum mendukung kalibrasi dan uji target lintas rekaman
pada platform yang sama**. Setiap pasangan platform/tombol hanya memiliki
satu WAV mentah; potongan dan folder agregat tidak menambah rekaman baru.
Tidak menjalankan pelatihan atau evaluasi akurasi MKA pada tahap ini.

Kelanjutan terpisah: [transfer langsung Harrison → MKA tanpa kalibrasi](MKA_ZEROSHOT_OBSERVATIONS.md)
sudah selesai dengan protokol baru dan seluruh prediksi diaudit. Hasil
akurasi rendah tersebut tidak mengubah keputusan kelayakan split di sini.

Sumber: Karwan Mahdi Rawf dan Ayub Abdulrahman,
[MKA v3, Mendeley Data](https://data.mendeley.com/datasets/bpt2hvf8n3/3),
DOI 10.17632/bpt2hvf8n3.3, CC BY 4.0. Angka berikut merupakan analisis lokal.

## Data yang benar-benar diperiksa

Arsip berukuran 1.997.707.877 byte cocok dengan SHA256 penyedia:
`01fad57bf972782aa1cab74602e1965c0e489d7dceb37e36668a1d44f95ecb64`.
Diagram penyerta juga cocok dengan checksum. Semua 15.654 berkas dalam
17.218 entri ZIP dibaca dan lolos CRC. Tidak mengekstrak keseluruhan arsip
atau memodifikasi audio asli.

| Folder | WAV mentah | Kelas mentah | Kelas bersama Harrison | Potongan pada folder audio |
|---|---:|---:|---:|---:|
| hp | 71 | 71 | 36 | 429 |
| Lenovo | 70 | 70 | 36 | 420 |
| MSI | 70 | 70 | 36 | 420 |
| Mac | 61 | 61 | 36 | 366 |
| Messenger | 70 | 70 | 36 | 420 |
| Zoom | 70 | 70 | 36 | 420 |

Seluruh 5.790 berkas audio yang dikenali libsndfile dapat didecode, tanpa audio
hening total/non-finite. Ada 2.881 fingerprint PCM unik pada seluruh kelas.
Fingerprint memakai rate, bentuk array dan semua kanal float64, sehingga
tidak hanya membandingkan nama atau ukuran file.

Audit pelengkap mendecode semua 120 objek teks Praat `Sound 2` di folder
fitur (60 pada folder platform dan 60 pada agregat). Seluruhnya identik
dengan PCM WAV yang sudah diperiksa, sehingga jumlah representasi audio
yang diperiksa menjadi 5.910 berkas, tetap 2.881 waveform unik. Ekstensi
atau nama folder fitur tidak dijadikan alasan untuk melewatkan data audio.

Semua **2.890 audio dalam `All Dataset` mempunyai salinan byte persis**
di folder platform. HP/8 mempunyai tiga potongan tambahan yang merupakan
duplikat. Sebanyak 13 audio ditemukan di folder fitur; semuanya salinan
audio mentah/potongan yang sudah ada. Di luar 36 kelas bersama, terdapat
alias nama `equal`/`equal(=)` dan WAV HP `altlR`/`Rctrl` yang identik;
perlu memeriksa nama label jika kelak memakai kelas tambahan tersebut.

## Asal potongan dan metadata

Seluruh **1.299 potongan yang diperiksa pada 36 kelas** cocok persis
dengan interval WAV mentah nominal, pada semua kanal. Setelah menghapus
tiga duplikat HP/8, tersedia **1.296 potongan PCM unik**, masing-masing
enam untuk 216 pasangan platform/tombol. Ada tiga pasangan interval
potongan unik yang overlap; tiga overlap tambahan berasal dari duplikat.
Rincian frame tersedia pada laporan provenance, tanpa menganggap
potongan sebagai event ketukan yang sudah dianotasi.

Semua 2.472 berkas dalam folder metadata platform berisi tabel numerik,
bukan keterangan sesi, partisipan, atau mikrofon. Pemeriksaan terpisah
atas 412 header WAV mentah menemukan hanya chunk standar `fmt` dan
`data`, tanpa chunk metadata tambahan. Tidak menggunakan nomor potongan,
waktu ZIP, atau nama platform untuk mengarang ID sesi/keyboard.

Sebanyak 72 WAV Harrison asli diperiksa; tidak ada salinan byte/PCM
lengkap pada MKA. Pencarian salinan yang dipotong/resample/ditransformasi
terhadap Harrison belum dilakukan, sehingga pemeriksaan ini tidak
membuktikan independensi asal rekaman secara menyeluruh.

Audit awal menemukan bug pada penelusuran stereo: pemilihan kanal
bervariansi terbesar secara terpisah dapat memasangkan indeks kanal
berbeda. Pemeriksa diperbaiki agar memakai indeks kanal yang sama saat
tata kanal cocok, diuji dengan regresi stereo, lalu seluruh audit diulang.
Ambang |r|>=0,999 dan aturan data/split tidak berubah. Hasil awal dan kode
awalnya dipertahankan pada `reports/mka-audit-initial-stereo-check/` dan
snapshot berlabel `initial-stereo-check`; gunakan hasil final yang ditautkan
di bawah.

## Keputusan dan rancangan yang dapat dipertanggungjawabkan

Manifest bersih menyimpan 216 kelompok WAV induk dan 1.296 potongan unik.
ID kelompok berasal dari fingerprint PCM induk, bukan ID sesi. Field
session/keyboard/participant serta alokasi split tetap kosong karena
metadata tidak mendukung pengisiannya.
Manifest memilih potongan dari folder audio; alias tambahan dalam format
Praat dicatat pada audit pelengkap dan tidak menambah label baru.

Enam potongan unik per kelas hanya cukup, secara jumlah, untuk K=5 dan
satu potongan uji per kelas, atau K=1 dan lima potongan uji. K=10 tidak
tersedia dari potongan yang disediakan. Audio mentah mungkin memuat lebih
banyak ketukan, tetapi jumlah event validnya belum dianotasi; enam
potongan tidak berarti hanya enam ketukan direkam. Menambah isolasi
ketukan dari WAV yang sama tetap tidak menghasilkan sesi uji baru.

Karena itu **tidak membuat split acak sebagai pengganti uji lintas
rekaman**, dan tidak menjalankan kurva few-shot dengan klaim tersebut.
MKA masih dapat dipertimbangkan untuk evaluasi transfer langsung
Harrison→MKA tanpa kalibrasi, atau generalisasi lintas folder platform.
Keduanya menjawab pertanyaan berbeda dan memerlukan protokol evaluasi
tersendiri. Untuk pertanyaan efisiensi kalibrasi lintas sesi, diperlukan
dataset eksternal yang menyediakan rekaman terpisah per kondisi/tombol.

Artefak:

- [Protokol audit](MKA_AUDIT_PROTOCOL.md).
- [Laporan lengkap](../reports/mka-audit/laporan.md),
  [grafik asal potongan](../reports/mka-audit/asal-potongan.png), dan CSV audio.
- [Manifest bersih 36 kelas](../reports/mka-36/manifest.json) dan
  [pemeriksaan header](../reports/mka-headers/headers.json).
- [Audit format Praat](../reports/mka-praat-audit/objects.json) dan
  [snapshot pelengkap](results/mka-praat-2026-11-08.json).
- [Snapshot audit](results/mka-audit-2026-11-08.json) dan
  [snapshot kurasi](results/mka-clean-36-2026-11-08.json).
