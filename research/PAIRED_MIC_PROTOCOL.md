# Dua mikrofon, satu ketukan: protokol pengukuran

15 November 2026. Lanjutan setelah uji publik HEAR/NC1. Pertanyaan primer:
berapa contoh per huruf diperlukan, dan berapa perubahan top-1 ketika
mikrofon kalibrasi cocok atau berbeda dari mikrofon uji? Satu pengguna,
satu keyboard dan dua mikrofon pada posisi tercatat. Ini pilot satu setup,
bukan estimasi populasi perangkat. Jangan rekam percakapan, password,
atau input lain; gunakan hanya prompt huruf yang ditampilkan di terminal.

## Jadwal dan pemisahan

Isi alias peserta, identitas keyboard, dua label perangkat fisik,
zona waktu, dan deskripsi tetap posisi/jarak/gain/kanal/latar. Dua rute
input harus berasal dari dua mikrofon, bukan salinan sumber yang sama.
Gunakan dua kanal pada antarmuka yang sama jika tersedia; alternatif
dua input PortAudio yang direkam serentak. Pertahankan mode dan rute
(nama endpoint/host API, sample rate dan kanal), posisi, dan gain pada
semua hari. Dua perangkat terpisah dapat berbeda pada pemrosesan dan
latensi selain transduser: hasil berlaku untuk rantai rekam keseluruhan.

| Sesi | Waktu | Huruf/ketukan terpandu | Tujuan |
|---|---|---:|---|
| `pilot` | Sebelum pengukuran, opsional | 2 per huruf dari `adhtu`, 10 total | Periksa dua kanal, timing, tingkat bunyi dan pemotongan. Tidak melatih model. |
| `calibration` | Hari 1 | 16 per huruf a–z, 416 total | Data kalibrasi. Minimal 10 per huruf harus lolos pada kedua mikrofon. |
| `test1` | Hari 2 atau setelahnya | 12 per huruf a–z, 312 total | Uji ditahan; minimal 5 per huruf yang sama lolos pada kedua mikrofon. |
| `test2` | Hari lain setelah `test1` | 12 per huruf a–z, 312 total | Uji ditahan kedua; syarat sama. |

Jeda sedikitnya 0,8 detik antar ketukan; lepaskan tombol segera. Tiap
sesi dimulai dengan tiga detik tanpa mengetik dan tidak ada suara lain.
Urutan prompt berbeda antar sesi dan seimbang. Spasi menjeda; Enter
melanjutkan. Ketukan salah/terlalu cepat ikut pengecekan tetangga,
tetapi tidak menjadi contoh dukungan/query. Ctrl+C berhenti. Sinkronisasi
tidak memakai isi audio untuk menjodohkan dua sesi. Setelah kalibrasi,
bekukan fit dan split dukungan sebelum merekam `test1`. `test2` harus
hari lokal berbeda dari kedua sesi sebelumnya. Jangan mengganti sesi
berdasarkan skor. Jika syarat kualitas/coverage gagal, laporkan gagal;
gunakan ID studi baru untuk pengukuran yang diulang.

## Timing dan kualitas sebelum prediksi

Gunakan hanya input terminal aktif untuk label. Setiap ADC menyimpan
anchor (nomor frame awal blok, timestamp ADC PortAudio). Pada keystroke,
estimasi waktu pada setiap stream dari pembacaan waktu stream yang
diapit clock monotonic. Lalu interpolasi dari anchor stream itu sendiri.
Tidak menyamakan nilai clock mentah lintas perangkat. Mode antarmuka
memakai satu timeline. Timestamp terminal merupakan penanda, bukan
onset fisik. Dekode WAV mono float32 untuk masing-masing kanal. Tolak
sesi dengan buffer hilang, anchor tak monotonic, anomali waktu blok
>10 ms terhadap sample rate, atau stream gagal. Durasi/jeda hari berasal
dari jam kalender, tidak dipakai untuk mengubah event menjadi sampel.

Untuk tiap event gunakan blok RMS 5 ms terkuat pada −100 hingga +50 ms
di sekitar marker masing-masing. Tengah blok itu menjadi pusat crop
−15/+65 ms. Hindari interval kontrol dan event tetangga dalam jendela
gabungan. Tolak crop tidak penuh, RMS ≤1e-4 atau puncak blok kurang dari
4×RMS awal tenang, sampel clipping (|x|≥0,999), baca clock >5 ms atau
keterlambatan membaca clock >20 ms dari event, dan perbedaan posisi
puncak terhadap marker >50 ms antara kedua kanal. Peak bisa mewakili
impact lain; ini bukan anotasi onset mekanik. Jika crop gagal di satu
kanal, keluarkan event di kedua kanal. Laporkan semua penolakan dan
coverage per huruf; periksa salinan PCM serta rute mikrofon ganda.
Aturan ditetapkan sebelum data nyata, tidak disetel setelah lihat skor.

## Evaluasi terkunci

Primer a–z. Bandingkan matriks A→A, A→B, B→A, B→B dengan MFCC78
di 16 kHz dan Ridge alpha 0,001/row L2 yang dibekukan dari SKAID.
Untuk setiap K=1,2,3,5,10 dan draw 101/102/103, gunakan identitas
ketukan yang sama bagi dukungan A/B; urutan nested terpisah per huruf.
Semua event query lolos dipakai di kedua mikrofon, tanpa memilih
subset dari hasil. Kontrol label acak K10 seimbang dan memakai
permutasi sama pada A/B. Laporkan top-1, macro recall, confusion matrix,
kontrol, coverage, mean draw, dan hasil masing-masing hari. Persentase
macro recall dilakukan per huruf sebelum dirata-ratakan. Jangan
menganggap tiga draw atau dua view audio sebagai pengguna independen.

Sekunder memakai subset 12 huruf preset SKAID dengan identitas
kalibrasi/query sama. K0 memakai kepala MFCC+Ridge yang sudah dilatih
dari 5.117 klip Phase 1 SKAID pada run publik; K0 26 huruf tidak
kompatibel. Resep historis dipilih menggunakan union 6.137 label
SKAID. K>0 tidak fit ulang pada data SKAID; laporkan biaya fit target
26K / 12K, union dukungan, label pool dan query, serta penolakan.
Merekam A/B serentak memakai satu label per event, bukan dua.

Untuk query B, delta kanal cocok = B→B dikurangi A→B. Untuk query A,
= A→A dikurangi B→A. Hitung berpasangan pada event/hari/draw sama,
tanpa memilih satu kanal terbaik dari test. Laporkan K pertama yang
mencapai acuan 80%/90% dari mean top-1 tiap jalur di grid, jika ada.
K yang tidak diuji bukan batas minimum terukur. Sintetis hanya untuk
mengetes alur; tidak dicampur dengan rekaman nyata atau dipakai sebagai
bukti. Sistem ini mengukur klasifikasi potongan terpandu, belum deteksi
audio bebas. Semua perhitungan memakai CPU.

API clock: [PortAudio](https://www.portaudio.com/docs/v19-doxydocs/structPaStreamCallbackTimeInfo.html)
dan [python-sounddevice](https://python-sounddevice.readthedocs.io/en/0.5.5/api/streams.html).
