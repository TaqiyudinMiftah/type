# Audit rekaman publik yang terpisah

Tanggal lokal: 14 November 2026. Catatan ini ditetapkan sesudah
inventaris awal arsip/format, sebelum audit PCM penuh atau prediksi
classifier. Tujuan audit ialah menghitung pasangan rekaman yang benar-benar
dapat dipisahkan dan menyatakan perubahan kondisi yang terdokumentasi.
Ketersediaan data dilihat dari rilis yang diunduh, bukan jumlah pada paper.

## HEAR v1.0 (on-device)

Pakai rilis `v1` aset GitHub `HEAR.zip` 768.636.949 byte,
SHA256 `4b0f1908d0d1498ed448577a500acd3a0ad24c0362ba37558034660516b1dcf6`.
Rilis ini hanya on-device. Jangan mengartikan `.ogg/.webm`, codec,
browser atau kanal stereo sebagai mikrofon fisik yang berbeda.
Perangkat adalah token metadata, bukan model keyboard yang terverifikasi.
Tidak menyebarkan audio, nama responden, demografi atau isi log lengkap.
Gunakan indeks paket dan alias pseudonim dalam keluaran audit.

Periksa seluruh ZIP luar dan ZIP per sesi; tolak path tidak aman, nama
duplikat atau CRC rusak. ABAIKAN metadata AppleDouble untuk hitungan
data. Cocokkan tuple peserta/perangkat/sesi/banyak event dengan
`tracking.csv`, tanpa memaksakan pasangan jika ambigu. Simpan status
join, waktu rekaman relatif, format, durasi/PCM fingerprint audio
induk, status WAV, serta count/masalah manifest. Setiap export harus
merujuk satu entri log identik (`key`, `t_ms`), nama member unik, dan
ukuran WAV tepat. Catat kasus log lebih panjang daripada audio;
jangan memberi waktu imajinatif pada event tersebut.

Hitung kelayakan untuk dua kosakata tetap: 12 huruf pilot SKAID dan
26 huruf ASCII a–z, casefold satu karakter ASCII. Gunakan tiga tier:

1. **Export linked**: WAV dapat didekode finite dan ukuran/rate/channel
   konsisten dengan metadata, serta referensi log/manifest unik dan valid.
2. **Isolated 80 ms**: potongan utuh −15/+65 ms di sekitar timestamp
   export, berada dalam WAV dan durasi metadata, tanpa keydown lain
   pada interval setengah terbuka. Parent export 260 ms harus utuh
   menurut metadata. Filter hanya melihat keydown; release tidak
   dilog, sehingga tier ini berbeda dari filter SKAID. Potongan mono
   harus memiliki RMS >1e-4 (−80 dBFS RMS); catat penolakan terpisah.
3. **Verified parent 80 ms**: tier 2 dan native rate/channel sama,
   ada padanan PCM pada perkiraan posisi timestamp dengan toleransi
   rounding ±2 frame. Pilih RMS error terkecil; syarat error relatif
   ≤0,10 terhadap RMS potongan multikanal. Jangan geser potongan atau
   memperkirakan onset dari hasil ini. Periksa duplikat PCM crop
   lintas paket; hapus semua kemunculannya dari tier 3. Pada pasangan
   fingerprint PCM induk sama, jangan hitung sebagai rekaman berbeda.

Pasangan utama: peserta/token perangkat sama, tracking unik pada
keduanya, sesi bernomor naik, start sesi kedua setelah akhir audio
sesi pertama. Hitung semua pasangan berarah dengan minimal K contoh
dukungan dan lima query per huruf, K=1,2,3,5,10, pada setiap tier.
Laporkan jumlah peserta unik, pasangan, dan pasangan berjarak ≥12 jam
(analisis sensitivitas). Jangan menghitung banyak pasangan dari satu
peserta sebagai banyak peserta independen. Kelayakan ini tidak menguji
akurasi, ketepatan onset fisik, atau perubahan mikrofon.

## Novox NC1 (Zenodo 17536616)

Arsip `dataset.zip` CC BY 4.0; pin metadata, ukuran, MD5 penyedia, dan
SHA256 `78a0b8d3d4f6a8e7b3386ccdbe6618037d5ba40aceac30e30c80f5ba32621956`.
Tiga kondisi rilis: Fnatic Gear Rush tenang, Fnatic dengan latar bising,
Anne Pro 2 tenang. Paper menyatakan mikrofon sama, satu peserta; tidak
menyamakan kode kondisi dengan sesi yang terdokumentasi. Gunakan tanggal
di nama file untuk pengelompokan *hari rekaman terindikasi*. Simpan
kecocokan tanggal ZIP; jam/offset antar format bukan timestamp keydown.
Jangan menyimpulkan kelompok berbeda hanya dari split acak file.

Dekode semua WAV tanpa ekstraksi arsip ke path eksternal. Periksa CRC,
nama/label/waktu unik, PCM finite, sample rate, channel, durasi,
fingerprint penuh dan 80 ms. Untuk potongan pembanding yang tetap,
tentukan pusat dari blok RMS 5 ms terkuat pada klip terisolasi,
ambil −15/+65 ms, tanpa melihat label atau hasil model. Catat gagal
batas/sinyal; potongan ini benar-benar hanya diagnostik untuk dataset
klip berlabel, bukan uji detektor audio kontinu. Waktu itu puncak RMS,
bukan onset log. Jika dipakai untuk studi klasifikasi, bekukan protokol
penggunaan split/kosakata dan resep sebelum menghitung skor.

Hitung jumlah per huruf untuk setiap kondisi×tanggal, duplikasi
antar kondisi/hari, data cukup untuk K=1,2,3,5,10 dan query ≥5 per
huruf, serta jumlah kelas yang benar-benar tercakup. Bedakan uji
hari baru pada setup sama dari perubahan keyboard/noise bersamaan
dengan hari. Jangan menganggap klip pada satu hari sebagai sesi
fisik independen. Tidak ada perbandingan mikrofon pada korpus NC1.

Sumber primer:
[HEAR v1](https://github.com/vikrant-vikram/Hear-Dataset/releases/tag/v1),
[tesis HEAR bab 5](https://arxiv.org/html/2611.13922v1#S5),
[rekaman NC1](https://doi.org/10.5281/zenodo.17536616),
[paper NC1](https://doi.org/10.3390/app16020563).
