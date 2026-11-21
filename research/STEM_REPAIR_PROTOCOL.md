# Stem yang mempertahankan beberapa fitur sejak lapisan awal

Ditentukan 8 November 2026 **setelah** keenam fit pengembangan dalam
`REPAIR_PROTOCOL.md` selesai dan **sebelum** fit berikut dimulai. Tidak satu
pun kandidat lulus; keputusan awal `failed_development` tetap dipertahankan.
Belum ada fit atau evaluasi uji ponsel pada tahap perbaikan ini.

Xavier saja masih berbeda jauh antar-inisialisasi. Residual/pre-norm membuat
dukungan ketiga seed mencapai 100%, 99,17%, 99,72%, tetapi validasi Zoom
79,17%, 29,17%, 69,44%. Jadi penguasaan data fit telah membaik, sementara
generalisasi antar-inisialisasi belum stabil. Autopsi kalibrasi ulang
BatchNorm pada salinan dua checkpoint tidak memperbaikinya: validasi
79,17→76,39% dan 29,17→27,78%; bobot/checkpoint asal tidak berubah.

Kode stem referensi untuk input mono memakai Conv(1→1)–ReLU–Conv(1→1)
sebelum ekspansi 1→64 dengan konvolusi 1x1. Hipotesis baru adalah bahwa
bottleneck satu kanal ini terlalu membatasi fitur spasial sejak awal.
Ini hipotesis pengembangan, bukan penyebab tunggal yang sudah dibuktikan.

## Satu kandidat tambahan, hanya pengembangan sumber

- Kandidat `wide-stem-residual` berasal dari `residual-prenorm`, mengganti
  `s0` dengan Conv3x3(1→32, tanpa bias)–GroupNorm(8,32)–GELU–
  Conv3x3(32→64, tanpa bias)–GroupNorm(8,64)–GELU; padding 1, stride 1.
  `mlp0` menjadi Identity, sehingga bentuk keluaran ke pooling tetap 64.
  Bobot dua konvolusi memakai Kaiming normal fan-out, gain ReLU.
  Blok setelah stem tetap memakai varian residual/pre-norm sebelumnya.
- Tidak mengganti frontend, data, augmentasi, pemilihan support, optimizer,
  LR, batch, jumlah update atau kriteria kelulusan dari protokol awal.
- Zoom saja; fit 360 contoh dari train sumber, 72 validasi, tiga inisialisasi
  42/43/44, 6.000 update. Tidak mengevaluasi 180 test sumber.
- Kriteria: masing-masing dukungan >=95%, logit tidak konstan, rerata
  validasi >=50% dan sample SD validasi <=10 pp. Tidak memilih seed/epoch.
- Jika gagal, simpan sebagai kegagalan pengembangan dan jangan buka uji
  ponsel. Tidak menambahkan kandidat lain ke protokol ini.
- Kunci keputusan dengan hash kode baru, kode/protokol tahap sebelumnya,
  hasil enam kegagalan pengembangan, data dan cache sebelum konfirmasi.

## Konfirmasi, audit dan interpretasi

Jika kandidat lulus, konfirmasi tepat seperti protokol awal: ponsel K=10,
seed dukungan 101, inisialisasi 42/43/44, 10.000 update, tanpa validasi target,
checkpoint terakhir saja. Label sumber yang digunakan untuk pengembangan
tetap 432 dan label target per fit tetap 360; kandidat baru tidak menambah
contoh fit/validasi. Fit berurutan pada RX 6600 LE, tanpa Radeon 780M.

Laporkan seluruh sembilan fit pengembangan serta ketiga hasil ponsel,
termasuk kandidat yang gagal. Replay seluruh checkpoint baru di CPU dan
verifikasi ID, normalisasi, anggaran dan checksum. Ini pencarian arsitektur
eksploratif memakai validasi sumber, bukan preregistrasi sejak awal riset.
Hasil ponsel lama telah dilihat, tanpa sesi baru; jangan menyebut konfirmasi
ini holdout baru. Perubahan stem/attention/normalisasi merupakan satu paket,
sehingga tidak memisahkan kontribusi setiap unsur atau membuktikan SOTA.
