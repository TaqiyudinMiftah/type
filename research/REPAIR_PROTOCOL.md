# Perbaikan aliran sinyal attention

Ditentukan 8 November 2026, setelah kontrol LeakyReLU tiga seed selesai dan
sebelum fit baru di bawah ini. Autopsi awal menunjukkan variasi keluaran
attention kedua hanya sekitar 2e-10. Kode referensi menginisialisasi seluruh
proyeksi attention dengan normal std=0,001; kedua blok tidak memiliki residual
atau LayerNorm. Hipotesis: pengerdilan sinyal ini ikut menyebabkan belajar
lambat dan sensitif terhadap seed. Ini terpisah dari ReLU terakhir yang mati.

## Pemilihan perbaikan memakai Zoom saja

- Data/frontend/cache identik dengan pilot 400 ms, 16 kHz, mel 64 x 64.
- Domain pengembangan **Zoom saja**. Gunakan split sumber seed 42 lama:
  train 648, validasi 72, test 180. Jangan evaluasi test sumber.
- Pilih 10 contoh per kelas dari train sumber dengan RNG NumPy seed 101:
  360 contoh fit. Validasi tetap 72, seluruhnya terpisah. Statistik input
  dihitung hanya dari 360 contoh fit. Total label sumber yang tersedia untuk
  pengembangan ini 432; terpisah dari domain target ponsel.
- Inisialisasi 42, 43, 44; 6.000 update per fit, batch 16 dengan pengembalian,
  Adam 5e-4 dan peluruhan linear ke nol. Augmentasi dan frontend mengikuti
  diagnostik lama. Cek dukungan/validasi pada 1.000, 3.000 dan 6.000 update;
  penilaian kelayakan hanya pada checkpoint terakhir, tanpa early stopping.
- Kandidat pertama `xavier-leaky`: ubah hanya delapan matriks proyeksi
  q/k/v/o pada dua attention ke Xavier uniform gain=1; bias tetap nol.
  Pertahankan intervensi LeakyReLU(0,01) hanya pada `mlp4[1]`. Tidak ada
  parameter tambahan, perubahan optimizer, atau perubahan vendor.
- Kriteria lulus pengembangan: ketiga fit finite, variasi logit dukungan
  >1e-4, akurasi dukungan masing-masing >=95%, rerata validasi >=50% dan
  sample SD validasi <=10 poin persentase. Catat gradien input sebelum fit
  dan gradien stem selama fit untuk menilai mekanisme.
- Jika kandidat pertama gagal, jalankan **satu** kandidat cadangan
  `residual-prenorm`: Xavier dan Leaky yang sama, tambahkan jalur residual
  pre-LayerNorm pada masing-masing attention; pada MLP token gunakan
  `Linear_skip(x) + MLP(LayerNorm(x))`, dengan skip Xavier tanpa bias dan
  dimensi keluaran tetap 384/768; LayerNorm keluaran MLP terakhir. Stem
  konvolusi, pooling, data dan jadwal tetap sama. Uji tiga seed yang sama
  dan kriteria yang sama. Ini modifikasi arsitektur, bukan CoAtNet kanonis.
- Ambil kandidat pertama yang lulus. Jika keduanya gagal, jangan memilih
  berdasarkan uji ponsel: tandai belum terselesaikan dan simpan kegagalan.
- Simpan keputusan beserta hash seluruh hasil pengembangan, kode, protokol,
  data dan cache **sebelum** evaluasi ponsel. Dilarang mengubah kandidat
  atau hyperparameter dari hasil uji ponsel.

## Konfirmasi perbaikan pada ponsel

- Hanya setelah keputusan terkunci, fit kandidat terpilih dari awal pada
  ponsel K=10, seed dukungan 101, inisialisasi 42/43/44. Support 360 label
  event 0–14, test tetap 360 event 15–24. Tidak ada validasi target.
- 10.000 update, Adam 5e-4, batch 16 replacement, linear LR ke nol;
  normalisasi dukungan dan augmentasi persis sama dengan kontrol lama.
- Uji setiap checkpoint terakhir sekali; laporkan seluruh seed serta
  rerata/sample SD. Bandingkan tiga kontrol ReLU lama dan tiga Leaky lama
  dari `runs/stability-gate`, dengan identitas support/test/config diperiksa.
- Ukuran utama: keluaran konstan, penguasaan dukungan, variasi antar-seed.
  Akurasi uji sekunder, jangan memilih seed terbaik atau memperpanjang fit
  karena hasil uji. Audit semua checkpoint baru di CPU, metrik dan ID contoh.
- Jalankan fit hanya pada RX 6600 LE; Radeon 780M tetap tidak diekspos.

Hasil bersifat eksploratif: uji ponsel ini pernah dilihat dalam eksperimen
lama, event berasal dari WAV yang sama, tanpa sesi/keyboard/partisipan baru.
Keberhasilan optimisasi tidak membuktikan generalisasi antar-sesi, SOTA,
atau jumlah contoh minimum. Perubahan inisialisasi memerlukan retraining;
checkpoint lama tidak diperbaiki dengan menginisialisasi ulang bobotnya.
