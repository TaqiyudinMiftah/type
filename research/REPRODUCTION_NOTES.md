# Catatan implementasi dan audit

## Lingkungan

- Dataset Harrison: commit `2ecdc09b6d1868aae50813f6a6c842bdd52dab06`, MIT.
- Backbone rujukan: `External-Attention-pytorch`, commit
  `c413dae1f053ac2bbff9e2d668e1d7a62f07c335`, MIT. Empat file sumber dikunci
  dengan SHA-256; salinan vendor tetap utuh.
- Python 3.12.13, PyTorch 2.14.0+rocm10.1.0, RX 6600 LE (`gfx1032`).
- Lingkungan ROCm terpisah dari PyTorch CPU; perintah memakai
  `.venv-rocm/bin/python` langsung.
- ROCR ordinal 0 dan HIP ordinal 0 menunjuk RX 6600 LE pada workspace ini.
  Identitas Radeon RX 6000 diperiksa sebelum alokasi model. Radeon 780M
  dikeluarkan dari profil perangkat aktif.

## Perbedaan dari artefak penulis

Kode latihan lengkap tersedia atas permintaan, sehingga percobaan ini memakai
backbone yang dikutip paper dengan pooling global dan head 36 keluaran.
Model memiliki 1.973.280 parameter. Satu kanal input dan ukuran 64×64 dipilih
secara eksplisit. Tidak ada bobot pralatih yang digunakan.

Hop 225 mengikuti Tabel 3; uraian paper menyebut 500/255. Bentuk mel 64×65 diubah
menjadi 64×64 melalui interpolasi bilinear. Mel memakai jendela Hann, power
spectrogram, log dB dengan floor 80 dB, dan frekuensi native tiap kanal.
Normalisasi mean/std per mel band hanya dipelajari dari 648 contoh latihan.
Time shift dilakukan secara sirkular hingga ±40% panjang gelombang, lalu dua
mask per sumbu dengan lebar maksimum enam bin. Mask diisi rata-rata contoh.

SVM memakai 20 MFCC serta delta orde 1/2 dari mel yang sama, kemudian mean dan
std menjadi 120 fitur. Ini baseline kontrol, bukan reproduksi model paper
terpisah. Tiga seed 42/43/44 mengubah split acak; split berurutan memakai
18/2/5 ketukan pertama/tengah/terakhir per kelas. CoAtNet memakai satu seed 42
per kanal. Determinisme bit per bit GPU tidak dipaksakan.

Paper melaporkan konfigurasi final `Random`, terpisah dari `Stratified`.
Percobaan ini menggunakan acak terstratifikasi untuk menjaga 18/2/5 contoh
setiap kelas. Karena itu, angka headline dibandingkan secara deskriptif dan
tidak dinyatakan sebagai reproduksi persis.

Commit rujukan repositori bertanggal Maret 2026, tetapi riwayat GitHub yang
diperiksa menunjukkan ketiga file model terakhir diubah pada Oktober 2021,
sebelum publikasi Harrison 2023. File model vendor tidak dimodifikasi.

## Isolasi

Isolator mengadaptasi fungsi README penulis (`n_fft=48`, hop 24, jendela
2.400 sampel sebelum dan 12.000 setelah onset). Zoom membuang satu detik awal
sesuai contoh kode. Ponsel mempertahankan rekaman penuh; contoh penyiapan khusus
ponsel tidak dipublikasikan dalam README tersebut.

Pencarian adaptif asli dibatasi 2.000 iterasi. Satu file, `phone/J`, tidak
mencapai hitungan 25 karena langkah mengecil membatasi rentang ambang.
Pencarian grid 4.096 titik menemukan 25 segmen pada ambang 4,060634. Perluasan
ini dicatat dalam manifest. Hitungan 25 memakai pengetahuan protokol dan belum
memastikan identitas semua transien. Tidak ada penggantian label atau pemilihan
timestamp berdasarkan skor klasifikasi.

Audit pada segmen yang benar-benar dipakai (`data/external/harrison-prepared/audit.json`):

| Kanal | Segmen | Gelombang unik | Duplikat persis | Jendela tumpang tindih | Jarak onset minimum |
|---|---:|---:|---:|---:|---:|
| Ponsel | 900 | 900 | 0 | 0 | 18.744 sampel |
| Zoom | 900 | 900 | 0 | 0 | 21.552 sampel |

Panjang potongan 14.400 sampel pada kedua kanal; durasi dan batas frekuensi mel
berbeda karena sample rate asli 44,1 kHz dan 32 kHz. Transfer kanal tidak
mengisolasi efek codec. Kontrol frontend seragam harus menjadi eksperimen baru.

## Runtime

Tes CNN kecil RX 6600 LE lulus forward, backward, dan optimizer. Saat inisialisasi
CoAtNet, MIOpen mengeluarkan pesan pustaka CK grouped convolution dan sebagian
kernel hipBLASLt `gfx1032` tidak ditemukan. Pelatihan kemudian tetap berjalan;
loss dan checkpoint validasi dicatat. Pesan ini tidak dipakai sebagai bukti
keberhasilan atau kegagalan reproduksi: kesimpulan numerik mengikuti hasil
eksperimen lengkap.

Kedua kanal selesai 1.100 epoch dengan loss finite. Replay CPU pada seluruh
10 konfigurasi menghasilkan confusion matrix uji dan transfer yang sama persis.
Artefak akhir dan observasi ada dalam `reports/harrison/laporan.md` dan
[OBSERVATIONS.md](OBSERVATIONS.md). Metrik makro memakai seluruh 36 kelas. Interval Wilson
merupakan pendekatan binomial pada 180 contoh; korelasi dalam rekaman dan
ketiadaan sesi independen membatasi penafsirannya.

## Eksperimen terpisah: pilot kalibrasi kanal

Protokol baru tersimpan di [FEWSHOT_PROTOCOL.md](FEWSHOT_PROTOCOL.md). Hasil lama
tidak ditimpa. Pilot memotong ulang rekaman penuh pada pusat event terverifikasi
manifest menjadi 16 kHz/400 ms dengan scipy polyphase; seluruh 1.800 waveform
unik, tanpa potongan berimpit atau padding. Jarak antarpusat minimum 425 ms.
Ini penyeragaman frontend, bukan verifikasi ulang label phone/J.

Backbone CoAtNet kecil dilatih ulang 600 epoch pada frontend baru. Pada target,
scratch dan adaptasi memakai normalisasi yang dihitung ulang dari dukungan yang
sama, masing-masing 1.000 update dengan learning rate yang ditetapkan di protokol.
Checkpoint akhir dipakai langsung; tidak ada label validasi target tambahan.
Pemilihan tiga dukungan diulang pada 360 event uji tetap per kanal.

BEATs merupakan tambahan pretrained umum yang terpisah dari reproduksi Harrison.
Kode Microsoft `5bdcf477f4d05b495e0512503b7d5692ab1e4568` (MIT), checkpoint
`BEATs_iter3_plus_AS2M.pt`, encoder 90.311.792 parameter dibekukan. Checkpoint
mirror dipatok revisi dan SHA256; tidak tersedia digest Microsoft untuk menyatakan
identitas lintas sumber secara independen. Manifest mencatat perbedaan sumber itu.

TorchAudio 2.14.0 yang cocok dengan torch lingkungan ini tidak ditemukan pada
index CPU resmi. Alih-alih mengganti lingkungan ROCm yang sudah berfungsi,
pilot memakai `kaldi.py` PyTorch murni dari repositori resmi PyTorch/audio,
commit `ec13a815b13ec6be3eeb8c3eb9ccb725dc322233` (BSD-2). Patch hanya mengubah
import BEATs ke adapter itu dan meniadakan import ekstensi native; helper
DCT/MFCC yang tidak digunakan dibuat gagal eksplisit. Fungsi fbank tetap utuh.

`research/check_kaldi_compat.py` membandingkan fbank dengan rilis resmi
TorchAudio v0.13.0, yang dekat dengan publikasi BEATs. Delapan klip nyata
memberikan nilai identik; checksum dan hasil ada di
`research/results/beats-kaldi-compatibility.json`. Ini pemeriksaan algoritme
frontend, bukan pengukuran akurasi classifier. Bentuk aktual pada 400 ms adalah
38×128 fbank dan 16 token×768; frame akhir yang tidak membentuk patch utuh
tidak masuk encoder. Tidak melakukan padding untuk memanjangkan klip.

Pada RX 6600 LE, ekstraksi fitur 1.800 klip (termasuk mel CoAtNet) selesai dalam
12,45 detik dan puncak alokasi tensor GPU sekitar 415 MiB. Waktu ini merupakan
ekstraksi berbatch yang diamortisasi dan mencakup pemanasan kernel awal, bukan
latensi deployment atau biaya adaptasi lengkap.
