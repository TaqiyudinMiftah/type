# Transfer langsung Harrison → MKA tanpa kalibrasi

Ditentukan 8 November 2026 setelah audit struktur/audio MKA selesai dan
sebelum inferensi model pertama pada MKA. Ini evaluasi eksploratif lintas
dataset; audio, label nama berkas, dan gambar provenance sudah diperiksa.
Protokol ini bukan praregistrasi eksternal atau evaluasi buta.

## Pertanyaan dan masukan tetap

Seberapa baik classifier sumber Harrison Zoom mengenali 36 tombol pada
potongan yang disediakan MKA v3 tanpa contoh kalibrasi target?

- Dataset: Multi-Keyboard Acoustic Datasets v3, Karwan Mahdi Rawf dan Ayub
  Abdulrahman, CC BY 4.0, https://data.mendeley.com/datasets/bpt2hvf8n3/3.
- Arsip SHA256: 01fad57bf972782aa1cab74602e1965c0e489d7dceb37e36668a1d44f95ecb64.
- Manifest kurasi SHA256: fc8697c58742521e0db5274bfd609d44b54dbc579b8083d719e7909cf6d87e05.
- Semua enam folder hp, Lenovo, MSI, Mac, Messenger, Zoom; urutan kelas
  1234567890qwertyuiopasdfghjklzxcvbnm. Nama folder adalah kategori berkas,
  bukan identitas fisik keyboard/mikrofon atau sesi yang terverifikasi.
- Tepat 1.296 potongan PCM unik: enam potongan per tombol/platform, dari
  216 WAV induk unik. Tiga salinan tambahan HP/8 dan seluruh salinan agregat
  dikeluarkan oleh manifest. Tidak menyaring berdasarkan prediksi/kualitas.
- Tiga pasangan potongan unik sebagian overlap. Unit potongan bukan
  anotasi satu ketukan, dan enam potongan satu tombol berbagi satu WAV.
- Sumber: compact-resnet 311.892 parameter dari kurva label sebelumnya,
  seed 42, 43, 44; gunakan best.pt hasil pemilihan validasi sumber saja.
  Tidak memakai checkpoint adaptasi ponsel, tidak memilih seed terbaik.
  Source train 648 + validation 72 label unik, dipakai bersama ketiga seed.

Checkpoint SHA256 yang telah dikunci sebelum eksperimen sebelumnya:

| Seed | SHA256 best.pt |
|---|---|
| 42 | 5d64810b83a827651016ccaee186a30a22a13d39169fee49642fb4152848971d |
| 43 | b8e9eb2fe3d753d74ab13d571536efed7b7b6306145a1e1c0e15f31ff7139304 |
| 44 | 7491c04fe1feadde71a8453f15c425247dd62e1ec31d7e8392df4fdc3006b8c8 |

Ketiga normalisasi sumber memiliki SHA256
2f9adc21a944b760e2258a7b386f0c1de1e75f0bd8c3e57a7010b0fcb937ecbd.
Konfigurasi, hasil, skor sumber lama, implementasi, protokol, manifest,
dan arsip di-hash dalam lock sebelum inferensi target. Tidak mengubah
implementasi model/frontend atau artefak eksperimen lama.

## Pemrosesan dan sensitivitas yang ditentukan di muka

1. Baca WAV yang ditunjuk manifest langsung dari ZIP, verifikasi hash
   berkas dan PCM seluruh kanal terhadap audit. Decode float32, rata-rata
   kanal sebagaimana sumber; resample_poly ke 16 kHz dengan rasio integer
   yang disederhanakan gcd dan pengaturan default SciPy. Tidak melakukan
   normalisasi gain, pengurangan DC, seleksi kanal, atau estimasi noise.
2. Kebijakan utama peak: pilih indeks pertama argmax(abs(mono16k)), ambil
   6.400 sampel mulai 800 sampel sebelum puncak, clamp start ke
   [0, jumlah_sampel - 6400]. Tidak ada padding; jika panjang <400 ms,
   gagal dan laporkan, jangan mengeluarkannya diam-diam. Aturan puncak
   ini tidak mengklaim onset/ketukan teranotasi dan berbeda dari isolator
   count-assisted sumber. Laporkan jumlah clamp.
3. Sensitivitas tetap tiled: seluruh jendela 6.400 sampel dengan stride
   3.200 dimulai dari nol, tambah jendela yang berakhir tepat di akhir jika
   belum ada. Rata-rata aritmetika logit tiap jendela, lalu argmax satu
   label per potongan. Bobot sama, tanpa pemilihan berbasis confidence.
   Laporkan keduanya, tidak mengganti hasil utama bila sensitivitas lebih baik.
4. Frontend yang sama dengan sumber: Hann512, hop160, power-STFT,
   64 mel, log10 dB, lantai 80 dB per jendela, interpolasi bilinear64×64.
   Normalisasi per mel band memakai mean/std source train tersimpan.
   Tidak menghitung ulang mean/std dari target, termasuk tanpa label.
5. Seluruh parameter dan buffer BatchNorm tetap. Gunakan eval() serta
   inference_mode(), requires_grad false, tanpa optimizer atau update.
   Periksa digest seluruh state sebelum/sesudah. Label target dipakai
   hanya saat menghitung metrik setelah skor seluruh seed tersimpan.

Tidak ada pemilihan jendela, threshold, parameter atau checkpoint setelah
melihat akurasi MKA. Kebijakan di atas berlaku sama pada semua platform.
Inferensi dan audit memakai CPU; jika perlu GPU hanya RX 6000 yang sudah
diotorisasi. Radeon 780M dilarang.

## Pengukuran dan audit

- Utama: accuracy top1, top5, macro-F1, confusion matrix per platform,
  per seed dan kedua kebijakan. Semua kelas memiliki enam contoh unik.
  Laporkan rerata dan sample SD (ddof=1) di antara tiga inisialisasi sumber.
  Peluang uniform top1 1/36=2,78%, top5 5/36=13,89%.
- Sekunder: rata-rata enam logit potongan per WAV induk untuk klasifikasi
  kelompok rekaman; n=36/platform, bukan enam event independen. Tidak
  memakai agregasi lintas seed atau mengartikan ini sebagai decoding teks.
- Deskriptif: histogram prediksi, banyak kelas yang pernah diprediksi,
  dan kelas prediksi paling sering untuk memeriksa kolaps classifier.
- Tidak memberikan confidence interval binomial per potongan: enam
  potongan satu WAV saling bergantung. SD antar seed bukan ketidakpastian
  populasi keyboard/sesi; tidak menggandakan n dengan jumlah seed.
- Kontrol: ulang inferensi holdout sumber Zoom (180) dan transfer phone
  K=0 lama (360), cocokkan prediksi individual dan metrik tersimpan.
  Hitung frontend CPU pada waveform kedua kontrol, bandingkan fitur cache
  GPU lama dan pastikan prediksi tidak berubah. Perbedaan numerik dilaporkan.
- Simpan semua logit jendela, indeks/asal, crop offsets, fitur dan hash,
  skor agregasi dan prediksi. Audit ulang seluruh crop dari ZIP dan
  inferensi dari checkpoint, cocokkan identitas, state, prediksi/metrik,
  serta integritas input lama. Hash laporan tidak mengganti hash run.

## Batas kesimpulan

Hasil hanya mengukur transfer dari classifier Harrison yang dipilih
sebelumnya ke potongan MKA yang diberikan. Model belum terbukti SOTA.
Dataset, keyboard, cara mengetik, kanal, segmentasi dan cara pengambilan
potongan dapat berubah sekaligus; tidak menyimpulkan penyebab codec saja.
Audit hanya menolak salinan persis Harrison, belum mencari semua bentuk
salinan transformasi. Nama Zoom di dua dataset tidak membuktikan kanal
atau identitas rekaman yang sama. Tidak ada uji kalibrasi/uji lintas
rekaman pada platform target, anotasi event, atau decoding kalimat.

Setelah hasil terbuka, setiap intervensi menjadi eksperimen lanjutan
terpisah. Tidak menjalankan ulang konfigurasi untuk memperbaiki skor ini.
