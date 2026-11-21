# BEATs dan ResNet beku: evaluasi lintas folder MKA

Ditentukan 8 November 2026 setelah hasil transfer Harrison → MKA tanpa
kalibrasi dibaca, sebelum ekstraksi embedding MKA dan fit kepala studi
ini. Ini studi eksploratif dengan protokol lokal, bukan praregistrasi
eksternal atau dataset uji yang belum pernah diperiksa.

## Pertanyaan

Apakah embedding BEATs pralatih lebih berguna daripada representasi
ResNet Harrison untuk mengklasifikasikan tombol pada folder platform MKA
yang tidak menyediakan label latih bagi predictor tersebut?

Ini pembelajaran terawasi pada folder MKA lain, berbeda dari evaluasi
Harrison → MKA tanpa fit target sebelumnya. Pertanyaan ini juga berbeda
dari kalibrasi ke sesi baru pada platform yang sama.

## Masukan dan representasi tetap

- MKA v3, Karwan Mahdi Rawf dan Ayub Abdulrahman, CC BY 4.0:
  https://data.mendeley.com/datasets/bpt2hvf8n3/3.
- Gunakan tepat 1.296 potongan PCM unik dalam manifest 36 kelas yang sudah
  diaudit, dari 216 WAV induk. Satu pasangan folder/tombol memiliki satu
  WAV induk dan enam potongan. Tidak memasukkan salinan agregat/duplikat.
- Pakai crop utama peak400ms mono16k yang sudah dibekukan pada studi
  sebelumnya, tanpa memilih aturan baru dari hasil MKA. Tidak memakai
  aturan tiled. Cache input SHA256:
  0abf749ab820ba176f021f6645b598380055d47513e54c1e4636c5e8b2770ec7.
- Hasil studi induk SHA256:
  5c9027ea875bc1080fa209d87a55e0eb3a2200be01ba95ca7996fa0220529c47.
- Urutan kelas: 1234567890qwertyuiopasdfghjklzxcvbnm.
- Folder: hp, Lenovo, MSI, Mac, Messenger, Zoom. Ini kategori berkas,
  belum merupakan ID fisik keyboard, mikrofon, partisipan atau sesi.
- BEATs_iter3_plus_AS2M beku, checkpoint mirror berpin yang telah dipakai
  pilot lama; SHA256 d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34.
  Mean pooling token terakhir menghasilkan 768 fitur. Fbank memakai
  implementasi vendor yang sudah diuji, fbank_mean15.41663/std6.55582
  bawaan; tidak mengestimasi statistik dari MKA. Tidak memakai head AudioSet.
- ResNet: ketiga best.pt Harrison Zoom seed42/43/44 yang dikunci dalam
  sources-locked.json lama. Ambil 128 fitur setelah global average pooling,
  sebelum Linear128→36. Mean/std mel dan BatchNorm tetap milik sumber.
  Tidak memakai model yang telah diadaptasi ke ponsel.
- Kedua encoder memakai eval(), requires_grad false, inference_mode().
  Digest seluruh parameter/buffer diverifikasi sebelum/sesudah ekstraksi.
  Tidak ada fine-tuning backbone, augmentasi, pemilihan layer, PCA, atau
  standardisasi antar sampel berdasarkan data uji.
- Ekstraksi BEATs memakai RX 6000 saja, lewat pemeriksa perangkat ketat
  proyek (RX 6600 LE lokal). Radeon 780M dilarang. Ekstraksi ResNet,
  kepala linear, dan audit CPU. Tidak mengubah lingkungan ROCm lewat uv run.

## Split dan anggaran label

Enam fold leave-one-platform-folder-out. Untuk setiap fold, seluruh 216
potongan dari satu folder menjadi uji; hanya lima folder lain yang boleh
menyediakan label fit. Tidak ada validasi atau pemilihan hyperparameter.

- Utama: K=6, semua enam potongan/tombol/folder latih. Total 1.080 label
  potongan, 180 WAV induk, lima WAV/tombol. Uji 216 potongan dari 36 WAV
  yang terpisah secara identitas/fingerprint PCM induk dan potongan.
- Sensitivitas jumlah contoh: K=1/2/3/5 per tombol PADA SETIAP folder latih,
  total 180K label/fit. Masing-masing memakai draw101/102/103.
  K=1 berarti lima contoh per tombol dari lima WAV induk, bukan satu
  contoh total atau satu contoh pada folder uji.
- Tiga draw merupakan permutasi tetap enam indeks per kelompok induk;
  dukungan nested untuk seluruh K, identik untuk seluruh encoder. Urutan
  permutasi global dibuat sebelum fit dan tidak bergantung pada hasil.
- K=6 memiliki satu set dukungan unik, bukan tiga ulangan draw semu.
- Tidak ada raw WAV, fingerprint PCM induk/potongan atau potongan overlap
  yang melintasi fit/uji satu fold. Tiga pasangan overlap yang diaudit
  tetap di sisi yang sama karena berada dalam satu induk/folder.
- Target fold mempunyai nol label fit. Secara keseluruhan cross-validation,
  semua 1.296 label dipakai di fold lain; ini bukan final test terpisah
  yang tetap tidak tersentuh setelah studi.
- Semua split/indeks/ID/hash dan konfigurasi dikunci sebelum kepala pertama
  dilatih. Tidak mengubah split atau mengecualikan platform yang sulit.

## Kepala yang sama dan kontrol negatif

Normalisasi L2 per vektor saja, float64, tanpa parameter hasil fit.
RidgeClassifier alpha=1.0, solver=cholesky, fit_intercept=True,
class_weight=None. Parameter ini mengikuti kepala alpha1 pilot terdahulu,
bukan hasil pencarian MKA. Tidak melakukan tuning alpha per encoder/fold.
Semua 36 kelas wajib muncul; simpan koefisien, intercept, indeks fit/uji,
skor dukungan/uji dan metrik. Jumlah fitur/parameter head berbeda, sehingga
ini perbandingan pipeline terpilih, bukan efek kausal arsitektur terisolasi.

Kontrol negatif K=6: satu permutasi label tingkat WAV induk untuk setiap
fold, seed997. Permutasikan multiset 180 label induk latih, pertahankan
label yang sama untuk seluruh enam potongan satu induk. Jumlah kelas tetap
seimbang (30 potongan/kelas), label uji asli. Permutasi identik untuk semua
encoder dan tidak diulang untuk mencari skor rendah/tinggi.

Total: 78 split normal × empat representasi = 312 kepala; enam kontrol
negatif × empat representasi = 24 kepala. Empat representasi adalah satu
BEATs dan tiga ResNet. Tidak menganggap BEATs sebagai tiga model berbeda.

## Pelaporan dan audit

1. Laporkan accuracy top1, top5, macro-F1, confusion matrix serta akurasi
   dukungan per fold/model/K/draw. Baseline peluang uniform top1 1/36,
   top5 5/36. Tidak menampilkan interval binomial per potongan karena
   enam potongan satu induk saling bergantung.
2. Sekunder: rata-rata enam skor uji satu WAV sebelum klasifikasi,
   36 WAV/fold. Ini klasifikasi kelompok berlabel sama, bukan decoding teks.
3. Utama K6: BEATs satu hasil/fold; ResNet semua tiga inisialisasi dan
   rerata/sample SD antar inisialisasi. Rata-rata enam fold bersifat
   deskriptif, fold berbagi data latih dan bukan enam sesi independen.
4. Ringkasan gabungan memakai prediksi out-of-fold seluruh 1.296 potongan,
   masing-masing satu kali per konfigurasi. ResNet dilaporkan per seed
   dan rerata, bukan ensemble. Kurva BEATs memakai SD tiga draw; ResNet
   memakai desain 3×3 inisialisasi/draw. Laporkan kedua faktor terpisah,
   dan jangan menafsirkan sembilan kombinasi sebagai sembilan dataset.
5. Selisih berpasangan memakai split yang sama. Untuk kurva, kurangi
   rerata tiga seed ResNet dari satu BEATs per draw; tiga selisih unik,
   bukan sembilan ulangan BEATs independen. Tidak memilih seed terbaik.
6. Audit ekstraksi BEATs pada 36 kontrol Harrison Zoom (indeks0,25,...875)
   terhadap cache lama. Verifikasi ekstraksi ResNet dengan merekonstruksi
   head sumber dan mencocokkan prediksi MKA zero-shot lama.
7. Audit CPU seluruh embedding: allclose atol=1e-3, rtol=1e-3; simpan
   selisih maksimum. Bangun ulang crop dari ZIP memakai auditor studi
   induk. Periksa ulang state, split, jumlah unik dan semua checksum.
8. Audit semua kepala pada embedding tersimpan, termasuk fit ulang dan
   penyelesaian ridge terpusat float64 secara langsung dari label fit.
   Koefisien/skor harus allclose atol=1e-9, rtol=1e-9 dan prediksi/metrik
   identik. Uji juga head tersimpan pada embedding CPU yang baru;
   laporkan perubahan prediksi karena numerik jika ada, tanpa menyembunyikannya.

## Batas interpretasi

BEATs memiliki pretraining/dimensi/biaya berbeda dari ResNet Harrison;
keunggulan tidak membuktikan SOTA atau penyebab tunggal arsitektur.
Dataset, platform, kanal dan segmentasi masih dapat berubah bersamaan.
Independensi WAV berdasarkan file/PCM bukan bukti independensi sesi atau
provenance fisik; audit belum menolak semua salinan transformasi.
Struktur MKA belum menyediakan rekaman kalibrasi/uji berbeda pada folder
yang sama. Hasil tidak menetapkan jumlah label minimum untuk mikrofon
baru, tidak menguji dataset sendiri, dan tidak mendecode kalimat bebas.

Semua intervensi setelah hasil terbuka memerlukan studi terpisah.
