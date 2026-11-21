# Regularisasi kepala dengan validasi bersarang pada MKA

Ditentukan 8 November 2026 setelah hasil Ridge alpha1 lintas folder MKA
dibaca, sebelum validasi grid atau inferensi uji baru studi ini. Ini
diagnostik eksploratif lokal, bukan praregistrasi eksternal atau final
test yang belum pernah diperiksa. Studi sebelumnya tetap utuh.

## Pertanyaan dan masukan tetap

Apakah kegagalan transfer lintas folder pada embedding beku bergantung
pada alpha1 yang dipilih sebelumnya? Bandingkan alpha1 dengan alpha
yang dipilih melalui validasi lintas folder hanya di dalam data latih.

- Gunakan anggaran penuh K6 saja: 1.080 potongan dari lima folder latih,
  216 potongan dari folder uji. Semua enam folder bergantian menjadi uji.
  Tidak mengulang kurva jumlah label sebagai bagian studi ini.
- Sumber MKA v3, Karwan Mahdi Rawf dan Ayub Abdulrahman, CC BY 4.0:
  https://data.mendeley.com/datasets/bpt2hvf8n3/3.
- Tepat 1.296 potongan unik, 216 WAV induk, 36 kelas. Crop peak400ms
  mono16k serta fitur tetap dari studi LOPO: satu BEATs768 dan tiga
  ResNet128 seed42/43/44. Tidak memilih encoder terbaik atau ensemble.
- Hasil LOPO SHA256 dc7120d828390de9f1fc61ec342eb73613e1026366715f4d448ef16e4d062f7e.
- Cache fitur SHA256 a0b14258bfbf84f77d2ff32304fa5b12769e4a681e74289f262f6c17384403cd.
- Manifest cache SHA256 c4e7ee56ccd04ecd6fac5b4dd167551a701fabcd56011b0e981327359394792f.
- Split LOPO lama SHA256 10de280d09d2ef4dc0fccbf01d0436529aaeb2ddf690e188192b899f09aa077d.
- Audit LOPO SHA256 538c3bc5f513db6e8d48940d3e213a8a9475bff83a7d3303892fe7c14056851b.
- Embedding CPU hasil audit sebelumnya SHA256
  081ad8e41ab6636f140f8c8e0b0cd4745d4081ba7c8ecbb30a9258b343677f44.
- Semua checksum/provenance model, input dan audit lama diverifikasi.
  Tidak melakukan ekstraksi baru, update backbone, normalisasi target,
  augmentasi, atau mengganti aturan crop. Seluruh tahap memakai CPU.
  Radeon 780M tetap dilarang; tidak memerlukan GPU baru.

## Pemisahan luar dan dalam

Enam outer fold persis K6 LOPO lama: lima folder untuk fit/seleksi,
satu folder uji dengan WAV induk dan PCM potongan terpisah.

Untuk setiap outer fold dan setiap representasi, jalankan lima inner
fold pada lima folder latih itu saja. Empat folder (864 potongan,
144 WAV) untuk fit kepala, satu folder (216 potongan, 36 WAV) untuk
validasi. Setiap potongan outer-train menjadi validasi tepat sekali.
Outer-test tidak masuk inner train atau validation. Tidak ada induk/PCM
yang melintasi ketiga peran. Label uji asli hanya untuk metrik akhir.

Grid alpha tetap:
1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100.

Kepala sama seperti LOPO: RidgeClassifier, solver cholesky,
fit_intercept=True, class_weight=None, float64, L2 per vektor tanpa
parameter fit. Hanya alpha yang berubah; tidak melakukan tuning
normalisasi, dimensi, checkpoint, layer, atau metrik seleksi.

Seleksi memakai jumlah prediksi benar pada gabungan lima inner validation
(total 1.080); semua folder berukuran sama. Bila jumlah benar sama persis,
pilih alpha terbesar. Tidak memakai tolerance tie, metrik uji, akurasi
dukungan, top5 atau macro-F1 sebagai pemecah seri. Laporkan pemilihan
batas grid; tidak memperluas grid sesudah melihat hasil.

Sesudah seluruh pilihan alpha dikunci, fit ulang satu kepala terpilih
pada seluruh lima folder outer-train, lalu prediksi outer-test. Simpan
seluruh kurva inner, koefisien/skor inner, pilihan, waktu dan hash.
Semua 48 pilihan harus selesai dan lock tersimpan sebelum prediksi
outer baru pertama. Baseline alpha1 dipakai ulang dari LOPO lama,
termasuk indeks/label/head/skor yang sama; tidak menghitung outer score
untuk kandidat alpha lain atau memilih alpha terbaik dari outer score.

## Kontrol dan jumlah fit

24 tugas normal = enam outer folder × empat representasi.
24 kontrol negatif memakai label permutasi tingkat WAV induk seed997
yang persis sama dengan K6 LOPO lama. Permutasi ditetapkan sekali di
outer-train, dan konsisten pada inner fit/validation; label outer-test
tetap asli. Metadata telah diperiksa sebelum protokol: seluruh kelas
masih muncul di semua inner training. Tidak mengulang permutasi untuk
mencari hasil yang diinginkan. Ini kontrol deskriptif, bukan uji p-value.

Total 48 tugas × sembilan alpha × lima inner fold = 2.160 fit inner;
48 refit akhir, total 2.208 kepala baru. Baseline 48 kepala alpha1 lama
dipakai ulang. Tidak menganggap kontrol/seed sebagai dataset independen.

## Pengukuran, audit dan keluaran

- Accuracy top1 utama; top5, macro-F1, confusion matrix, histogram
  prediksi dan akurasi dukungan sebagai deskriptif. Agregasi enam skor
  potongan per WAV menjadi pengukuran sekunder, 36 WAV/fold.
- Laporkan semua folder, representasi, alpha terpilih, kurva inner,
  dukungan, baseline dan selisih berpasangan pada potongan uji yang sama.
  Tidak ada confidence interval binomial potongan yang saling bergantung.
- Ringkasan out-of-fold memuat 1.296 potongan unik satu kali per model
  dan metode. BEATs satu model, ResNet ketiga seed dan mean/sample SD
  antar inisialisasi. Bukan ensemble, bukan SD populasi sesi.
- Skor inner pada alpha pemenang telah dipakai untuk seleksi dan optimis;
  jangan menafsirkannya sebagai perkiraan akurasi final yang bebas bias.
- Audit semua split dan checksum. Ulangi 2.160 fit inner dan seleksi
  tanpa membaca outer labels, cek koefisien/skor dengan penyelesaian
  ridge terpusat secara langsung. Toleransi numerik atol=rtol=1e-7
  untuk grid ini, dan prediksi serta jumlah benar/alpha pemenang harus
  identik. Toleransi lebih longgar daripada alpha1 karena grid memuat
  regularisasi kecil. Jangan melonggarkan setelah hasil dibuka.
- Audit 48 refit akhir, akurasi/indeks/head baseline alpha1 lama,
  lock seleksi sebelum evaluasi, dan seluruh metrik/prediksi individual.
- Terapkan head terpilih ke embedding CPU yang sudah diaudit, laporkan
  perbedaan prediksi jika ada. Tidak perlu mengulang encoder yang tidak
  berubah; sumber cache CPU tetap ditautkan secara eksplisit.
- Simpan run, selections-lock, split, NPZ inner/final, laporan CSV/JSON,
  grafik PNG/PDF, audit dan snapshot hasil. Tidak menimpa studi lama.

## Batas dan keputusan

Semua folder/label telah dianalisis sebelumnya dan ikut fit pada outer
fold lain. Nested CV ini menjaga pemilihan alpha setiap predictor agar
terpisah dari folder ujinya, tetapi tidak membuat dataset kembali buta.
Hasil tidak menetapkan SOTA, efek kausal arsitektur, atau kebutuhan label
minimum mikrofon baru. Nama folder bukan ID perangkat/sesi terverifikasi.
Pemisahan WAV/PCM belum membuktikan independensi seluruh transformasi.
MKA masih tidak menyediakan rekaman kalibrasi/uji terpisah pada folder
yang sama. Tidak ada dataset pribadi atau decoding teks pada studi ini.

Laporkan apa pun hasilnya. Jika alpha terpilih tetap dekat peluang,
nyatakan bahwa tuning regularisasi ini belum menyelesaikan transfer;
jangan menyimpulkan seluruh model/fitur akustik mustahil berhasil.
Setiap intervensi berikutnya memerlukan protokol terpisah.
