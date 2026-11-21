# Transfer lintas rekaman Lenovo LOQ

Tujuan tahap ini adalah mengukur transfer model dari kalibrasi `f558fd22` ke rekaman lain dan mengukur pengaruh jumlah label kalibrasi target di rekaman terpisah. Klasifikasi menggunakan potongan audio di sekitar timestamp yang direkam bersama ketukan oleh kolektor. Setiap fitur berasal dari audio potongan itu.

## Rekaman dan pemilihan potongan

Gunakan WAV mono 48 kHz dari paket kolektor yang telah diimpor, lalu cocokkan hash isi dengan metadata. Pakai metode timing pilot `f558fd22`, dengan RMS AC 5 ms, ambang relatif terhadap 0,5–2,5 detik hening, dan pencarian puncak ±65 ms pada Spasi pemeriksaan. Penanda ganjil dipakai untuk fit konsensus affine. Penanda genap menilai residual. Laporan timing standar dan hasil pilot disimpan terpisah. Segmen 80 ms mencakup 15 ms sebelum posisi terkoreksi. Tolak near-neighbor press/release, clipping, dan transien lemah pada −15 hingga +35 ms. Pada v2 gunakan event kuota terpilih. Pada v1 gunakan semua event yang memenuhi aturan kelas a–z dan Spasi.

Eksperimen retrospektif memakai `e9be52b8`. Metadata mikrofon dan peserta cocok dengan `f558fd22`. Audit timing pilot masih `needs_review`. Untuk diagnosis eksploratif, gunakan affine bila terdapat setidaknya dua penanda fit konsensus per ujung, sedikitnya tiga penanda validasi per ujung, residual validasi ≤20 ms, dan masalah tercatat sebatas dukungan kurang atau penanda lemah/ambigu. Simpan status `needs_review` dan tandai skor sebagai eksplorasi. Rekaman `8cc93c85` hanya diperiksa kualitasnya karena audio sangat lemah, sesi terputus, dan fit timing gagal.

Aturan retrospektif dikunci setelah inspeksi kelayakan waktu dan jumlah label, sebelum membaca skor klasifikasi `e9be52b8`. Uji K=0 dengan tiga model tersimpan pada setiap potongan yang lolos di `e9be52b8`, dengan keluaran 27 kelas. Gunakan ulang peristiwa yang sama untuk sensitivitas posisi potongan Ridge ±12,5 ms. Untuk kurva K, ambil dua pertiga awal durasi mengetik sebagai kandidat dukungan dan sepertiga akhir sebagai uji, dijaga 150 ms di kedua sisi batas. Kelas harus memiliki sedikitnya 10 dukungan dan lima uji. Pakai semua potongan uji yang lolos pada kelas tersebut.

## Kurva anggaran label

Pertahankan MFCC 78 dimensi pada 80 ms dan 16 kHz. Ridge memakai standardisasi dari bagian fit, normalisasi L2 per potongan, dan alpha 10 yang dipilih pada validasi `f558fd22`. Bandingkan Ridge sumber + dukungan target dan Ridge dari dukungan target saja. Sumber memakai tepat bagian latih pilot terdahulu pada kelas yang dibandingkan. K=0 memakai fit sumber pada kelas kurva yang sama. Ambil subset tersarang pada K=1, 2, 3, 5, dan 10 dengan sepuluh seed tetap 200–209. Gunakan query yang sama untuk semua metode, anggaran, dan seed. Acak label dukungan untuk kontrol target-only pada anggota dukungan yang sama. Laporkan label sumber, label target, akurasi ketukan, akurasi rata-rata per kelas, dan hitungan query. Sebaran antar-seed mengukur variasi pilihan dukungan pada satu query tetap.

## Evaluasi prospektif

Untuk kanal yang sama, gunakan `Uji pertama` sebagai sumber dukungan target dan `Uji kedua` sebagai query pada studi kuota asli. Untuk mikrofon fisik baru, pakai sesi `Kalibrasi` dan `Uji pertama` dari satu studi kuota baru oleh peserta yang sama. Catat mikrofon, posisi, ruangan, dan pengaturan pengambilan untuk interpretasi kedua kondisi. Metadata harus mendukung keunikan rekaman, urutan hari, peserta yang sama, kanal yang konsisten dalam pasangan target, dan kondisi terhadap sumber yang ditentukan. Pertahankan kedua WAV dalam peran berbeda.

Gunakan aturan potongan dan model di atas pada ketiga rekaman. Untuk eksperimen prospektif, kedua rekaman target harus lolos audit timing pilot. Kurva alfabet dan Spasi memakai semua 27 kelas. Tiap anggaran K membutuhkan sedikitnya K potongan dukungan dan lima query per kelas. Tetap ukur K=0 pada seluruh potongan query yang layak dan laporkan anggaran yang didukung data. Bekukan rencana, hash, dan batas antarperan sebelum evaluasi. Keluaran menyatakan klasifikasi bertimestamp untuk masing-masing kondisi. Perbandingan mikrofon membutuhkan data nyata pada kedua kondisi dan catatan pergantian perangkat fisik.
