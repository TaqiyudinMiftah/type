# SKAID: pemilihan normalisasi dan regularisasi dari ID sumber

Tanggal lokal: 9 November 2026. Protokol ditulis setelah membaca pilot
alpha1: mean K10 MFCC52,29%, ResNet pralatih43,12%, BEATs26,31%, serta
fit dukungan yang masih terbatas. Semua query pilot sudah diperiksa.
Ini kelanjutan eksploratif pada kohort yang sama, bukan test baru yang
belum dilihat. Protokol dikunci sebelum menghitung kandidat baru.

## Data dan pertanyaan

Pertahankan seluruh17 ID,12 kelas, crop80 ms, delapan encoder beku,
K1/2/3/5/10, tiga draw nested dan60 query per ID dari pilot. Tidak
mengekstrak ulang encoder atau mengubah crop/label/split. Seluruh pekerjaan
memakai CPU dan cache yang sudah diaudit. Bandingkan kebijakan pemilihan
resep dengan baseline alpha1/per-row L2 pada contoh yang identik.

Untuk model ID h, pilih resep hanya dari16 ID lainnya. Pemilihan tidak
menerima statistik/label query h atau label dukungan h sebagai input.
Setelah pilihan dikunci, preprocessing dan kepala target hanya memakai
dukungan K pada h. Tidak mengeluarkan kondisi gagal atau memilih seed.

## Dua normalisasi, sembilan alpha

Urutan resep: `row_l2` kemudian `support_z_l2`, masing-masing pada alpha
`1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100`.

- `row_l2`: L2 per baris pada fitur float64, persis baseline pilot.
- `support_z_l2`: mean dan population-SD per fitur dipelajari hanya dari
  dukungan ID/draw/K bersangkutan. SD≤1e-12 diganti1, tanpa clipping query.
  Transform `(x-mean)/scale`, lalu L2 per baris. Query tidak memengaruhi
  mean, scale, atau parameter kepala. Ini scaler eksplisit; tidak memakai
  aturan deteksi near-constant implisit yang dapat berbeda antar pustaka.

Semua kepala memakai objektif RidgeClassifier, intercept aktif, target
biner−1/+1 per kelas, tanpa class/sample weights. Tidak menambah label
validasi target. Resep mencakup baseline `row_l2,alpha1`.

## Tahap validasi sumber dan penggunaan kembali komputasi

Untuk setiap ID, encoder, K dan draw, hitung18 kandidat dari dukungan
Phase1 dan nilai pada query Phase2. Setiap kandidat ID i adalah data
validasi sumber untuk16 fold h≠i. Karena itu kandidat seluruh ID memang
dihitung; tidak menyatakan bahwa query setiap ID tidak pernah disentuh
selama keseluruhan studi. Larangan yang ditegakkan adalah **baris ID h
tidak masuk pemilihan resep model h**.

Cache per-ID ini boleh dipakai ulang lintas fold karena head kandidat
ID i bergantung hanya pada dukungan i dan resep yang ditetapkan, tanpa
fit lintas ID atau parameter dari query mana pun. Jangan menghitung atau
memilih oracle resep terbaik dari query ID h. Tidak mengadaptasi bobot
encoder, memilih keluarga fitur, atau memperluas grid dari hasil luar.

Gunakan eigen decomposition Gram dukungan yang dicenter untuk menghitung
path9 alpha; clamp eigenvalue negatif karena roundoff ke0. Simpan seluruh
skor query dan hitungan benar dukungan/validasi. Untuk baseline alpha1,
bandingkan path dengan hasil pilot (atol/rtol1e-8), lalu pakai ulang skor
pilot persis agar tie numerik tidak mengubah pembanding. Ini2.448 kondisi
×18 resep=44.064 kandidat; 2.448 baseline dipakai ulang,41.616 kandidat baru.
Normalisasi2×2.448 memerlukan4.896 faktorisasi Gram, bukan44.064.

## Pemilihan per keluarga dan outer ID

Satu resep dipilih bersama untuk tiap `(held_ID, keluarga, K)`.
Keluarga MFCC/BEATs merata-ratakan tiga draw. Keluarga ResNet pralatih/
acak merata-ratakan tiga model×tiga draw; tidak memilih seed terbaik.
Semua ID sumber mendapat bobot sama. Karena query/seed/draw lengkap dan
seimbang, maksimalkan jumlah benar integer di seluruh16 ID sumber.
Jika seri, pilih alpha terbesar, lalu pilih `row_l2` jika masih seri.

Simpan408 pilihan (340 normal +68 kontrol) beserta16 ID sumber dan
total kebenaran tiap resep. Fungsi seleksi menolak ID h, ID ganda,
ID hilang dan bentuk metrik salah. Kunci semua pilihan ke `selection.json`
sebelum tahap refit/evaluasi luar. Pemilihan tidak dibuka kembali setelah
melihat hasil ID h.

Pada tahap akhir, fit ulang2.448 kepala terpilih menggunakan sklearn
RidgeClassifier cholesky dengan normalizer terpilih pada dukungan h saja.
Simpan koefisien/intercept/mean/scale/skor query dan dukungan. Baseline
pilot tetap dipertahankan. Evaluasi K0 tidak berubah dan dipakai ulang.

## Kontrol dan biaya label

Kontrol negatif tetap K10 dengan permutasi support-label yang sama dengan
pilot; query true-label tidak diubah. Kontrol melewati validasi16 ID,
pemilihan18 resep dan refit yang sama. Jangan memberi kontrol resep yang
dipilih dari label normal, atau hanya mengacak label setelah seleksi.

Budget target per percobaan tetap12×K label classifier. Untuk setiap
pilihan, laporkan label sumber **unik** yang dipakai kriterianya:
union dukungan tiga draw pada16 ID ditambah960 query unik (16×60).
Jangan mengalikan jumlah label karena seed/encoder/resep memakai ulang
contoh sama. Laporkan pula union sumber untuk seluruh K dan ID h,
serta720 label Harrison yang sudah dipakai ResNet pralatih. Biaya generic
pretraining BEATs tidak disamakan dengan0 label total.

Sumber dan target berganti peran antar fold; overlap fold dicatat. Biaya
source-validation ini merupakan tambahan terhadap baseline alpha1,
bukan perbaikan gratis pada budget total anotasi. Anotasi log untuk crop,
filter dan stratifikasi masih di luar12×K, seperti pada pilot.

## Verifikasi dan pelaporan

Tes sebelum validasi baru: path Gram vs RidgeClassifier pada dua
normalisasi dan seluruh alpha; label kelas lengkap dan degenerate
feature; preprocessing support-only; penolakan overlap source/held ID;
independensi pilihan h dari perubahan baris metrik h; tie rule dan label
cost union; kontrol harus memilih resepnya sendiri.

Audit ulang semua44.064 kandidat terhadap sklearn, skor baseline persis,
408 pilihan dan2.448 refit. Bandingkan koefisien/skor final dengan dual
centered solve independen setelah transform support-only. Toleransi
parameter/skor final: atol1e-6,rtol1e-5; simpan maksimum error dan setiap
perubahan prediksi numerik, jangan disembunyikan. Audit statistik scaler
dengan rumus population variance independen. Hindari dekompresi NPZ
berulang per kandidat: baca array yang dibutuhkan satu kali per shard.

Gunakan fitur CPU yang sudah diekstrak lengkap pada audit pilot, dipatok
hash, sebagai kontrol independen input numerik; tidak mengklaim ekstraksi
CPU baru. Refit/evaluasi semua kepala terpilih pada fitur CPU dan laporkan
perubahan prediksi. Tidak mengubah pilihan resep dari kontrol CPU.

Laporan mempertahankan semua keluarga, K, seed, draw, source/held role,
pilihan di batas grid dan kontrol. Mean serta sample-SD antar17 ID setelah
rerata predictor di dalam ID; CI bootstrap10.000 ID seed7471 dan delta
berpasangan ke baseline. Karena fold berbagi data sumber dan dataset
sudah diperiksa, CI bersyarat/deskriptif, bukan konfirmasi generalisasi
ke populasi mikrofon/sesi baru. Laporkan fit dukungan agar kenaikan
training accuracy tidak disamakan dengan transfer.

Jika mean melewati80%/90%, sebut hanya K terkecil pada grid ini, dengan
CI dan source-label cost; bukan minimum label pasti atau angka end-to-end.
Simpan protokol, hash, tabel validasi, semua pilihan, prediksi per contoh,
kurva dan ringkasan yang dapat direproduksi. Metadata tetap belum
menetapkan mikrofon/keyboard/sesi fisik baru atau klasifikasi alfabet penuh.

Rujukan numerik: [Scikit-learn 1.9.1, Ridge regression/classification]
(https://github.com/scikit-learn/scikit-learn/blob/1.9.1/doc/modules/linear_model.rst)
dan implementasi Cholesky dari pustaka yang terpasang. Audit membandingkan
path Gram dengan implementasi tersebut sebelum menafsirkan hasil.
