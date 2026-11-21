# SKAID: koreksi drift dan kelayakan jendela

Tanggal lokal: 9 November 2026. Protokol ditulis setelah membaca diagnostik
offset tunggal (44/50 pasangan lolos), sebelum menghitung atau mengevaluasi
parameter drift. Studi eksploratif pada data yang sudah diperiksa, belum
konfirmasi pada rekaman baru. Semua perhitungan CPU; hasil lama tetap utuh.

## Masukan, pembagian dan urutan kerja

Pertahankan 50 pasangan, event initial press, envelope/onset energi 10 ms,
origin timestamp, batas paruh waktu dan template Gaussian studi sebelumnya.
Empat pasangan yang tidak bisa diperiksa tetap dilaporkan. Jangan memilih
hanya pasangan gagal atau mengubah kosakata berdasarkan akurasi huruf.
Tidak membaca `full_text` atau melatih classifier.

Verifikasi checksum semua masukan/artefak/protokol/kode. Tahap `select`
hanya menerima fitur audio tanpa label, initial press **paruh awal**,
batas fit dan offset kasar hasil fit lama. Nama tombol dan timestamp
paruh akhir tidak menjadi argumen fungsi seleksi parameter. Audio penuh
tetap digunakan untuk transformasi/pencarian, seperti studi sebelumnya;
bagian akhir bukan holdout audio yang sama sekali tak dilihat.

Kunci parameter seluruh pasangan dan kontrol dalam `selection.json`
sebelum menjalankan tahap `evaluate`. `evaluate` tidak memperbarui model.
Kecocokan waktu paruh akhir lama sudah diketahui; pemisahan tahap ini
tidak menjadikan dataset kembali buta.

## Tiga model dengan representasi evaluasi yang sama

Koordinat x adalah awal bin log 10 ms, y adalah koordinat audio dalam
satuan bin: **y = a + (1+b)x**. Parameter drift ppm = b×1e6.

1. **Legacy**: a dari puncak global fit lama, b=0.
2. **Refined constant**: a dari rerata berbobot offset delapan blok fit,
   b=0; jika blok tidak cukup, gunakan legacy.
3. **Affine policy**: a,b dari regresi linear berbobot delapan blok fit,
   jika memenuhi gate berbasis fit saja; selain itu refined constant.

Delapan blok membagi paruh awal log dengan batas floor(cut×q/8).
Template blok memakai event floor(relative_ms/10), Gaussian σ=1 bin,
truncate=3. Cari korelasi Pearson lokal di ±100 ms terhadap offset kasar,
memakai aturan overlap studi lama (≥50% durasi dan ≥50% event).
Puncak blok menjadi layak jika ≥25 press, coverage≥80%, korelasi≥0,10,
margin terhadap puncak lain di luar ±30 ms≥0,02. Jika tiga bin sekitar
puncak layak dan melengkung ke bawah, interpolasi parabola dengan
fraksi dibatasi ±0,5 bin; selain itu fraksi nol.

Absis regresi adalah rerata bin initial press blok yang masuk audio,
ordinat adalah lag lokal dikurangi awal blok, termasuk fraksi parabola.
Bobot adalah jumlah press tercakup. Minimal empat blok layak, dengan
rentang absis≥cut/2 (seperempat jumlah bin fase hingga pembulatan satu
bin), diperlukan untuk refinement. Offset
konstan memakai rerata berbobot ordinat. Fit affine memakai weighted
least squares dengan centering; gate tambahan: |drift|≤200 ppm dan
residual maksimum blok≤15 ms. Jangan clipping slope ke batas dan jangan
memilih model dari validasi. Simpan kandidat affine, gate dan alasan
fallback secara eksplisit. Gate ini keputusan eksploratif, bukan batas
fisika perangkat yang sudah diverifikasi.

Semua model dievaluasi dengan menginterpolasi fitur onset audio secara
linear pada y=a+(1+b)x, tanpa padding audio di luar rentang. Template log
tetap sama. Normalisasi Pearson hanya pada posisi yang tersedia. Legacy
harus cocok dengan metrik lama dalam toleransi numerik, sehingga perubahan
representasi tidak disalahartikan sebagai manfaat drift. Minimal overlap
50% durasi dan 50% event untuk skor yang dianggap layak.

## Kontrol dan diagnostik akhir

Gunakan seed lama 17282184+indeks pasangan. Untuk 19 kontrol fit,
acak gap milidetik seperti sebelumnya; cari offset kasar global kontrol,
lalu fit ulang refined constant dan affine policy dengan aturan sama.
Kontrol harus mengikuti fleksibilitas prosedur yang sedang dibandingkan.
Simpan model kontrol dan state RNG sesudah 19 permutasi dalam lock seleksi.
Evaluasi 99 permutasi gap bagian akhir pada model yang sudah tetap,
menggunakan kelanjutan RNG itu; seluruh metode menerima permutasi sama.

Laporkan distribusi perubahan korelasi fit/akhir, jumlah membaik/menurun,
seluruh fallback dan pasangan gagal. Flag waktu sementara memakai kriteria
lama: ≥25 press per bagian, coverage≥80%, r fit/akhir≥0,10, margin puncak
global fit≥0,02, semua 19 kontrol fit serta 99 kontrol akhir lebih rendah,
dan residual lag lokal validasi≤30 ms. Residual lokal dipindai ±250 ms
dalam langkah 10 ms untuk diagnostik saja, tidak mengubah model terkunci.
Resolusi kontrol terbatas; tidak menyatakan pengujian konfirmatori atau
p-value terkoreksi banyak perbandingan.

## Audit jendela setelah parameter terkunci

Jalankan untuk semua model dan semua pasangan, sambil membedakan pasangan
yang lolos dan gagal flag waktu. Gunakan timestamp event asli (selisih
integer milidetik sebelum float) dan y=a+(1+b)x. Ketelitian fitur fit
tetap 10 ms; ini belum ground truth onset manual pada ketelitian sampel.

Jendela yang ditetapkan sebelumnya: (panjang, sebelum press) dalam ms:
**(400,50), (200,30), (120,20), (80,15)**. Tidak menggeser setiap crop
ke peak terbesar. Proyeksikan event ke frame native terdekat (`np.rint`),
kemudian awal = frame_press − round(pre_ms×rate/1000), akhir = awal +
round(length_ms×rate/1000). Pembulatan frame maksimal setengah sampel;
ketelitian fitur fit tetap 10 ms. Jendela half-open [awal,akhir) harus
seluruhnya berada dalam PCM native. Laporkan tiga tingkatan:

- **Lengkap**: seluruh jendela berada dalam audio.
- **Tanpa press tetangga**: lengkap dan tidak memuat initial press lain,
  termasuk tombol di luar calon kosakata.
- **Tanpa event tombol lain**: syarat sebelumnya, tidak memuat release
  milik press lain atau release tak berpasangan, serta target memiliki
  release berpasangan positif pada fase sama. Release milik target
  boleh berada di jendela. Autorepeat OS bukan press fisik tambahan.

Rekonstruksi pasangan press/release dari urutan event penuh per log;
catat unmatched dan lintas fase. Tidak mengeklaim jendela bebas ekor
suara/noise hanya karena tidak ada event lain dari log.

Hitung label unik per kelas/fase pada setiap tingkatan dan durasi.
Laporkan kelayakan Phase1≥K dan Phase2≥1, juga Phase2≥5 per kelas,
untuk K=1/2/3/5/10, pada alfabet26 dan subset eksploratif12
`etaoinshrdlu`. Jangan menganggap ≥1 query sama dengan evaluasi stabil.
Audit hitungan ini dilakukan sebelum akurasi huruf; pemilihan protokol
klasifikasi berikutnya harus menyebut hasil penyaringan dan perubahan
kosakata. Jangan memindahkan kelas yang tidak ada ke label lain.

## Verifikasi dan keluaran

Tes sintetis harus memeriksa tanda/rate drift, fallback, batas overlap,
ketetapan model sebelum evaluasi akhir, dan perbedaan press/release
tetangga pada interval half-open. Replay semua seleksi/prediksi waktu
dari masukan yang dipatok; periksa regresi terhadap penyelesaian weighted
least squares independen, baseline legacy, dan hitungan jendela langsung
pada sampel event. Simpan NPZ/JSON, CSV per pasangan dan kelayakan K,
grafik sebelum/sesudah serta laporan seluruh kegagalan. Belum mengklaim
jumlah label minimum, model terbaik, atau adaptasi ke mikrofon/sesi baru.
