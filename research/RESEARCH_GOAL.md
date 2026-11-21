# Tujuan riset pengenalan tombol dari suara ketikan

Pertanyaan utama: **berapa contoh berlabel per tombol yang diperlukan
agar model dapat mengenali tombol dari bunyi ketikan pada rekaman baru?**
Yang ingin diukur adalah kurva akurasi terhadap biaya kalibrasi target,
serta seberapa jauh kurva itu berubah ketika mikrofon atau kanal berubah.

Mulai dari kajian penelitian, reproduksi independen dan dataset publik.
Gunakan hasil tersebut untuk memilih baseline, memeriksa kebocoran dan
menetapkan protokol. Setelah tahap publik, rekaman yang dikumpulkan
pengguna dapat dipakai untuk uji terpisah. Jangan menentukan resep model
dari bagian uji tersebut.

## Bukti yang diperlukan

1. **Pisahkan kondisi rekaman.** Catat identitas perangkat, mikrofon,
   kanal, pengguna dan sesi. Tahan rekaman uji yang terpisah dari
   kalibrasi. Bila pertanyaannya efek mikrofon, usahakan samakan faktor
   lain atau sebutkan faktor yang berubah bersama.
2. **Ukur biaya dan hasil pada beberapa K.** Laporkan akurasi top-1 per
   tombol pada K=0 bila kelas/model sumber kompatibel, serta K=1, 2, 3,
   5 dan 10 pada target. Hitung label sumber, label target, anotasi
   segmentasi, dan label evaluasi secara terpisah. Cakupan huruf dan
   keputusan penyaringan data harus ditetapkan sebelum evaluasi.
3. **Pisahkan klasifikasi dari deteksi.** Klasifikasi pada potongan yang
   waktunya diketahui menguji informasi akustik. Bila protokol kemudian
   memakai audio kontinu, ukur ketukan yang terlewat/berlebih dan
   kesalahan pengenalan secara terpisah. Hasil bertimestamp tidak
   boleh disebut pengenalan teks bebas.
4. **Bekukan prosedur sebelum uji independen.** Bandingkan dengan
   baseline sederhana dan kontrol label acak. Laporkan variasi antar
   rekaman, seluruh kondisi yang ditetapkan, dan batas klaim. Ambang
   80% dan 90% adalah acuan membaca kurva, bukan syarat kelulusan riset
   atau alasan mengubah test set.

## Posisi per 14 November 2026

- [Reproduksi Harrison](OBSERVATIONS.md) menunjukkan akurasi tinggi
  pada kanal yang sama. [Kalibrasi kanal](RESNET_FEWSHOT_OBSERVATIONS.md)
  mencapai 83,52% pada lima contoh per tombol, namun masih pada
  kumpulan WAV yang sama.
- [Transfer MKA](MKA_NESTED_OBSERVATIONS.md) mendekati peluang acak,
  menunjukkan keterbatasan resep yang diuji pada rekaman lain.
- [SKAID dengan pemilihan resep dari ID lain](SKAID_SELECTION_OBSERVATIONS.md)
  memberi kurva K1–10 pada 17 ID dan 12 huruf. MFCC+Ridge mencapai
  75,78% pada K10; pemilihan resep membutuhkan 5.749–5.807 label
  sumber per fold di luar 120 label target. Segmentasi berbasis log.
- [Perluasan C SVM](SKAID_SVM_C_OBSERVATIONS.md) memberi 76,63% linear
  dan 76,83% RBF untuk MFCC pada K10. Selisih terhadap Ridge masih
  kecil dan interval berpasangan mencakup nol. MFCC+Ridge tetap
  pembanding utama untuk eksperimen berikutnya.

Tahap publik memberi baseline dan batasnya. Langkah yang paling
informatif berikutnya ialah menyiapkan uji pada rekaman independen
dengan metadata kondisi yang jelas. Sampai ada pengujian itu, kurva
SKAID adalah hasil pengembangan bersyarat, bukan estimasi kebutuhan
label untuk mikrofon baru atau bukti SOTA.

Pembaruan 15 November: [uji pada HEAR dan Novox NC1](PUBLIC_RECORDING_OBSERVATIONS.md)
sudah memisahkan sesi/tanggal kalibrasi dan uji. HEAR mencapai 11,48%
K10 pada tiga ID. Pada NC1 yang tenang, top-1 K10 berubah dari 44,42%
asli→asli menjadi 6,43% asli→µ-law, dan kembali 45,65% jika dukungan
juga µ-law. Perbandingan ini memakai event yang sama, tetapi kanal
diubah offline. Bukti perubahan mikrofon fisik masih belum ada.

[Audit lanjutan](../reports/public-transfer-evidence/laporan.md) menguji
kedua arah pada 487 ketukan NC1 yang sama. Pada K10, kalibrasi yang
cocok memberi +31,90 poin persentase pada audio asli dan +39,22 pada
audio codec. Efek tetap besar ketika satu kelas mana pun dikeluarkan.
Pada HEAR, 18 pasangan hanya mencakup tiga ID; untuk pasangan dengan
jeda ≥12 jam, selisih K10 terhadap K0 berubah tanda saat salah satu
dari dua ID dikeluarkan. Bukti dari simulasi codec belum menjawab
berapa contoh yang dibutuhkan ketika mikrofon fisik berubah.

[Kajian khusus mikrofon](MICROPHONE_EVIDENCE.md) mengonfirmasi bahwa
HEAR lengkap mendokumentasikan pasangan laptop–ponsel, tetapi rilis
publiknya belum memuat kanal ponsel. Tesis Maurya mengaitkan tabel
KEYAC dengan subset HEAR; semua 216 nilai tabel cocok sehingga hasil
itu tidak dihitung sebagai replikasi pada korpus independen. Belum
ada data publik terverifikasi dalam kajian ini yang sekaligus memenuhi
pemasangan ketukan lintas mikrofon, pemisahan hari uji dan anggaran
label yang dibutuhkan untuk perbandingan kita.

Pada ekspor HEAR v1, kanal laptop sendiri tidak punya pasangan sesi
memenuhi K1 dan lima query per huruf untuk seluruh a–z. Menambah
rekaman ponsel dari ketukan yang sama tidak menambah cakupan huruf.
Untuk 12 huruf K10, [batas atas dari metadata](results/hear-session-coverage-2026-11-15.json)
ialah 17 susunan tiga sesi dari dua ID; belum ada susunan yang
terverifikasi mempunyai dua kanal fisik.

[Audit asal data Harrison](HARRISON_PROVENANCE.md) memeriksa kembali
hasil 83,52% pada K5. Semua 30 fit target memiliki 36 WAV induk yang
sama antara kalibrasi dan uji, walau ID potongannya terpisah. Penanda
waktu relatif juga tidak cukup untuk mengidentifikasi ketukan yang sama
antar Zoom dan ponsel. Jadi hasil lama tetap pilot perpindahan jalur,
belum bukti untuk hari baru atau perubahan mikrofon yang diisolasi.

[Audit anggaran SKAID](SKAID_BUDGET_RELIABILITY.md) memperjelas variasi
di balik rerata. Pada K10, Ridge mencapai 80% setidaknya sekali untuk
9/17 ID dan pada ketiga pilihan kalibrasi untuk 4/17 ID. Tidak ada ID
yang mencapai 90% pada ketiga pilihan di salah satu dari tiga metode
yang dibandingkan. Hasil ini tetap pada 12 huruf dan ketukan uji fase
SKAID yang sama, sehingga belum mengukur kebutuhan label mikrofon fisik.

Pemeriksaan tambahan [HERO dan Tietz](MICROPHONE_EVIDENCE.md) belum
memberi tabel biaya label pada pasangan mikrofon dengan hari uji terpisah.
Hasil lintas peserta HERO dan hasil autentikasi Tietz tidak mengisi
kekurangan ini. Rilis HEAR publik masih satu kanal, dan workspace belum
memiliki rekaman fisik berpasangan. Untuk meneruskan pengukuran diperlukan
bukti berpasangan yang baru, misalnya hasil agregat dengan asal sesi,
jumlah ketukan uji, dan jumlah label per kondisi yang dapat diverifikasi.

## Pilot Lenovo LOQ, 21 November 2026

Model Ridge dari kalibrasi `f558fd22` mengenali 84/509 potongan pada rekaman
lama `e9be52b8` dengan metadata mikrofon dan peserta yang sama. Untuk huruf
a–z, hasilnya 27/404 (6,68%). Audit timing rekaman lama masih `needs_review`.
Dalam kurva eksploratif 11 kelas pada WAV lama, akurasi rata-rata per tombol
naik dari 19,76% pada K0 ke 26,70% pada K10 sumber + target. Uji sensitivitas
terpisah menunjukkan bahwa pergeseran crop ±12,5 ms berpengaruh besar.
[Transfer retrospektif](../reports/loq-transfer-e9be52b8-replay/laporan.md) ·
[Sensitivitas posisi potongan](../reports/loq-crop-robustness/laporan.md)

[Protokol prospektif](LOQ_TRANSFER_PROTOCOL.md) menetapkan **Uji pertama**
sebagai dukungan dan **Uji kedua** sebagai rekaman query untuk kanal Realtek.
Pasangan rekaman itu diperlukan untuk mengukur 27 kelas pada sesi terpisah.
Eksperimen perubahan mikrofon memerlukan pasangan target dengan perangkat
fisik berbeda dan catatan kondisi perekaman.

[Kesiapan Uji pertama](../reports/loq-first-session-readiness/laporan.md)
memungkinkan evaluasi K0 dari satu ZIP dengan model yang telah dikunci.
Dari rekaman kalibrasi, 482 ketukan yang sama lolos pada posisi potongan
nominal serta dua skenario offset ±12,5 ms. Jumlah minimum per tombol
adalah 11 pada `x`. Pengukuran K pada data prospektif memakai rekaman
dukungan dan query yang terpisah.
