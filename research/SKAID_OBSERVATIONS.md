# SKAID: hasil audit dan sinkronisasi awal

Tanggal lokal: 9 November 2026. **44 dari 50 pasangan audio–log yang
dapat diperiksa memenuhi kriteria waktu sementara.** Ada 20 ID dengan
kedua fase lolos, termasuk 17 ID tanpa initial press yang diproyeksikan
di luar audio. Ini membuka kandidat uji kalibrasi pada rekaman terpisah,
setelah ketepatan segmentasi diperiksa. Belum ada akurasi klasifikasi baru.

## Apa yang telah dijalankan

Arsip audio publik [SKAID v1](https://zenodo.org/records/17282184)
diunduh dengan checksum penyedia dan SHA256. Audit native membaca CRC
seluruh anggota ZIP serta mendekode **53 file** pada laju aslinya:
52 M4A mono 48 kHz dari 26 ID × dua fase, ditambah satu WAV mono 16 kHz.
Semua berhasil didekode dan finite. Sumber berlisensi CC BY 4.0.

Jumlah log sebenarnya 27 ID, 202.510 event. `qt_1750019330494` tidak
memiliki audio; `qt_1753467140611` memiliki 7.452 event dengan satu
timestamp unik. Dua ID tersebut tidak masuk diagnostik waktu.
WAV tambahan memiliki panjang sama dan waveform sangat mirip dengan
M4A pasangannya setelah resample (r=0,991565), tetapi gagal ambang ketat
salinan resample. WAV tidak dijadikan pengambilan independen.

[Protokol native](SKAID_AUDIT_PROTOCOL.md) ditulis sebelum dekode;
[protokol waktu](SKAID_ALIGNMENT_PROTOCOL.md) ditulis setelah inventaris,
sebelum korelasi. Pola initial press dari paruh waktu awal log dipakai
untuk memilih satu offset terhadap onset energi 10 ms. Autorepeat yang
ditandai state press/release dikeluarkan; nama tombol tidak masuk
korelasi. Offset tersebut tetap pada pemeriksaan pola waktu paruh akhir.
Seluruh audio tanpa label digunakan untuk transformasi dan pencarian;
ini bukan holdout audio yang sama sekali belum dilihat.

Setiap pasangan mendapat 19 kontrol pengacakan gap dengan pencarian
ulang, serta 99 kontrol pada offset tetap di bagian validasi. Flag lolos
juga memerlukan coverage≥80%, korelasi fit/validasi≥0,10, margin
puncak≥0,02, ≥25 press per bagian dan selisih lag lokal≤30 ms.
Resolusi kontrol terbatas dan belum dikoreksi untuk banyak perbandingan;
flag bersifat diagnostik, bukan sertifikasi setiap onset.

## Hasil dan batas

Median korelasi kelompok 44 pasangan yang lolos: **0,374 fit** dan
**0,312 validasi**. Pergeseran lag validasi lokal −20 sampai +20 ms.
Enam pasangan gagal ambang korelasi; kontrol dan coverage juga gagal
pada sebagian di antaranya. Beberapa rekaman menunjukkan perubahan
lag bertahap 10–40 ms. Model offset tunggal belum memperbaiki drift.
Kegagalan ini belum membuktikan log salah atau suara tombol tidak ada.

Lima pasangan yang lolos masih memiliki press di luar rekaman
(1, 80, 1, 1, dan 3 event). Press tersebut tidak menjadi contoh audio.
Bin yang berada dalam audio juga belum menjamin jendela crop lengkap.

Hitungan awal label setelah proyeksi waktu, untuk 20 ID dengan kedua
fase lolos: setiap kelas memerlukan ≥K initial press Phase 1 dan ≥1
Phase 2. Pemeriksaan kelengkapan crop dan overlap belum diterapkan.

| K | ID memenuhi 26 huruf | ID memenuhi subset 12 huruf eksploratif |
|---:|---:|---:|
| 1 | 3 | 20 |
| 2 | 2 | 20 |
| 3 | 0 | 20 |
| 5 | 0 | 20 |
| 10 | 0 | 20 |

Subset eksploratif: `e,t,a,o,i,n,s,h,r,d,l,u`. Ini diperiksa setelah
kelayakan alfabet penuh dibaca, sebelum akurasi model; belum menjadi
kosakata eksperimen yang dikunci. Pada 17 ID tanpa press di luar audio,
seluruhnya memenuhi hitungan 12 kelas, tetapi hanya dua memenuhi
26 huruf K1. Hasil 12 kelas kelak tidak boleh disebut pengenalan seluruh
alfabet. K10 pada 12 kelas berarti 120 label unik per ID.

Median jarak press berurutan antar-rekaman lolos adalah **136,5 ms**;
median fraksi gap<400 ms per rekaman **90,85%**. Memindahkan crop
400 ms Harrison akan sering memasukkan ketukan tetangga. Segmentasi dan
aturan overlap perlu ditetapkan sebelum membandingkan model.

Metadata hanya menyebut kategori Desktop/Laptop dan ID log/fase, belum
ID keyboard/mikrofon fisik atau sesi baru. Phase 1 dan Phase 2 merupakan
M4A terpisah, tetapi tidak otomatis menguji adaptasi ke mikrofon baru.
Prosedur ini juga belum mereproduksi klasifikasi EchoStrike: paper
menjelaskan penyesuaian offset manual, sedangkan tabel offset belum
ditemukan pada arsip audio/log, README dan metadata yang diperiksa.

## Verifikasi dan artefak

Semua 50 pasangan diputar ulang dan seluruh metrik/array identik.
Rekonstruksi state press, konvolusi template independen dan korelasi
langsung ikut diperiksa. Selisih korelasi FFT vs dot product maksimal
2,46e-14. Ini memverifikasi perhitungan, belum ketepatan onset fisik.
Seluruh 90 tes proyek, Ruff lint/format, pemeriksaan lock dan build
wheel/sdist lulus. Tahap ini sepenuhnya memakai CPU.

- [Laporan, grafik dan batas lengkap](../reports/skaid/laporan.md).
- [Ringkasan native](../reports/skaid-audit/summary.json),
  [timing per pasangan](../reports/skaid-alignment/timing.csv), dan
  [audit replay](../reports/skaid-alignment-audit/audit.json).
- [Snapshot](results/skaid-audit-2026-11-09.json) menyimpan hash dan hasil.

Langkah berikutnya adalah membandingkan offset tunggal dengan koreksi
offset+drift yang dipelajari pada bagian awal saja. Sesudah pemeriksaan
onset dan crop, kunci kosakata, label budget dan split Phase 1→Phase 2
sebelum mengukur klasifikasi. Hasil lama Harrison dan MKA tetap sebagai
eksperimen terpisah dengan protokolnya masing-masing.
