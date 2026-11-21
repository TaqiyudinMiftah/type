# SKAID: koreksi drift dan kandidat pilot kalibrasi

Tanggal lokal: 9 November 2026. **Koreksi offset+drift meningkatkan pasangan
yang lolos kriteria waktu sementara dari 44 menjadi 48 dari 50.** Setelah
memeriksa batas audio dan event tetangga, ada 17 ID yang cukup untuk pilot
12 huruf, K1/2/3/5/10, dengan ≥5 query per kelas pada rekaman Phase 2.
Belum ada pengukuran akurasi pengenalan huruf pada SKAID.

## Pembanding yang dikendalikan

[Protokol](SKAID_DRIFT_PROTOCOL.md) ditetapkan sebelum fit drift, setelah
diagnostik offset lama dibaca. Tiga metode memakai onset energi dan
template event yang sama. Delapan blok paruh awal log menentukan offset
lokal; parameter diperiksa dengan gate fit dan fallback yang ditetapkan
sebelumnya. Semua 50 pilihan nyata dan 950 kontrol terkunci sebelum
evaluasi pola waktu bagian akhir. Nama huruf tidak masuk fit.

| Metode | Pasangan lolos /50 | ID kedua fase lolos | Median r akhir, semua 50 |
|---|---:|---:|---:|
| Offset lama, grid 10 ms | 44 | 20 | 0,3018 |
| Offset diperhalus, tanpa drift | 45 | 21 | 0,3291 |
| Kebijakan offset+drift | 48 | 23 | 0,4305 |

Baseline lama, kontrol dan flag cocok pada representasi evaluasi bersama.
Koreksi drift memberi manfaat tambahan di atas sekadar memperhalus offset.
Terhadap offset diperhalus, korelasi akhir membaik pada 41 pasangan,
tetap pada enam, dan menurun pada tiga. Kebijakan tidak memilih metode
per rekaman dari hasil akhir. Terhadap legacy, 44 membaik dan enam tetap.

Affine dipakai pada 44 pasangan; enam memakai fallback karena blok fit
layak tidak cukup. Rentang drift admitted −50,41 sampai +131,97 ppm;
ini parameter pencocokan waktu, bukan pengukuran jam perangkat keras.
Pair45, Phase 2, menunjukkan dampaknya pada rekaman sekitar 25 menit:
r akhir −0,0759→0,4279 setelah drift≈−41,56 ppm dipelajari dari bagian awal.

Dua pasangan belum lolos: `qt_1749848552186` Phase 2 dan
`qt_1753464926271` Phase 1. Keduanya tetap memakai baseline karena tidak
memiliki blok fit yang layak untuk estimasi drift. Hasil tidak dibuang
atau diselaraskan berdasarkan akurasi classifier.

## Kelayakan jendela

Tabel memakai 23 ID dengan dua fase lolos pada kebijakan affine,
**K10 per kelas Phase 1 dan ≥5 query per kelas Phase 2**, untuk subset
eksploratif `e,t,a,o,i,n,s,h,r,d,l,u`.

| Crop / sebelum press | Lengkap | Tanpa press tetangga | Tanpa event tombol lain |
|---|---:|---:|---:|
| 400/50 ms | 21 | 0 | 0 |
| 200/30 ms | 21 | 8 | 1 |
| 120/20 ms | 21 | 16 | 11 |
| 80/15 ms | 21 | 21 | 17 |

Semua jendela harus lengkap di PCM native. Filter terakhir juga
menyingkirkan release milik tombol lain/tidak berpasangan dan meminta
release target berpasangan positif pada fase sama. Release target
sendiri boleh ada. Tombol di luar subset tetap dihitung sebagai tetangga.
Perhitungan memakai interval half-open pada frame native terdekat.

Tidak adanya event lain di log bukan bukti bebas noise atau ekor suara
ketukan sebelumnya. Ketelitian fitur fit masih 10 ms; onset fisik setiap
event belum dianotasi manual. Jendela 80 ms paling mencukupi contoh pada
grid ini, tetapi belum terbukti terbaik untuk klasifikasi. Ia juga
memotong ekor suara, sehingga frontend/model memerlukan protokol baru.

Untuk alfabet 26 huruf, syarat ≥5 query per kelas tidak dipenuhi satu ID pun pada
crop 80 ms, bahkan sebelum menyaring event tetangga. Pilot 12 kelas tidak
boleh dinyatakan sebagai pengenalan seluruh alfabet atau jumlah label
minimum untuk semua tombol.

## Verifikasi

Audit mengulang **1.000 pipeline clock** (50 nyata + 950 kontrol),
**150 evaluasi metode**, semua mask dan agregat jendela. Seluruh hasil
identik. Regresi dibandingkan dengan weighted least squares independen
(selisih maksimal 5,46e-12 bin); **30.000 interval** diperiksa dengan scan
event langsung. Kepemilikan release diperiksa melalui pencarian release
berikutnya per tombol. Seluruh 95 tes proyek, Ruff lint/format, lock dan
build wheel/sdist lulus. Semua pekerjaan tahap ini menggunakan CPU.

- [Laporan dan grafik](../reports/skaid-drift/laporan.md).
- [Pilihan sebelum evaluasi](../runs/skaid-drift/selection.json) dan
  [hasil per pasangan](../runs/skaid-drift/timing.csv).
- [Kelayakan seluruh K/kosakata/jendela](../runs/skaid-drift/label-feasibility.csv).
- [Audit](../reports/skaid-drift-audit/audit.json) dan
  [snapshot](results/skaid-drift-2026-11-09.json).

## Kelanjutan

Kunci pilot 12 kelas dengan 17 ID, crop 80 ms, K=1/2/3/5/10 dan query tetap
seimbang dari Phase 2 sebelum membuka akurasi. Bandingkan representasi
akustik pada dukungan unik yang sama, serta kontrol label. Akui penggunaan
timestamp acuan dan ketidakpastian onset; evaluasi detektor dilakukan
terpisah. Kedua fase merupakan rekaman berbeda, tetapi metadata belum
menetapkan mikrofon, keyboard atau sesi baru.

Studi ini eksploratif karena pola waktu dataset sudah diperiksa. Seluruh
audio tanpa identitas huruf dipakai untuk transformasi/pencarian, dan
kontrol 19/99 belum dikoreksi untuk banyak perbandingan. Hasil membuktikan
perbaikan pencocokan waktu pada setup ini, belum performa classifier,
status SOTA, atau generalisasi ke mikrofon baru.
