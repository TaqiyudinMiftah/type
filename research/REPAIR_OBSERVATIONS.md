# Kestabilan baseline: hasil 8 November 2026

Keluaran konstan dan variasi besar dari inisialisasi berhasil diatasi pada
tiga percobaan ponsel yang diuji. Baseline aktif berikutnya adalah
**compact-resnet**, CNN residual 311.892 parameter yang dilatih dari awal.
Model ini menggantikan backbone hybrid kecil untuk kontrol scratch; model
referensi dan seluruh hasil reproduksi lama tetap disimpan.

Pada K=10 (360 label target, 36 kelas), seed dukungan 101 dan 360 event uji
ponsel tetap, hasil akhir tiga inisialisasi adalah:

| Seed | Akurasi dukungan | Top-1 uji | Benar/uji |
|---|---:|---:|---:|
| 42 | 100,00% | 90,00% | 324/360 |
| 43 | 100,00% | 88,89% | 320/360 |
| 44 | 100,00% | 88,33% | 318/360 |

| Baseline scratch | Rerata ± sample SD uji | Run konstan |
|---|---:|---:|
| Referensi ReLU | 28,24% ± 26,29 pp | 1/3 |
| Referensi dengan LeakyReLU terakhir | 50,28% ± 20,59 pp | 0/3 |
| CNN residual kompak | **89,07% ± 0,85 pp** | **0/3** |

Semua memakai dukungan/uji yang sama, tiga seed inisialisasi 42/43/44,
10.000 update, Adam 5e-4, batch 16 replacement dan peluruhan linear ke nol.
Frontend 16 kHz/400 ms, augmentasi dan normalisasi dukungan sama. Hanya
checkpoint terakhir yang dievaluasi, tanpa validasi target atau pemilihan
seed terbaik. Jumlah parameter turun dari 1.973.280 menjadi 311.892.

## Apa yang diperbaiki dan bagaimana dipilih

Autopsi sebelumnya menemukan ReLU terakhir yang mati. Perubahan LeakyReLU
mencegah kasus itu, tetapi variasi antar-inisialisasi tetap besar. Pemeriksaan
berikutnya menemukan proyeksi attention std=0,001 tanpa residual/LayerNorm;
variasi antar-input dan gradien awal sangat kecil. Xavier pada delapan
matriks attention memulihkan kekuatan gradien awal, tetapi belum membuat
validasi konsisten. Residual/pre-norm menguasai dukungan, sementara sebagian
inisialisasi tetap memiliki validasi rendah. Kalibrasi ulang BatchNorm pada
salinan dua checkpoint tidak membantu; replay diagnostik CPU identik.

Selanjutnya diuji stem lebar, lalu alternatif CNN residual kompak. Seluruh
**12 fit pengembangan, empat kandidat dalam tiga tahap protokol**, dilakukan
pada **Zoom saja**: 360 contoh fit dari train sumber lama dan 72 validasi.
Sebanyak 180 test sumber tidak dievaluasi. Statistik input hanya dari fit.
Total label sumber unik untuk pengembangan 432, dipakai ulang antar-kandidat.

Kriteria sebelum fit: setiap seed dukungan >=95%, tidak konstan, rerata
validasi >=50%, sample SD validasi <=10 pp. Kesembilan fit hybrid gagal
seleksi; stem lebar memiliki SD 10,79 pp dan tetap dinyatakan gagal.
CNN kompak mencapai dukungan 100% pada semua seed dan validasi
80,56%/83,33%/81,94% (**81,94% ± 1,39 pp**), sehingga lulus.
Keputusan/hash dikunci sebelum tiga fit dan evaluasi ponsel baru.

CNN kompak memakai stem 16 kanal, empat BasicBlock berlebar 16/32/64/128,
BatchNorm/ReLU, shortcut per blok, global average pooling dan Linear128→36.
BN terakhir cabang residual diinisialisasi nol. Rancangan residual merujuk
[He dkk.](https://arxiv.org/abs/1512.03385), dengan ukuran lokal yang berbeda
dari ResNet-18. Ini alternatif arsitektur dan kapasitas, sehingga perbaikan
tidak membuktikan kontribusi tunggal residual, ukuran, normalisasi atau stem.

## Verifikasi dan batas hasil

- Seluruh 15 checkpoint baru (12 pengembangan, tiga ponsel) direplay di CPU.
  Prediksi individual, confusion matrix, top-1/top-5, macro-F1 dan akurasi
  dukungan cocok. Selisih skor absolut maksimum CPU/GPU sekitar 3,63e-5.
- ID contoh, anggaran update, normalisasi hanya pada dukungan, checksum
  data/kode/protokol dan keputusan sebelum konfirmasi diperiksa. Checkpoint
  serta hasil kontrol lama tetap cocok dengan audit sebelumnya.
- Seluruh fit eksperimen memakai **RX 6600 LE**, tanpa Radeon 780M.
  **46 tes lulus**, Ruff dan format bersih, sdist/wheel berhasil dibangun.

Ini klasifikasi potongan ketukan Harrison dalam rekaman yang sama, bukan
evaluasi deteksi/teks bebas atau sesi baru. Arah diagnosis berasal dari
hasil ponsel historis yang sudah dilihat; seleksi baru pada Zoom tidak
mengubah ponsel menjadi holdout konfirmatori baru. Ketidakpastian isolasi
phone/J masih berlaku. Pengembangan memakai ulang validasi sumber dan
merupakan pencarian arsitektur eksploratif.

SD uji adalah variasi tiga inisialisasi pada **satu** pilihan dukungan,
bukan variasi sesi/pilihan dukungan atau interval ketidakpastian populasi.
Semua hasil CNN baru adalah scratch, bukan transfer tanpa label atau
fine-tuning sumber→target. K=10 saja belum menetapkan jumlah contoh minimum,
dan 89,07% belum berarti target rerata 90% tercapai. Perbandingan dengan
BEATs/pilot lama perlu dukungan dan anggaran yang disamakan; hasil ini tidak
menetapkan SOTA.

Artefak lengkap:

- [Protokol attention](REPAIR_PROTOCOL.md), [stem](STEM_REPAIR_PROTOCOL.md),
  [CNN](CNN_REPAIR_PROTOCOL.md).
- [Snapshot seluruh pengembangan dan konfirmasi](results/repair-2026-11-08.json).
- `reports/repair/laporan.md`, `perbaikan.png` dan `perbaikan.pdf`.
- `reports/repair-audit/audit.json`, `classifications.csv`.
- `runs/cnn-repair-development/decision.json` dan
  `runs/cnn-repair-confirmation/compact-resnet-s{42,43,44}/last.pt`.

Kelanjutan: [kurva efisiensi label CNN](RESNET_FEWSHOT_OBSERVATIONS.md)
membandingkan scratch dengan pretraining Zoom→ponsel pada K=1/2/3/5/10.
Tahap berikutnya memakai tiga pilihan dukungan sekaligus inisialisasi;
SD-nya memiliki cakupan variasi berbeda dari konfirmasi perbaikan ini.
