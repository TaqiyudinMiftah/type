# Hasil tahap dataset eksternal

Eksperimen 7 November 2026 selesai: kajian 10 penelitian, audit dataset asli,
delapan kontrol SVM, dua pelatihan CoAtNet, dan replay seluruh checkpoint di CPU.
Tahap ini sepenuhnya menggunakan data eksternal.

## Hasil utama

Dataset [Harrison 2023](https://github.com/JBFH-Dev/Keystroke-Datasets):
36 tombol `0–9, a–z`, 900 segmen ponsel dan 900 Zoom. CoAtNet dilatih
**1.100 epoch per kanal pada AMD Radeon RX 6600 LE**, sekitar 32,5 menit total.
Radeon 780M dikeluarkan dari konfigurasi perangkat aktif.

| Kanal | MFCC–SVM, rerata ± SD tiga split | CoAtNet, benar/uji | Top-1 CoAtNet | Macro-F1 | Top-5 | Angka paper |
|---|---:|---:|---:|---:|---:|---:|
| Ponsel | 84,26% ± 3,06 poin | 172/180 | **95,56%** | 0,9533 | 100,00% | 95% |
| Zoom | 71,11% ± 3,89 poin | 168/180 | **93,33%** | 0,9339 | 99,44% | 93% |

Angka acuan berasal dari [paper Harrison, Toreini & Mehrnezhad](https://arxiv.org/abs/2308.01074).
Percobaan independen ini mendekati angka headline pada kedua kanal. Selisih
+0,56 dan +0,33 poin belum membuktikan keunggulan, terutama karena detail
implementasi dan pembagian datanya berbeda. Interval Wilson 95% untuk akurasi
adalah 91,48–97,73% pada ponsel dan 88,71–96,15% pada Zoom; ini pendekatan
binomial pada contoh yang masih berasal dari kumpulan rekaman yang sama.

Checkpoint ponsel dipilih pada epoch 655 (validasi 100%), Zoom pada epoch 1.020
(validasi 95,83%). Test set tidak digunakan untuk memilih checkpoint.

## Apa yang dipelajari

1. **Bunyi tombol cukup informatif dalam kondisi rekaman yang sama.** CoAtNet
   menghasilkan skor lebih tinggi daripada kontrol SVM pada percobaan ini.
   SVM tetap berguna sebagai pembanding murah; pencarian enam konfigurasinya
   memerlukan sekitar 0,24 detik setelah fitur tersedia.
2. **Perpindahan kanal gagal mempertahankan skor.** Model ponsel mencapai
   4,67% pada seluruh 900 contoh Zoom; model Zoom mencapai 3,78% pada ponsel.
   Peluang acak seragam adalah 2,78%. Perubahan ini mencakup mikrofon,
   pengolahan audio, sample rate, batas frekuensi mel, dan durasi potongan,
   sehingga belum dapat diatribusikan hanya pada codec.
3. **Urutan pengambilan data berpengaruh pada baseline.** SVM pada lima ketukan
   terakhir per tombol mencapai 82,22% untuk ponsel dan 61,11% untuk Zoom.
   Split ini tetap berada dalam file yang sama, sehingga bukan uji lintas sesi.
4. **Isolasi dan definisi metrik perlu diaudit.** Satu rekaman, `phone/J`,
   memerlukan perluasan pencarian ambang; jumlah 25 segmen tidak membuktikan
   semua label/onset benar. Kasus itu ditandai, bukan disembunyikan.
5. **Angka paper tidak selalu dapat dibandingkan langsung.** Kajian membedakan
   top-1 tombol, keberhasilan kata, pemulihan karakter berbantuan bahasa,
   dan kemiripan semantik. Paper yang ditarik tidak dijadikan benchmark.

## Batas reproduksi

Kode pelatihan lengkap penulis tidak ada dalam repositori publik. Percobaan
memakai backbone yang dikutip penulis, ditambah pooling dan head 36 kelas;
asumsi kanal input, normalisasi, resize, dan pemilihan model dicatat.

Paper final memakai split `Random`, sedangkan percobaan ini memakai **acak
terstratifikasi**, 18/2/5 contoh per kelas untuk latihan/validasi/uji. Hop 225
mengikuti tabel paper, yang berbeda dari angka pada uraiannya. Ekstensi isolasi
`J` juga merupakan modifikasi. Karena itu, hasil ini disebut percobaan
reproduksi independen, bukan reproduksi persis.

Satu tombol hanya memiliki satu file per kanal. Identitas tombol dan file
berimpit; belum ada sesi, pengguna, atau keyboard independen untuk uji.
CoAtNet baru memakai satu seed per kanal. Hasil ini belum mengukur deteksi
ketukan tanpa bantuan jumlah, CER kalimat alami, atau kinerja pada keyboard
pengguna. Audit menemukan nol duplikat persis dan nol jendela tumpang tindih;
keduanya tidak menggantikan pengujian pada rekaman baru.

## Artefak dan verifikasi

- [Laporan lengkap, grafik, dan confusion matrix](../reports/harrison/laporan.md).
- [Prediksi per contoh dan potongan audio kesalahan](../reports/harrison-cases/README.md).
- [Snapshot metrik ringan](results/harrison-2026-11-07.json).
- [Kajian penelitian](LITERATURE.md) dan [catatan implementasi](REPRODUCTION_NOTES.md).
- [Kode eksperimen](../src/ketik_audio/harrison.py) dan [perintah reproduksi](../README.md).

Replay CPU menghasilkan confusion matrix uji dan transfer yang **sama persis
untuk seluruh 10 konfigurasi**, tanpa pelatihan ulang. Verifikasi perangkat
lunak: 26 tes lulus, Ruff lulus, dan paket berhasil dibangun. Model, split,
normalisasi, history, versi software, commit sumber, serta hash tersedia di
`runs/harrison-reproduction/`.

Prioritas eksperimen lanjutan adalah frontend seragam, audit onset pada subset,
dan pengujian lintas sesi/perangkat dari dataset eksternal. Dataset yang akan
dikumpulkan pengguna selanjutnya menjadi pengujian terpisah dengan sesi uji
yang ditahan sampai konfigurasi model ditetapkan.

Kelanjutan tahap ini sudah dijalankan dalam
[pilot kalibrasi kanal](FEWSHOT_OBSERVATIONS.md): frontend seragam, CoAtNet
scratch/adaptasi dan BEATs beku, 94 kondisi utama serta dua diagnostik
optimisasi. Hasil baru dipisahkan dari tabel reproduksi di atas. Batas
pelatihan scratch dan satu kegagalan optimisasi mencegah kesimpulan peringkat
arsitektur atau efisiensi label yang terlalu kuat.
