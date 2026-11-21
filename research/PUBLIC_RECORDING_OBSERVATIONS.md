# Temuan pada rekaman publik terpisah

15 November 2026. Resep MFCC+Ridge SKAID yang sama tidak mempertahankan
akurasinya pada dua korpus baru. Pada NC1, kalibrasi pada kanal codec
yang cocok membantu, tetapi hasil terbaik tetap di bawah 46% top-1.

## Data yang benar-benar tersedia

[Rilis HEAR v1.0](https://github.com/vikrant-vikram/Hear-Dataset/releases/tag/v1)
publik hanya memuat rekaman on-device. Arsip 768.636.949 byte cocok
dengan SHA256 penyedia. Audit mencakup 73 paket, 36 ID peserta,
36 token perangkat, dan 37.452 WAV. Lima belas ID punya beberapa paket;
tidak ada ID dengan beberapa token perangkat. Sebanyak 69 paket cocok
unik dengan `tracking.csv`. Tujuh paket memuat total 4.595 log event
melewati durasi audio menurut metadata.

Semua 37.452 WAV cocok dengan manifest dan log. Dari jumlah itu,
27.524 lolos syarat crop 80 ms; 22.110 juga cocok dengan PCM induk
pada toleransi yang dikunci. Untuk seluruh label, 8.700 gagal ambang
RMS, 1.226 memiliki keydown tetangga, dan dua crop/export tidak utuh.
Di antara crop yang lolos tahap kedua, 1.990 tidak cocok rate/channel
induk; 3.424 tidak cocok PCM dalam batas yang ditentukan. Kecocokan
potongan dengan audio induk tidak membuktikan onset fisik atau
ketiadaan release tetangga.

| Tier HEAR, K10 + lima query per huruf | ID | Pasangan 12 huruf | Pasangan 26 huruf |
|---|---:|---:|---:|
| Export cocok log | 5 | 43 | 0 |
| Crop 80 ms lolos filter | 4 | 19 | 0 |
| Crop cocok audio induk | 3 | 18 | 0 |

[Novox NC1](https://doi.org/10.5281/zenodo.17536616) menyediakan
4.030 WAV 48 kHz mono dan tanggal di nama file. Semua tanggal cocok
dengan tanggal pada ZIP. Ada **44 pasangan PCM identik lintas kelompok
kondisi**; keluar seluruh 88 salinan, menyisakan 3.942 klip. Dokumen
eksperimen menyebut mikrofon Novox NC1 dan perubahan keyboard/latar.
Tanggal digunakan sebagai kelompok rekaman, tanpa mengklaim setiap
klip pada hari yang sama adalah sesi tersendiri. Rujukan:
[Rzemieniuk dkk.](https://doi.org/10.3390/app16020563).

## Apakah kalibrasi membantu?

Resep row L2 dan Ridge alpha 0,001 ditetapkan dari SKAID sebelum
evaluasi. Pada HEAR, model yang dilatih di SKAID mendapat **10,00%**
tanpa label HEAR. Kalibrasi K10 dari sesi lain mendapat **11,48%**,
dengan rentang antar tiga ID 10,00–14,44%; kontrol label acak 9,07%
dan peluang uniform 8,33%. Mean semua 18 pasangan, dirata-ratakan
per ID lebih dahulu, 11,80%. Delapan pasangan dengan jarak awal sesi
≥12 jam meliputi dua ID dan mencapai 8,73%. Ini tidak menunjukkan
transfer yang kuat untuk resep tersebut. Tidak ada estimasi populasi
dari tiga ID atau klaim bahwa metode lain juga akan gagal.

Di NC1, kepala dilatih dari Fnatic tenang 10 Juli dan diuji pada
kelompok 19 Juli. Model selalu mengeluarkan 26 kelas. Quiet mempunyai
25 huruf uji setelah penghapusan duplikat (`e` tidak tersisa).

| Kondisi NC1, K10 | Query unik | Top-1 | Macro recall 25 huruf sama |
|---|---:|---:|---:|
| Fnatic tenang | 487 | 44,42% | 45,15% |
| Fnatic bising | 815 | 32,60% | 32,77% |
| Anne Pro tenang | 795 | 3,44% | 3,45% |

Bising/Anne tidak menerima label kalibrasi dari kondisi itu. Karena
data NC1 hanya satu peserta dan faktor hari/kondisi tidak dikontrol
penuh, tabel tersebut tidak mengisolasi pengaruh keyboard atau noise.

## Kontrol kanal pada klip yang sama

Pada Fnatic tenang, klip dukungan dan query tetap dari 10 dan 19 Juli.
Setiap klip punya versi asli dan versi 16→8 kHz, µ-law, lalu 8→16 kHz.
Ini kontrol perubahan pemrosesan offline, bukan eksperimen mikrofon
atau pengujian sistem VoIP lengkap. Kalibrasi yang cocok dengan kanal
uji memberi perbaikan pada ketiga draw di semua K yang diuji.

| K | Asli→asli | Asli→µ-law | µ-law→µ-law | Manfaat kalibrasi cocok |
|---:|---:|---:|---:|---:|
| 1 | 23,27% | 4,24% | 22,72% | +18,48 pp |
| 2 | 25,94% | 6,37% | 29,09% | +22,72 pp |
| 3 | 32,72% | 9,51% | 34,77% | +25,26 pp |
| 5 | 38,54% | 9,58% | 41,62% | +32,03 pp |
| 10 | 44,42% | 6,43% | 45,65% | +39,22 pp |

Pada K10, perbaikan berpasangan berada di rentang +38,19 hingga
+39,84 pp antar draw. Draw berbagi klip uji; rentang ini bukan CI.
Anggaran fit K10 ialah 120 label per model HEAR dan 260 per model
NC1. Label berasal dari pool yang sudah berlabel. Resep memakai
6.137 label pengembangan SKAID historis, termasuk 5.117 yang dipakai
ulang untuk fit K0 HEAR. Versi codec tidak menambah label unik.

## Verifikasi dan implikasi

Audit re-dekode 5.164 klip terpilih dan 1.154 versi codec, memeriksa
split dan kontrol, serta replay 181 kepala pada 450 kondisi. Selisih
skor maksimal 1,36e−13, tanpa perubahan prediksi. Seluruh 133 tes
lulus. Semua eksperimen memakai CPU, tanpa ekstraksi neural baru.

Kecocokan kanal ternyata penting, bahkan saat event ketikan dan
jumlah anotasi tetap. Namun, K≤10 belum mendukung akurasi 80%/90%
di kedua korpus pada protokol ini. Berikutnya, uji berpasangan dengan
dua mikrofon fisik dan hari kalibrasi/uji berbeda diperlukan untuk
menjawab kebutuhan label pada mikrofon baru. Sebelum membandingkan
fitur baru pada HEAR, audit onset manual dapat memisahkan masalah
penempatan crop dari keterbatasan representasi. Hasil sekarang tetap
klasifikasi klip; belum mengukur deteksi atau akurasi teks kontinu.

- [Audit rekaman](PUBLIC_RECORDING_AUDIT_PROTOCOL.md) dan
  [protokol baseline](PUBLIC_RECORDING_BASELINE_PROTOCOL.md).
- [Laporan, kurva dan biaya label](../reports/public-recording-baseline/laporan.md).
- [Inventaris HEAR](../reports/hear-recording-audit/summary.json),
  [inventaris NC1](../reports/nc1-recording-audit/summary.json),
  dan [audit skor](../reports/public-recording-baseline-audit/audit.json).
- [Snapshot](results/public-recording-baseline-2026-11-15.json).
