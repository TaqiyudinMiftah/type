# Baseline terkunci pada rekaman publik yang terpisah

15 November 2026. Ditulis sesudah audit HEAR dan Novox NC1, sebelum
ada hasil classifier untuk kedua dataset. Tujuannya mengukur kebutuhan
kalibrasi pada rekaman lain, dan memisahkan perubahan kanal buatan dari
perubahan hari, keyboard, atau latar asli yang juga dapat terjadi.

Gunakan MFCC 78 dimensi dari pilot SKAID: crop 80 ms, 16 kHz, mean/SD
13 MFCC, delta dan delta-delta. Ridge sklearn float64, row L2, alpha
0,001, solver cholesky, intercept aktif, tanpa class weighting.
Resep dipilih pada data SKAID sebelumnya (85 pilihan MFCC identik),
tidak disetel dari data HEAR atau NC1. Semua tahap memakai CPU.

## HEAR v1 on-device

Klasifikasikan 12 huruf preset SKAID. Pakai semua 18 pasangan berarah
yang mencapai tier `verified80`, K=10, query ≥5 per huruf, pada tiga
ID. Analisis utama memakai satu pasangan per ID: urutkan menurut waktu
mulai kalibrasi paling awal, lalu waktu mulai query paling awal, lalu
indeks paket. Pasangan lain adalah sensitivitas; rata-ratakan dalam ID
sebelum menghitung mean antar ID. Laporkan juga sensitivitas ≥12 jam
untuk pasangan yang memenuhi, tanpa menambah jumlah ID. Metadata
memisahkan sesi, tetapi tidak menguji pergantian mikrofon fisik.

Untuk tiap rekaman kalibrasi, tiga draw berlabel 101/102/103 memakai
prefix nested K=1,2,3,5,10 per kelas. Untuk setiap rekaman query ambil
lima per kelas sekali; gunakan tetap di semua K/draw/pasangan terkait.
Uji K=0 dengan kepala terpisah yang dilatih pada seluruh 5.117 klip
Phase 1 yang tersedia pada cache SKAID, tanpa pakai label HEAR. Ini
transfer sumber langsung, bukan kepala K>0 yang diinterpolasi. Semua
K>0 melatih kepala dari label target saja. 6.137 label SKAID lama
merupakan union pengembangan resep, termasuk 1.020 query SKAID; 5.117
label fit K0 sudah termasuk di dalamnya.

## NC1, tiga kondisi asli dan kontrol kanal

Klasifikasikan seluruh 26 huruf menggunakan kalibrasi Fnatic tenang
10 Juli 2025. Uji pada semua WAV yang lolos audit tanggal 19 Juli:
Fnatic tenang (487; tidak ada label `e`), Fnatic bising (815), dan
Anne Pro tenang (795). Output classifier selalu 26 huruf. Laporkan
akurasi top-1 dan balanced accuracy (macro recall kelas yang hadir);
laporkan perbandingan macro recall pada 25 huruf yang sama tanpa `e`
secara terpisah. Tidak imputasi kelas hilang atau menggabungkan tanggal.
Ketiadaan `e` di Fnatic tenang setelah penghapusan duplikat membatasi
klaim. Noise/Anne tidak punya sesi kalibrasi target masing-masing;
hasil asli keduanya adalah transfer dari Fnatic tenang, tanpa label
baru dari kondisi uji. K0 sumber 12 kelas tidak kompatibel dengan
NC1 26 kelas. Satu peserta dan tanggal pada nama file bukan estimasi
populasi atau bukti bahwa perbedaan disebabkan satu faktor.

Untuk kontrol kanal, gunakan hanya Fnatic tenang pada kedua tanggal,
dengan empat kombinasi kalibrasi/query asli atau µ-law 8 kHz. Untuk
versi µ-law, resample crop 16→8 kHz lewat resample_poly Kaiser 5,
kodekan/dekode `pcm_mulaw` FFmpeg, lalu 8→16 kHz dengan filter sama.
Resampling per crop, tanpa mencampur antar klip. Label dan ID support
dan query identik dengan kombinasi asli. Pemrosesan offline menguji
shift resampling+codec secara berpasangan, tanpa paket hilang, pemrosesan
perangkat, atau pergantian mikrofon. Ini bukan simulasi VoIP penuh.
Pada setiap K, ukur selisih uji µ-law dengan kalibrasi µ-law vs
kalibrasi asli, dan penalti µ-law untuk kalibrasi asli. Kondisi bising
dan Anne hanya diuji dalam audio asli.

## Lock, kontrol, dan pelaporan

Urutkan klip per indeks audit. Ranking per kelas memakai NumPy RNG
SeedSequence entropy [20261115, role, group_index, draw]; role 1 untuk
support, role 2 untuk query HEAR (draw 0). group_index adalah posisi
kelompok pada urutan nama string. Gunakan ranking yang sama bagi kodek
dan semua query yang berbagi kelompok kalibrasi. Kontrol K10 memakai
permutasi sample-level seed [20261115, 3, group_index, draw], tetap
seimbang. Catat identitas semua klip, peran, transformasi, dan hash PCM
sebelum fit/prediksi. Larang duplikat lintas rekaman atau klip yang
berperan sebagai kalibrasi dan query dalam pasangan yang sama.

Label fit target: 12K (HEAR) atau 26K (NC1) per model; versi codec
tidak menambah event/label unik. Dukungan dipilih dari pool berlabel;
laporkan ukuran pool, union label K10 tiga draw, dan label query
terpisah. Log/export HEAR, WAV berlabel NC1, dan QA bukan biaya nol.
Tidak menganggap 180/270 keystroke berulang sebagai peserta independen.
Laporkan per-ID/pasangan (HEAR) dan per kondisi/draw (NC1), top-1,
macro recall, kontrol, dan threshold 80/90 sebagai deskripsi grid.
Tidak membuat CI populasi baru dari tiga ID atau satu peserta NC1.

Hash manifest audit, paket, seluruh masukan dan keluaran. Re-dekode
seluruh klip terpilih dan replay normalisasi, skor Ridge dengan
solusi regresi langsung, metrik, nesting, kontrol, dan pemisahan
rekaman. Laporan bersyarat pada klasifikasi potongan tertentu, tanpa
onset fisik baru, SOTA atau klaim deteksi teks kontinu.

Sumber: [rilis HEAR](https://github.com/vikrant-vikram/Hear-Dataset/releases/tag/v1),
[rilis NC1](https://doi.org/10.5281/zenodo.17536616),
[G.711 ITU-T](https://www.itu.int/rec/T-REC-G.711/en).
