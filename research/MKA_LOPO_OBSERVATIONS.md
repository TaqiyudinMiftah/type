# BEATs dan ResNet lintas folder MKA: 8 November 2026

**BEATs beku belum memperbaiki generalisasi lintas folder pada protokol
yang diuji.** Dengan lima folder latih dan satu folder uji, akurasi
gabungan BEATs hanya **2,16%** (28/1.296 potongan), sedangkan ResNet
**2,65% ± 0,66 pp** antar tiga inisialisasi sumber. Peluang uniform
36 kelas adalah 2,78%. Ini deskripsi hasil, bukan uji kesetaraan statistik.

Kelanjutan terpisah: [pemilihan regularisasi melalui validasi bersarang](MKA_NESTED_OBSERVATIONS.md)
sudah selesai. Tuning hanya memakai folder latih dan tetap belum
memperbaiki transfer. Hasil tersebut tidak mengganti baseline alpha1 di sini.

Studi ini memakai protokol baru setelah kegagalan transfer langsung
Harrison→MKA dibaca. Kepala dilatih dengan label MKA dari folder lain;
hasilnya berbeda dari eksperimen sebelumnya yang tidak melakukan fit
pada MKA sama sekali. Semua encoder dan statistik sumber tetap beku.

## Pemisahan dan perbandingan

Enam fold menahan seluruh folder hp/Lenovo/MSI/Mac/Messenger/Zoom secara
bergantian. Latih K6 memakai 1.080 potongan dari 180 WAV induk; uji
216 potongan dari 36 WAV induk. Tidak ada induk atau fingerprint PCM
potongan yang melintasi fit/uji satu fold. Setiap konfigurasi menghasilkan
prediksi out-of-fold untuk seluruh 1.296 potongan, masing-masing satu kali.

BEATs_iter3_plus_AS2M memakai 768 fitur mean-pooled; ResNet Harrison
seed42/43/44 memakai 128 fitur sebelum classifier. Kedua metode menerima
crop peak400ms mono16k yang sama dan kepala Ridge alpha1, normalisasi
L2 per vektor, serta label/indeks fit yang identik. Tidak ada fine-tuning,
validasi target, tuning alpha, pemilihan seed terbaik, atau ensemble.

| Folder uji | BEATs | ResNet, mean ± SD inisialisasi |
|---|---:|---:|
| hp | 2,31% | 3,09% ± 1,17 pp |
| Lenovo | 1,39% | 3,09% ± 0,96 pp |
| MSI | 3,70% | 4,01% ± 2,67 pp |
| Mac | 1,39% | 3,40% ± 0,27 pp |
| Messenger | 1,85% | 1,08% ± 0,53 pp |
| Zoom | 2,31% | 1,23% ± 1,07 pp |

BEATs mempunyai satu model tetap. ResNet gabungan per seed mencapai
3,40%, 2,39%, dan 2,16%. SD di atas bukan ketidakpastian populasi sesi.
Folder berbagi data latih antar fold; nama folder belum merupakan
identitas fisik perangkat/partisipan/sesi yang terverifikasi.

## Jumlah contoh tidak menyelesaikan transfer

K adalah contoh per tombol PADA SETIAP lima folder latih, bukan jumlah
total per tombol atau label dari folder uji. Dukungan K1/2/3/5 nested
pada tiga draw101/102/103. K6 memakai semua enam potongan, hanya satu
set dukungan unik. Seluruh K tetap mencakup lima WAV induk/tombol.

| K/folder latih | Label/fit | Rerata BEATs | Rerata ResNet |
|---:|---:|---:|---:|
| 1 | 180 | 2,85% | 2,63% |
| 2 | 360 | 2,85% | 2,99% |
| 3 | 540 | 2,80% | 2,87% |
| 5 | 900 | 2,34% | 2,67% |
| 6 | 1.080 | 2,16% | 2,65% |

BEATs K<6 dirata-ratakan atas tiga draw; ResNet atas desain tiga
inisialisasi × tiga draw. K6 tidak diberi tiga ulangan dukungan semu.
Laporan memisahkan variasi inisialisasi dan pilihan dukungan serta
selisih berpasangan pada split yang sama. Tidak terlihat kenaikan
akurasi uji yang konsisten pada grid ini.

BEATs menyesuaikan data latih jauh lebih baik: rerata akurasi dukungan
92,41% pada K1 dan 70,32% pada K6. ResNet masing-masing 34,67% dan
29,67%. Informasi yang memisahkan contoh latih belum berpindah ke folder
baru; sinyal tersebut dapat terkait identitas rekaman. Ini tidak
membuktikan bahwa fitur tombol universal telah dipelajari. Alpha1 tetap
dan backbone beku juga belum menguji seluruh kapasitas model/optimasi.

Kontrol satu pengacakan label tingkat WAV induk pada K6 menghasilkan
BEATs 3,09% dan ResNet 3,50% ± 0,91 pp. Kontrol mempertahankan jumlah
contoh seimbang dan label konsisten di dalam setiap WAV; bukan
distribusi null atau uji signifikansi. Tidak ada bukti peningkatan
transfer yang berguna dari perbandingan ini.

## Reproduksi dan audit

Seluruh **336 kepala linear** selesai: 312 fit normal dan 24 kontrol.
Audit terpisah membangun ulang crop dari ZIP dan embedding di CPU,
memeriksa split/label/checksum, mengulang fit serta menyelesaikan ridge
terpusat secara langsung. Semua koefisien/skor memenuhi toleransi,
prediksi/metrik cocok, dan seluruh parameter/buffer encoder tetap sama.

Selisih embedding BEATs GPU/CPU maksimum 0,00001031, ResNet identik.
Tidak ada perubahan prediksi pada seluruh kondisi. Tersedia 84.672
prediksi potongan/WAV untuk audit; jumlah baris ini bukan jumlah contoh
independen tambahan.

BEATs diekstrak pada **AMD Radeon RX 6600 LE**, sekitar 3,12 detik
termasuk kontrol sumber, di luar waktu load/hash/probe. ResNet diekstrak
di CPU; semua kepala sekitar 6,60 detik dan audit CPU 25,68 detik.
Radeon 780M tidak digunakan; tidak ada pelatihan ulang backbone.

Pemeriksaan proyek: **74 tes lolos**, Ruff lint/format lolos, wheel dan
sdist berhasil dibangun serta memuat kedua modul studi baru. Lima tes
baru memeriksa split induk/nesting, penolakan kebocoran, kontrol label
tingkat WAV, kepala 36 kelas yang tidak fit pada uji, dan agregasi induk.
Semua checksum artefak serta 84.672 baris prediksi diperiksa kembali;
verifikasi run selesai tidak mengubah artefak atau melatih kepala baru.

## Arti dan langkah berikutnya

Representasi BEATs lebih mampu membedakan data latih di konfigurasi ini,
tetapi tidak memberi transfer lintas folder yang berguna. Mengganti
encoder dengan model lebih besar saja belum menyelesaikan masalah.
Ini perbandingan dua pipeline terpilih, bukan peringkat SOTA: dimensi,
pretraining, parameter dan biaya sumber berbeda.

Diagnostik berikut yang layak ialah pemilihan kepala/regularisasi hanya
dari folder latih melalui validasi bersarang, tanpa memilih dari folder
uji. Itu dapat memeriksa ketergantungan hasil pada Ridge alpha1.
Pertanyaan apakah sedikit label target dapat membantu tetap perlu
dibedakan: MKA tidak menyediakan WAV kalibrasi/uji berbeda pada folder
yang sama, sehingga efisiensi adaptasi ke sesi baru belum dapat diukur.
Tidak menjalankan tuning baru setelah membaca skor uji ini.

Seluruh folder telah diperiksa sebelumnya dan semua label ikut fit di
fold lain. Pemisahan WAV/PCM bukan bukti independensi semua transformasi
atau sesi fisik. Perubahan keyboard, kanal, segmentasi dan cara mengetik
belum dapat dipisahkan secara kausal. Tidak ada klaim decoding teks.

Artefak: [protokol](MKA_LOPO_PROTOCOL.md),
[laporan lengkap dan grafik](../reports/mka-lopo/laporan.md),
[336 kondisi](../reports/mka-lopo/per-condition.csv),
[audit CPU](../reports/mka-lopo-audit/audit.json),
[prediksi individual](../reports/mka-lopo-audit/classifications.csv),
[lock dan split](../runs/mka-lopo/lock.json),
[hasil mentah](../runs/mka-lopo/results.json),
[snapshot JSON](results/mka-lopo-2026-11-08.json).
