# Kelayakan validasi eksternal berikutnya

Tanggal pemeriksaan: 8 November 2026. Kurva CNN Zoom→ponsel tetap memakai
Harrison dan mengikuti protokol tersendiri. Catatan ini memeriksa apakah
dataset lain dapat memberi pengujian pada rekaman baru; belum melatih model
atau mengukur akurasi pada dataset-dataset berikut.

Pembaruan 9 November: [audit audio dan sinkronisasi SKAID](SKAID_OBSERVATIONS.md)
sudah dijalankan. Seluruh 53 audio didekode; 44/50 pasangan dengan jam
log yang dapat diperiksa memenuhi kriteria kecocokan waktu sementara.
Ada 20 ID dengan kedua fase lolos; hitungan 26 huruf kini hanya memenuhi
K1 pada tiga ID dan K2 pada dua ID. Subset 12 huruf eksploratif memiliki
hitungan cukup hingga K10, sebelum penyaringan crop/overlap. Catatan
di bawah adalah hasil katalog awal sebelum audio diunduh, bukan
status audit audio terkini. Belum ada akurasi klasifikasi SKAID.

Kelanjutan tanggal sama: [koreksi offset+drift](SKAID_DRIFT_OBSERVATIONS.md)
menghasilkan 48/50 pasangan lolos dan 23 ID dengan dua fase lolos. Setelah
filter event tetangga, crop 80 ms menyediakan 17 ID untuk 12 huruf, K hingga 10
dan ≥5 query/kelas Phase 2. Ini kelayakan pilot, belum akurasi model atau
sertifikasi onset fisik per-event. Hasil offset tunggal di atas dipertahankan
sebagai pembanding.

Pilot klasifikasi berikutnya sudah dijalankan pada 17 ID dan 12 huruf:
[kurva kalibrasi SKAID](SKAID_PILOT_OBSERVATIONS.md). MFCC mencapai 52,29%
pada K10/120 label dukungan, ResNet pralatih 43,12%, BEATs 26,31%.
Belum mencapai rerata 80%/90%. Budget ini menghitung label kepala,
bersyarat pada segmentasi/filter berbasis log dan query yang distratifikasi;
belum kebutuhan label end-to-end pada mikrofon baru.

Kelanjutan [seleksi resep antar-ID](SKAID_SELECTION_OBSERVATIONS.md)
mencapai 75,78% MFCC, 69,96% ResNet pralatih dan 69,80% BEATs pada K10,
dengan label sumber tambahan untuk memilih normalisasi/alpha dari 16 ID
selain ID yang dievaluasi. Dataset dan split tetap. Pada K10, total
sumber 5.749–5.807 label unik berada di luar 120 label classifier target;
mean 80%/90% masih belum tercapai. Ini validasi silang pada kohort yang
sudah diperiksa, bukan konfirmasi pada mikrofon/sesi baru.

Kelanjutan 11 November: [SVM linear dan RBF](SKAID_SVM_OBSERVATIONS.md)
mempertahankan cohort dan budget tersebut. MFCC+RBF mencapai 76,73%
pada K10; kenaikan +0,95 pp terhadap Ridge memiliki CI berpasangan
−1,37 hingga +3,27 pp. Belum mencapai mean 80%, dan fitur neural
menurun dengan SVM. Banyak pilihan RBF berada di batas C tertinggi;
hasil ini belum menetapkan optimum kernel atau validasi mikrofon baru.

Pada [perluasan C](SKAID_SVM_C_OBSERVATIONS.md), MFCC+linear mencapai
76,63% dan MFCC+RBF 76,83% pada K10, masih pada cohort dan biaya sama.
Selisih terhadap Ridge 75,78% kecil dengan CI berpasangan mencakup nol.
Ketidakpastian utama untuk [tujuan riset](RESEARCH_GOAL.md) tetap
kinerja dan kebutuhan label pada rekaman independen.

Pembaruan 15 November: [HEAR dan NC1 sudah diaudit dan diuji](PUBLIC_RECORDING_OBSERVATIONS.md).
Rilis resmi HEAR v1 tersedia sebagai aset GitHub, dengan kanal on-device
saja. Untuk 12 huruf, tiga ID memenuhi K10 pada sesi terpisah setelah
verifikasi PCM. NC1 memisahkan hari dan kondisi; audit mengeluarkan 44
pasangan PCM silang kondisi. Kedua rilis mendukung pengujian terbatas
lintas rekaman, tetapi belum membandingkan mikrofon fisik yang berbeda.
Tabel katalog awal di bawah tidak mencerminkan ketersediaan rilis HEAR ini.

Pembaruan: [audit audio MKA sudah selesai](MKA_OBSERVATIONS.md).
Arsip, seluruh berkas, asal potongan dan metadata telah diperiksa;
manifest 36 kelas tersedia. Hasil audit belum mendukung kalibrasi/uji
lintas rekaman pada platform target yang sama.

## Ketersediaan sumber primer

| Dataset | Bukti yang diperiksa | Implikasi bagi eksperimen berikutnya |
|---|---|---|
| MKA v3 | Enam platform; audio mentah, segmen, dan fitur. Deskripsi menyebut enam segmen satu detik per kelas/platform. CC BY 4.0. | Audit audio mentah dan asal sesi diperlukan sebelum menentukan K dan split; enam segmen bukan bukti tersedianya 10 label kalibrasi ditambah uji independen. |
| SKAID v1 | Arsip audio dan log tersedia; kegiatan mengetik terstruktur dan bebas. Metadata API menyatakan CC BY 4.0. | Layak diaudit untuk ketikan beruntun, tetapi kosakata, sinkronisasi, dan pembagian rekaman perlu protokol tersendiri. |
| HEAR | Repositori resmi yang diperiksa hanya memuat README; belum ditemukan audio atau tautan unduh korpus di sana. | Belum bisa dijadikan eksperimen yang dapat dijalankan dari sumber yang ditemukan. |

Sumber: [MKA v3, Mendeley Data](https://data.mendeley.com/datasets/bpt2hvf8n3/3),
[SKAID, Zenodo](https://zenodo.org/records/17282184),
[metadata SKAID](https://zenodo.org/api/records/17282184), dan
[repositori HEAR](https://github.com/vikrant-vikram/Hear-Dataset).
Kreator MKA: Karwan Mahdi Rawf dan Ayub Abdulrahman; kreator SKAID:
Benjamin Quattrone dan Youakim Badr. Penghitungan log merupakan analisis
lokal atas data SKAID, dengan penggabungan huruf besar/kecil yang dijelaskan
di bawah.
Tidak menganggap pernyataan keragaman dataset sebagai bukti bahwa kelompok
train dan test tertentu sudah independen.

## Pemeriksaan nyata terhadap log SKAID

README, metadata rekaman, dan arsip log 1.063.303 byte diunduh ke
`data/external/skaid-catalog/`; audio belum diunduh. URL, ukuran, SHA256, dan
checksum penyedia disimpan pada `fetch.json`. Analisis event tidak membaca
CSV `full_text`. Skrip yang dapat dijalankan ulang:

```bash
uv run python research/catalog_skaid.py --output reports/skaid-feasibility
```

[Katalog lengkap](../reports/skaid-feasibility/catalogue.json) dan
[ringkasan hitungan](../reports/skaid-feasibility/kelayakan.md) menemukan
27 berkas log event, 102.842 press dan 99.668 release. Setiap berkas memuat
satu session ID; belum diverifikasi bahwa berkas-berkas tersebut mewakili
partisipan atau keyboard independen. Kolom aktual adalah `Session ID`,
`Timestamp (ms)`, `Event`, `Key`, dan `Phase`.
Agregat dan hash juga disimpan pada
[snapshot riset](results/skaid-label-feasibility-2026-11-08.json).

Untuk memeriksa skenario kalibrasi Phase 1→uji Phase 2, huruf besar/kecil
digabung sebagai tombol fisik. Syarat per log: semua kelas memiliki minimal
K press pada Phase 1 dan minimal satu press pada Phase 2.

| K | Log memenuhi seluruh 26 huruf | Log memenuhi 26 huruf +10 digit |
|---:|---:|---:|
| 1 | 7 | 0 |
| 2 | 6 | 0 |
| 3 | 1 | 0 |
| 5 | 0 | 0 |
| 10 | 0 | 0 |

Pada Phase 1, jumlah huruf paling jarang hanya 1–3 press per log. Karena
itu protokol 36 kelas K=1/2/3/5/10 Harrison tidak dapat disalin langsung ke
pembagian dua fase ini. Pengulangan atau augmentasi tidak menambah label
unik. Mengurangi kosakata atau menggabungkan log akan mengubah pertanyaan
riset dan harus ditetapkan sebelum mengevaluasi model.

Ada pula satu log dengan 7.452 event tetapi hanya satu nilai timestamp
unik, seluruhnya dalam notasi ilmiah. Fakta ini berasal dari arsip yang
diunduh; belum diketahui penyebabnya. Konversi angka tidak dapat memulihkan
waktu per-event yang tidak tercatat. Log tersebut memerlukan sumber waktu
lain atau harus dikeluarkan dari evaluasi yang membutuhkan alignment
milidetik. Perbedaan jumlah press dan release juga belum dijelaskan.

SHA256 arsip log:
`5e84c19209df891b88e8de600f171511e9af370be2de903d85353bd0b2c19eda`.

## Keputusan untuk tahap berikutnya

Kurva label Harrison beserta audit checkpoint sudah selesai. Audit MKA
menemukan satu WAV mentah per tombol/platform, tanpa metadata sesi yang
mendukung pembagian kalibrasi/uji lintas rekaman. Folder agregat merupakan
salinan dan tidak memberi kelompok baru. MKA dapat dipertimbangkan untuk
transfer langsung tanpa kalibrasi atau evaluasi lintas folder platform,
dengan protokol terpisah; belum menjadi konfirmasi efisiensi kalibrasi
lintas sesi. Lihat [hasil lengkap](MKA_OBSERVATIONS.md).

SKAID tetap kandidat evaluasi ketikan beruntun setelah pasangan audio-log,
offset/drift, jendela yang bertumpang tindih, dan kosakata yang memiliki
dukungan cukup diverifikasi. Kelayakan hitungan di atas belum membuktikan
bahwa semua press menghasilkan contoh audio yang valid. Pertanyaan
kalibrasi lintas sesi membutuhkan dataset dengan rekaman per kondisi
yang benar-benar terpisah; tidak menganggap salah satu katalog ini
sudah menyediakan pembagian tersebut.

Model dan resep latihan harus dikunci sebelum membuka hasil uji dataset
baru. Tetapkan pemisahan per rekaman/sesi/keyboard sesuai metadata yang
benar-benar tersedia; laporkan deteksi event terpisah dari klasifikasi
potongan. Tidak menyatakan jumlah label minimum, generalisasi baru, atau
status SOTA dari pemeriksaan katalog ini.
