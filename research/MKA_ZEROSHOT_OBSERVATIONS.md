# Harrison → MKA tanpa kalibrasi: hasil 8 November 2026

Transfer langsung belum berhasil. Compact-resnet yang mencapai 95–96,67%
pada holdout sumber Harrison Zoom hanya mencapai rerata **2,62–4,63%**
pada enam folder MKA. Peluang uniform untuk 36 tombol adalah 2,78%.
Angka yang dekat peluang ini merupakan deskripsi; belum melakukan uji
kesetaraan statistik atau mengklaim tidak ada sinyal akustik yang berguna.

Evaluasi memakai seluruh **1.296 potongan PCM unik** dari 216 WAV induk,
36 kelas, enam folder dan tiga checkpoint sumber seed42/43/44. Tidak ada
fit target, update bobot, normalisasi target, atau pemilihan seed terbaik.
Sumber menggunakan 648 train + 72 validasi label unik sebelumnya. Tahap
ini memakai CPU; tidak menggunakan Radeon 780M atau fit GPU baru.

Kelanjutan terpisah sudah membandingkan
[BEATs dan ResNet melalui kepala yang belajar dari folder MKA lain](MKA_LOPO_OBSERVATIONS.md),
dengan protokol dan anggaran label berbeda. Hasil berikutnya tersebut
tidak mengganti baseline tanpa kalibrasi yang dilaporkan di sini.

## Semua platform

Rerata ± sample SD antar tiga inisialisasi sumber. SD bukan variasi sesi
atau ketidakpastian populasi keyboard. Masing-masing platform memiliki
216 potongan dari 36 WAV, dengan enam potongan yang saling bergantung
per WAV. Nama platform mengikuti dataset, belum merupakan ID fisik.

| Folder | Utama: peak 400 ms | Sensitivitas: tiled 400 ms |
|---|---:|---:|
| hp | 3,09% ± 0,27 pp | 2,78% ± 0,46 pp |
| Lenovo | 2,62% ± 1,49 pp | 1,54% ± 1,49 pp |
| MSI | 4,63% ± 0,93 pp | 4,17% ± 0,93 pp |
| Mac | 3,55% ± 1,41 pp | 3,24% ± 0,80 pp |
| Messenger | 4,17% ± 0,80 pp | 3,09% ± 0,27 pp |
| Zoom | 2,93% ± 0,27 pp | 2,78% ± 0,00 pp |

Aturan utama memilih argmax amplitudo mono16k pertama, lalu mengambil
400 ms dengan 50 ms konteks sebelum puncak; crop di-clamp bila melewati
batas. Sensitivitas memakai semua jendela 400 ms ber-stride200 ms,
ditambah jendela akhir, lalu merata-ratakan logit. Kedua aturan dikunci
sebelum prediksi MKA dan keduanya dilaporkan. Seluruh statistik input
dan BatchNorm memakai sumber tetap.

Agregasi enam potongan ke satu prediksi WAV juga rendah: rerata peak
1,85–5,56% pada 36 WAV/platform. Tidak ada ensemble lintas seed.

## Apa yang diperiksa

1. Kontrol holdout sumber Zoom menghasilkan 171/180, 174/180 dan 173/180
   prediksi benar. Kontrol transfer Harrison phone K=0 menghasilkan
   22/360, 16/360 dan 15/360. Seluruh prediksi individual cocok dengan
   hasil lama, termasuk saat frontend CPU dihitung ulang dari waveform.
2. Audit terpisah membangun ulang semua 1.296 crop peak dan 5.184 jendela
   tiled dari ZIP, memeriksa provenance/label/offset, lalu mengulang semua
   seed dan aturan crop. Seluruh fitur/logit CPU identik dan metrik cocok.
   Digest seluruh bobot dan buffer BatchNorm tetap.
3. Model hanya memprediksi 2–14 dari 36 kelas pada satu platform/seed
   dalam aturan utama. Contoh hp seed44: 213/216 prediksi adalah '8';
   pada tiled, 216/216 adalah '8'. Ini bukti bias classifier pada target,
   bukan bukti penyebab tunggal tertentu.
4. Sebanyak 442/1.296 crop peak di-clamp ke batas potongan. MKA belum
   memiliki anotasi onset tiap ketukan; potongan dapat memuat beberapa
   transien. Sensitivitas tiled yang juga buruk belum menyingkirkan
   seluruh pengaruh perbedaan segmentasi sumber/target.

Pemeriksaan proyek: **69 tes lolos**, Ruff lint/format lolos, paket wheel
dan sdist berhasil dibangun. Delapan tes baru mencakup batas crop,
resampling/kanal, agregasi bertingkat, metrik berkelompok, dan BatchNorm
beku. Audit data/model di atas memeriksa hasil nyata di luar tes sintetis.

## Implikasi riset

Keunggulan adaptasi yang terukur dalam Harrison belum membuktikan bahwa
classifier dapat langsung dipakai pada dataset lain. Hasil ini menjadi
baseline transfer tanpa kalibrasi, bukan pengganti kurva few-shot target
dan bukan klaim model SOTA. Perubahan dataset, keyboard, cara mengetik,
kanal, serta segmentasi terjadi sekaligus; penurunan tidak dapat
diatribusikan hanya ke codec/mikrofon. Label folder Zoom di dua dataset
juga tidak membuktikan kondisi rekaman yang sama.

Untuk melanjutkan pertanyaan representasi, langkah yang layak adalah
kontrol embedding audio pralatih dengan WAV induk yang terpisah, misalnya
evaluasi lintas folder platform MKA melalui protokol baru. Hasil itu
tetap bukan uji kalibrasi ke sesi baru pada platform yang sama. Untuk
mengukur efisiensi kalibrasi tersebut, diperlukan dataset yang memiliki
rekaman kalibrasi dan uji terpisah per tombol; struktur MKA saat ini belum
memenuhinya. Tidak menjalankan intervensi baru setelah membaca hasil ini.

Audit belum mencari semua salinan Harrison yang telah dipotong atau
ditransformasi. Nama platform dan WAV induk tidak disulap menjadi ID
sesi/keyboard/partisipan. Tidak ada eksperimen decoding kalimat atau
penghitungan event dari audio bebas.

Artefak: [protokol sebelum inferensi](MKA_ZEROSHOT_PROTOCOL.md),
[laporan dengan gambar dan semua metrik](../reports/mka-zeroshot/laporan.md),
[36 kondisi per seed](../reports/mka-zeroshot/per-condition.csv),
[audit replay](../reports/mka-zeroshot-audit/audit.json),
[prediksi individual](../reports/mka-zeroshot-audit/classifications.csv),
[hasil mentah](../runs/mka-zeroshot/results.json), dan
[snapshot JSON](results/mka-zeroshot-2026-11-08.json).
