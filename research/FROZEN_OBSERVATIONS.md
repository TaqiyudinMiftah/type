# Kepala beku dan kestabilan optimisasi — 8 November 2026

Kelanjutan [pilot few-shot](FEWSHOT_OBSERVATIONS.md) memakai dataset eksternal
Harrison dan RX 6600 LE. Hasil lama tetap utuh. Tahap ini menguji kepala linear
dengan backbone beku dan menelusuri hambatan belajar pada checkpoint ponsel
yang gagal. Data target dan uji tetap sama; hasil masih eksploratif dalam WAV,
bukan konfirmasi pada sesi/pengguna/keyboard baru.

## Adaptasi kepala linear

94 kondisi baru selesai: K=1/2/3/5/10 dengan tiga dukungan nested, serta
empat kondisi K=0. Seluruh parameter dan buffer BatchNorm encoder terverifikasi
tetap. Kepala Ridge alpha=1 dan normalisasi L2 fitur sama seperti BEATs pilot;
tidak ada label validasi atau penalaan hyperparameter dari uji target.

Pada K=10, ada 360 label target per fit. Rerata tiga dukungan:

| Metode | Target Zoom | Target ponsel |
|---|---:|---:|
| CoAtNet beku, normalisasi sumber | 15,46% | 24,44% |
| CoAtNet beku, normalisasi dukungan | 26,85% | 24,35% |
| CoAtNet acak beku, normalisasi sumber | 2,78% | 2,78% |
| BEATs beku + linear, pilot sebelumnya | 47,22% | 52,96% |
| CoAtNet adaptasi 1.000 update, sebelumnya | 32,87% | 33,15% |

Normalisasi dukungan membantu target Zoom pada konfigurasi ini, tetapi
pembekuan backbone belum memperbaiki skor metode sebelumnya. Kepala yang sama
memperoleh 89,44% pada domain sumber ponsel dan 95,00% pada sumber Zoom
(180 contoh uji per sumber). Representasi sumber masih berguna dalam domain,
tetapi Ridge/L2 ini belum memindahkannya dengan baik ke kanal lain. Ini tidak
membuktikan semua informasi hilang: geometri fitur, regularisasi, normalisasi,
dan pengolahan kanal masih membatasi interpretasi. Random control hanya satu
inisialisasi, bukan model scratch yang telah dilatih.

Audit CPU seluruh 94 kepala cocok persis untuk skor/confusion matrix dan
metrik. Dukungan, anggaran label serta normalisasi diverifikasi. Ekstraksi
delapan subset cache encoder dan 30 subset support_norm juga dibandingkan
dengan GPU; seluruh bobot/buffer tetap sama. Empat pemeriksaan domain sumber
dapat direproduksi. Detail dan SD tersedia dalam
[laporan kepala beku](../reports/frozen-head/laporan.md).

## Hambatan langsung pada checkpoint ponsel

Autopsi hanya memakai 360 contoh dukungan K=10 seed 101. Pada checkpoint
scratch 10.000 update ponsel, seluruh input ReLU `0.mlp4.1` negatif, rentang
sekitar −0,15498 hingga −0,00054. Gate menghasilkan nol untuk setiap posisi
dan contoh. Gradien input serta blok sebelumnya nol pada mode eval dan
salinan train, sementara classifier menghasilkan logit identik antarcontoh.

Mengganti hanya gate ini menjadi LeakyReLU(0,01) pada salinan checkpoint,
tanpa optimizer step, memulihkan rentang logit sekitar 1,13e-5 dan norm
gradien input 7,57e-8. Ini menunjukkan hambatan gradien lokal dan memberi
hipotesis perbaikan; perubahan belum membuktikan peningkatan akurasi atau
menjelaskan mengapa pelatihan awalnya memasuki wilayah negatif. Model Zoom
yang berhasil dan model sumber ponsel tidak menunjukkan ReLU mati tersebut.

[Protokol autopsi/kepala beku](FROZEN_HEAD_PROTOCOL.md) dan JSON lengkap di
`reports/fewshot-autopsy/` mencatat seluruh tahap, norm gradien dan intervensi.
Checkpoint lama dan kode vendor tidak diubah. Tidak ada contoh uji baru yang
dipakai dalam autopsi.

## Kontrol pelatihan dari awal

[Protokol kontrol aktivasi](STABILITY_PROTOCOL.md) ditetapkan setelah autopsi:
ReLU asli vs LeakyReLU pada gate itu saja, seed inisialisasi 42/43/44, dukungan
K=10 seed 101 dan 10.000 update. Kontrol ReLU seed 42 memakai hasil lama yang
identik konfigurasinya; lima fit lain merupakan pelatihan baru. Seluruh hasil
dan [laporan kontrol](../reports/stability-gate/laporan.md) dipisahkan dari
pilot, tanpa memilih seed atau checkpoint dari akurasi uji.

Seluruh kontrol selesai dan enam checkpoint cocok saat diulang di CPU.
Dengan dukungan tetap 360 label dan 360 contoh uji ponsel:

| Seed pelatihan | ReLU asli | LeakyReLU pada satu gate | Selisih (pp) |
|---:|---:|---:|---:|
| 42 | 2,78% | 69,17% | +66,39 |
| 43 | 26,67% | 28,33% | +1,67 |
| 44 | 55,28% | 53,33% | −1,94 |
| Rerata ± SD | 28,24% ± 26,29 pp | 50,28% ± 20,59 pp | +22,04 ± 38,45 |

LeakyReLU tidak menghasilkan logit konstan pada ketiga run yang diuji,
sedangkan ReLU asli mengalaminya pada satu dari tiga run. Kenaikan rata-rata
terutama berasal dari pemulihan seed 42; seed 44 sedikit menurun. Akurasi
dukungan LeakyReLU 99,44%, 75,00%, dan 97,78%; optimisasi seed 43 masih belum
menguasai dukungan sebagaimana dua run lain. Jangan memilih 69,17% sebagai
kinerja representatif seluruh model atau menyimpulkan kestabilan populasi
dari tiga run.

SD kontrol ini adalah variasi inisialisasi pada dukungan yang sama, sedangkan
SD kepala beku adalah variasi dukungan pada encoder tetap. Keduanya tidak
langsung dapat dipakai untuk menilai metode mana lebih stabil secara umum.
Kontrol ReLU 42 dipakai ulang dari diagnostik lama, lima fit lainnya baru;
dataset, normalisasi, optimizer, schedule dan seed diperiksa sesuai protokol.

## Apa yang sudah lebih jelas

1. Ada hambatan optimisasi yang terlokalisasi pada gate terakhir di checkpoint
   ponsel yang gagal. Ini menjelaskan mengapa sekadar menambah update tidak
   memulihkan run itu. Perubahan satu aktivasi dari awal memungkinkan seed
   tersebut belajar kembali, tetapi belum memastikan semua run berakurasi tinggi.
2. Adaptasi kepala beku, pada Ridge/L2 yang diuji, tidak cukup memulihkan
   perpindahan kanal. Statistik input target membantu sebagian kondisi, tetapi
   representasi sumber dan domain rekaman tetap berpengaruh.
3. Kurva label lama pada batas 1.000 update belum menjadi dasar yang layak
   untuk klaim efisiensi label atau model SOTA. Kontrol K=10 ini juga belum
   menentukan jumlah minimal contoh karena tidak menguji ulang seluruh grid K.

Langkah berikutnya yang terukur adalah memilih konfigurasi optimisasi melalui
data sumber/dev, lalu membandingkan scratch dan adaptasi dengan anggaran yang
memadai dan beberapa inisialisasi. Setelah protokol ditetapkan, konfirmasi pada
rekaman/sesi independen dari dataset eksternal; pengumpulan dataset pribadi
tetap tahap berikutnya sesuai rencana pengguna. Jangan terus menala pada test
Harrison yang sama lalu menyebutnya generalisasi ke kondisi baru.

Verifikasi tahap ini: 39 tes lulus, audit 94 kepala plus empat pemeriksaan
sumber, serta replay enam kontrol CNN. Bobot, buffer, konfigurasi, indeks,
normalisasi, checkpoint dan hasil lama terjaga. Snapshot ringan disimpan dalam
`research/results/frozen-head-2026-11-08.json` dan
`research/results/stability-gate-2026-11-08.json`.

Kelanjutan: [perbaikan kestabilan](REPAIR_OBSERVATIONS.md) menguji konfigurasi
melalui Zoom, kemudian mengunci CNN residual kompak sebelum uji ponsel baru.
Hasil K=10 pada tiga inisialisasi mencapai 89,07% ± 0,85 pp, dengan dukungan
100% pada semua seed. Rincian kandidat gagal, 15 replay CPU dan batas
interpretasi ada pada laporan tersebut.
