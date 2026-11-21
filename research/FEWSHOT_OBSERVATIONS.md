# Hasil pilot kalibrasi kanal

Eksperimen utama selesai pada dataset eksternal Harrison, seluruh komputasi
neural memakai RX 6600 LE. Ada dua pretraining CoAtNet sumber, 60 pelatihan
target, 32 kepala linear BEATs, serta dua evaluasi CoAtNet tanpa kalibrasi:
94 kondisi evaluasi total. Kode, indeks dukungan/uji, checkpoint, normalisasi,
history, skor, waktu dan memori tersimpan di `runs/fewshot-pilot/`.

## Hasil utama yang terukur

Audio dipotong ulang dari WAV asli menjadi mono 16 kHz/400 ms. Seluruh 1.800
potongan unik dan tidak berimpit. Model sumber frontend baru menghasilkan
93,89% uji ponsel dan 92,78% Zoom; 600 epoch per kanal. Hasil reproduksi lama
1.100 epoch dan frontend native tetap disimpan terpisah.

CoAtNet tanpa kalibrasi pada 360 event target tetap masih 5,56% untuk
ponsel→Zoom dan 3,61% untuk Zoom→ponsel. Penyeragaman frontend belum cukup
mengatasi perpindahan domain rekaman pada konfigurasi ini. Ini bukan ablation
tunggal terhadap reproduksi lama karena anggaran pelatihan dan split uji juga
berubah, dan belum mengisolasi penyebab mikrofon, codec, sesi, atau penekanan.

BEATs + kepala linear menghasilkan kurva berikut, rerata ± SD sampel atas
tiga pilihan dukungan. K adalah contoh berlabel per tombol; 36K label per fit.
CoAtNet scratch/adaptasi memakai 1000 update; BEATs dibekukan dan kepala Ridge
ditetapkan alpha=1, tanpa validasi target atau tuning dari data uji.

| K | Label target per fit | BEATs, target Zoom | BEATs, target ponsel |
|---:|---:|---:|---:|
| 1 | 36 | 21,39% ± 2,27 pp | 25,28% ± 0,83 pp |
| 2 | 72 | 26,76% ± 1,58 pp | 31,76% ± 2,80 pp |
| 3 | 108 | 30,56% ± 1,21 pp | 36,76% ± 3,06 pp |
| 5 | 180 | 37,31% ± 1,53 pp | 45,09% ± 3,22 pp |
| 10 | 360 | 47,22% ± 0,48 pp | 52,96% ± 2,05 pp |

Pada K=10, CoAtNet adaptasi menghasilkan 32,87% ± 0,98 pp pada Zoom dan
33,15% ± 3,07 pp pada ponsel. Scratch 1000 update hanya 7,69% ± 4,27 pp dan
5,28% ± 0,28 pp. **Ini perbandingan pada anggaran update tetap, bukan peringkat
arsitektur atau bukti efisiensi label dibanding scratch yang sudah terlatih.**
Akurasi dukungan scratch K=10 hanya 8,70% dan 6,76% rata-rata, sehingga batas
optimisasi adalah masalah nyata. Model adaptasi juga hanya menguasai 57,50%
dan 50,19% dukungannya pada K=10. Tidak ada metode utama yang mencapai
rata-rata top-1 80% atau 90% sampai K=10 pada konfigurasi yang ditetapkan.

## Batas penafsiran

Protokol utama dibekukan sebelum evaluasi target. Masing-masing target memakai
15 event pertama per kelas sebagai kandidat kalibrasi dan 10 event terakhir
sebagai uji. Dukungan nested memakai seed 101/102/103; seed pelatihan tetap 42.
Tiga draw memakai 360 event uji yang sama, bukan 1080 contoh independen.
Anggaran 36K menghitung label adaptasi per fit; pilot keseluruhan tetap
memerlukan label kumpulan kalibrasi dan label uji, serta data sumber yang
berbeda untuk pretraining. BEATs membawa pretraining AudioSet AS2M.

Kedua kanal dan seluruh rekamannya telah dievaluasi pada eksperimen sebelumnya.
Ini hasil eksploratif, bukan holdout konfirmatori baru. Pemisahan dalam WAV
tidak menggantikan pemisahan sesi; identitas file dan kelas masih berimpit.
Ketidakpastian isolasi phone/J tetap berlaku. Perbandingan keluarga model
tidak menetapkan SOTA untuk pengenalan tombol.

## Verifikasi dan artefak

- [Laporan utama dan kurva](../reports/fewshot-pilot/laporan.md).
- [Protokol sebelum evaluasi](FEWSHOT_PROTOCOL.md).
- [Audit prediksi CPU dan WAV kesalahan](../reports/fewshot-audit/README.md).
- [Protokol diagnostik optimisasi](FEWSHOT_DIAGNOSTIC_PROTOCOL.md), ditentukan
  setelah dua hasil scratch K=1 pertama dan sebelum evaluasi K=10 selesai.

Seluruh 94 prediktor diulang di CPU dari bobot/kepala yang tersimpan; confusion
matrix, top-1/top-5 dan macro-F1 cocok persis. Anggaran label, dukungan dan
normalisasi juga cocok. BEATs diulang pada CPU dari 16 potongan waveform asli
cache; selisih fitur maksimum sekitar 1,31e-5 terhadap GPU. Fbank dibandingkan
dengan TorchAudio v0.13.0 pada delapan klip nyata dan identik.

SHA256 hasil utama:
`0b033014568b62ce8b9ed8f1773ed7d33af50c23740d2d99e1763e671416cba1`.

## Diagnostik optimisasi selesai

Dua fit tambahan memakai K=10 dan seed dukungan 101 yang sama dengan pilot,
masing-masing dilatih ulang dari inisialisasi seed 42 hingga 10.000 update.
Schedule linear juga diperpanjang ke 10.000 update. Tidak memilih checkpoint
dari uji. Angka berikut satu draw, bukan rerata tiga pengulangan.

| Target | Scratch 1.000 update | Scratch 10.000 update | Dukungan setelah 10.000 | Adaptasi 1.000 update | BEATs + linear |
|---|---:|---:|---:|---:|---:|
| Zoom | 2,78% | **54,72%** | 95,83% | 31,94% | 47,50% |
| Ponsel | 5,56% | **2,78%** | 2,78% | 36,39% | 55,28% |

**Pada Zoom, urutan metode berubah ketika anggaran optimisasi scratch
diperpanjang.** Hasil utama tidak membuktikan bahwa pretrained membutuhkan
lebih sedikit label daripada scratch yang dilatih memadai. Pada ponsel,
pelatihan panjang tidak memperbaiki kegagalan: model memprediksi kelas `8`
untuk seluruh 360 contoh uji. Pemeriksaan CPU pada 360 contoh dukungan juga
menemukan logit identik antarcontoh (rentang maksimum 0), sedangkan aktivasi
ReLU pertama tidak seluruhnya nol. Ini observasi keluaran yang kolaps, bukan
diagnosis pasti penyebab atau bukti seluruh model CoAtNet selalu gagal.
Hasil dan hash pemeriksaan tersimpan di
`research/results/fewshot-diagnostic-stemcheck.json`.

[Laporan diagnostik lengkap](../reports/fewshot-diagnostic/laporan.md) menyimpan
kurva dukungan, waktu, protokol tambahan dan audit CPU dua checkpoint; seluruh
confusion matrix dan metrik diagnostik juga cocok persis dengan GPU. Hasil
utama tetap terpisah dan tidak ditimpa. Diagnostik hanya satu inisialisasi dan
dukungan per target; tidak menetapkan peringkat model yang andal.

SHA256 hasil diagnostik:
`eb52e50c09f47eba61e80f761164a0fd86ddd668105d0fa737b0186fe01b2b88`.

## Langkah berikutnya yang didukung hasil

Prioritas pertama adalah menstabilkan optimisasi sebelum memperluas kurva
efisiensi label: periksa aktivasi/gradien, ulangi inisialisasi, tentukan
hyperparameter hanya dari sumber/dev, dan uji adaptasi kepala klasifikasi
dengan backbone CoAtNet dibekukan. Bandingkan scratch dan adaptasi dengan
anggaran yang cukup serta laporkan kegagalan, bukan hanya run terbaik.
Kegagalan satu inisialisasi tidak boleh diperbaiki dengan memilih seed dari
hasil uji target. Setelah protokol layak dibekukan, konfirmasi pada dataset
eksternal yang menyediakan rekaman/sesi independen. Dataset pribadi tetap
menjadi tahap berikutnya, sesuai rencana pengguna.

Kesimpulan pertanyaan awal: pilot menunjukkan kenaikan skor BEATs saat label
ditambah, tetapi **belum menetapkan jumlah minimal label untuk akurasi tinggi
atau keunggulan adaptasi atas scratch**. Batas komputasi dan kestabilan
optimisasi menjadi bagian penting yang harus ditangani terlebih dahulu.

Kelanjutan 8 November selesai dalam
[uji kepala beku dan kestabilan](FROZEN_OBSERVATIONS.md): 94 kondisi kepala
linear baru, autopsi aktivasi/gradien, dan kontrol tiga inisialisasi untuk
perubahan satu gate. Hambatan ReLU pada run ponsel teridentifikasi dan seed
yang gagal dapat belajar kembali; hasil antarseed tetap bervariasi. Temuan
ini dipisahkan dari pilot dan diagnostik awal di atas.
