# Regularisasi bersarang MKA: hasil 8 November 2026

**Pemilihan regularisasi dari folder latih belum memperbaiki transfer.**
BEATs berubah dari 2,16% menjadi **1,77%**; rerata ResNet tetap **2,65%**,
dengan SD antar inisialisasi berubah dari 0,66 menjadi 0,39 pp.
Peluang uniform 36 kelas adalah 2,78%. Ini deskripsi hasil, bukan uji
kesetaraan statistik atau bukti seluruh pendekatan akustik mustahil.

## Apa yang dikendalikan

Enam outer fold mempertahankan split K6 LOPO: 1.080 potongan dari lima
folder latih, 216 potongan dari folder uji. Lima inner fold memakai
864 potongan/144 WAV untuk fit dan 216 potongan/36 WAV untuk validasi.
Induk WAV dan PCM potongan terpisah pada ketiga peran. Hanya label
outer-train yang diberikan kepada fungsi seleksi.

Sembilan alpha ditentukan sebelum percobaan: 1e-6 sampai 100 dengan
langkah faktor10. Seleksi memaksimalkan jumlah benar dari lima inner
validation; seri memilih alpha terbesar. Semua **48 pilihan dikunci
sebelum prediksi outer baru pertama**. Baseline alpha1 dipakai ulang
dari studi sebelumnya pada indeks/label yang sama. Tidak menghitung
outer score kandidat lain atau mengubah grid berdasarkan hasil uji.

Satu BEATs dan tiga ResNet sumber memakai embedding/crop yang identik
dengan studi LOPO. Tidak mengekstrak ulang encoder atau melakukan
fine-tuning. Seluruh tahap memakai CPU; Radeon 780M tidak digunakan.

## Seluruh folder

| Folder uji | BEATs alpha1 | BEATs nested | ResNet alpha1 | ResNet nested |
|---|---:|---:|---:|---:|
| hp | 2,31% | 0,00% | 3,09% | 2,47% |
| Lenovo | 1,39% | 2,31% | 3,09% | 3,40% |
| MSI | 3,70% | 2,78% | 4,01% | 3,24% |
| Mac | 1,39% | 2,78% | 3,40% | 1,70% |
| Messenger | 1,85% | 1,39% | 1,08% | 2,31% |
| Zoom | 2,31% | 1,39% | 1,23% | 2,78% |

ResNet tabel adalah rerata tiga inisialisasi, bukan ensemble.
Gabungan out-of-fold memuat 1.296 potongan unik satu kali per model.
ResNet nested seed42/43/44 mencapai 3,01%, 2,24%, dan 2,70%.
Fold berbagi data latih; SD tidak mengukur variasi sesi/keyboard.

Nilai alpha terpilih bervariasi antar folder/encoder. Dari 24 tugas
normal, empat memilih batas atas100 dan tidak ada yang memilih batas
bawah1e-6. Grid tidak diperluas setelah hasil dibaca. Kurva seluruh
kandidat dan pilihan per seed tersimpan, tanpa pemilihan seed terbaik.

## Fit meningkat, transfer belum

Rerata akurasi dukungan BEATs meningkat dari 70,32% menjadi **78,02%**;
ResNet dari 29,67% menjadi **55,55%**. Pada empat folder luar, BEATs
menguasai dukungan 100% tetapi akurasi uji hanya 0,00–2,78%. Informasi
yang membedakan contoh latih belum berpindah ke folder baru di setup ini.

BEATs memperbaiki 18 contoh dan membuat 23 kesalahan baru: benar total
28→23/1.296. Tiga ResNet memperbaiki 78 prediksi dan membuat 78 kesalahan
baru, sehingga reratanya tetap meskipun prediksi berubah. Hitungan
ResNet mengulang contoh yang sama pada tiga model, bukan 3.888 audio unik.

Rerata inner validation pada alpha pemenang adalah 3,02% BEATs dan
3,40% ResNet. Angka tersebut sudah digunakan untuk memilih alpha dan
optimis, sehingga bukan perkiraan akurasi final yang bebas bias.

Kontrol label acak tingkat WAV seed997 mengikuti prosedur seleksi yang
sama, memakai permutasi outer-train yang konsisten pada inner fit dan
validation. Hasil nested BEATs 2,70% dan ResNet 3,47% ± 0,66 pp.
Ini satu kontrol deskriptif, bukan distribusi null atau uji signifikansi.

## Reproduksi dan verifikasi

Terdapat **2.160 fit inner dan 48 refit akhir**, dengan 48 baseline
alpha1 lama sebagai pasangan. Auditor mengulang semuanya, menyelesaikan
ridge terpusat secara langsung, memeriksa indeks/label/checksum,
urutan lock, prediksi/metrik, dan aturan pemecah seri. Seluruh pilihan
alpha dan prediksi cocok; koefisien/skor inner identik pada replay.

Head terpilih juga diterapkan pada embedding CPU dari audit LOPO yang
dipatok. Tidak ada perubahan prediksi pada seluruh tugas. Tahap ini
tidak mengklaim melakukan ekstraksi CPU baru. Tersedia 2.160 baris
validasi dan 12.096 prediksi potongan/WAV akhir untuk audit.

Seleksi dan penyimpanan memerlukan 18,73 detik di luar load/hash awal;
refit/evaluasi/verifikasi akhir 2,81 detik; audit 116,09 detik.
Seluruh **80 tes lolos**, termasuk enam tes baru untuk split bersarang,
penolakan kebocoran PCM, aturan seri, isolasi seleksi dari data uji,
penolakan validasi berulang, serta efek shrinkage alpha.
Ruff lint dan format juga lolos.

## Implikasi dan batas

Pada encoder beku dan grid ini, tuning alpha belum menyelesaikan
generalisasi lintas folder. Keunggulan fit tidak membuktikan fitur tombol
yang universal; sinyal dapat terkait rekaman. Tidak menyimpulkan semua
model/segmentasi akan gagal atau kalibrasi target pasti akan berhasil.

Studi dipicu hasil sebelumnya dan semua folder sudah diperiksa serta
menjadi fit di fold lain. Nested CV memisahkan seleksi alpha setiap
predictor dari folder ujinya, tetapi tidak membuat data kembali buta.
Nama folder belum merupakan ID fisik/sesi terverifikasi. Pemisahan
file/PCM belum menolak semua salinan transformasi. Perubahan keyboard,
kanal, cara mengetik dan segmentasi belum dipisahkan secara kausal.

Langkah yang lebih informatif berikutnya adalah menguji manfaat label
domain target dengan rekaman kalibrasi/uji terpisah. MKA tetap hanya
memiliki satu WAV/tombol/folder dan belum mendukung efisiensi adaptasi
ke sesi baru. Kontrol dalam WAV dapat menjadi diagnostik tersendiri,
tetapi tidak menggantikan validasi lintas rekaman. Tidak memperluas
grid atau mengganti model melalui hasil uji studi ini.

Artefak: [protokol](MKA_NESTED_PROTOCOL.md),
[laporan dan grafik](../reports/mka-nested/laporan.md),
[pilihan alpha terkunci](../runs/mka-nested/selections-locked.json),
[48 hasil luar](../reports/mka-nested/per-condition.csv),
[2.160 validasi](../reports/mka-nested/inner-validation.csv),
[audit](../reports/mka-nested-audit/audit.json),
[prediksi berpasangan](../reports/mka-nested-audit/classifications.csv),
[snapshot](results/mka-nested-2026-11-08.json).
