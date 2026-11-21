# Diagnostik sinkronisasi SKAID v1

Ditulis setelah audit native `reports/skaid-audit/summary.json` selesai,
sebelum menghitung korelasi audio–event. Tanggal lokal: 9 November 2026.
Ini pemeriksaan kelayakan segmentasi, belum eksperimen klasifikasi atau
konfirmasi jumlah label minimum. Semua perhitungan memakai CPU.

## Sumber dan unit pemeriksaan

Gunakan 52 M4A asli, satu per ID/fase, dari arsip Zenodo yang telah dipatok.
Dua pasangan tanpa audio dan dua fase dengan timestamp runtuh tetap
tercantum sebagai tidak dapat diperiksa. Jangan memulihkan timestamp dari
urutan tombol atau memilih rekaman berdasarkan akurasi model.
Semua masukan, protokol, kode dan artefak audit sebelumnya diverifikasi
SHA256 sebelum dan sesudah perhitungan. Jangan mengubah audit native.

WAV tambahan pada ID `qt_1748184742062`, Phase 2, tidak masuk unit uji.
Dekode WAV dan M4A pasangannya pada laju native; resample M4A 48→16 kHz
dengan `scipy.signal.resample_poly`, up=1, down=3, window Kaiser 5.0,
padtype constant. Cari lag maksimum korelasi waveform dalam ±1 detik,
lalu hitung Pearson, gain least squares dan residual RMS setelah DC
dihilangkan. Laporkan semua metrik dan durasi; korelasi ≥0,999 dan
residual relatif ≤0,05 merupakan bukti kuat salinan yang diubah laju
sampelnya, bukan bukti rekaman independen. Tidak menguji transformasi lain.

## Representasi dan pencarian offset

Gunakan RMS native per 10 ms yang sudah disimpan, tanpa membaca `full_text`.
Hitung log(RMS) dengan floor max(median(RMS)×0,001, 1e-12), kemudian
positive first difference, dengan nilai awal nol. Winsorize pada persentil
99,5 dan simpan fitur ini. Tidak memilih parameter dari hasil per rekaman.
Seluruh audio tanpa identitas tombol digunakan untuk transformasi dan
pencarian; audio bagian akhir bukan holdout yang sama sekali tak dilihat.

Event adalah initial physical **press** menurut audit state press/release.
Buang hanya row press-while-held yang sudah ditandai audit; autorepeat OS
tidak diasumsikan ketukan fisik baru. Pertahankan row, timestamp dan semua
diagnostik. Nama tombol tidak masuk korelasi. Titik nol log adalah timestamp
event pertama dalam fase; hitung selisih integer sebelum konversi float.

Bagian fit adalah waktu 0 hingga setengah rentang fase; bagian validasi
adalah setengah rentang hingga event terakhir. Template impuls dijumlahkan
pada bin `floor(t/0,01)` dan dihaluskan Gaussian σ=1 bin, truncate=3,
mode constant. Template dipotong tepat pada batas fit/validasi. Jarak lag
10 ms. Konvensi: audio_time = log_relative_time + offset.
Jumlah bin = floor(span_ms/10)+1, batas fit = floor(jumlah bin/2).
Kuartal memakai floor(jumlah bin×q/4). Pembulatan batas maksimal satu bin.

Cari offset fit dengan Pearson pada overlap, menggunakan FFT correlation
dan jumlah kumulatif untuk normalisasi. Kandidat harus memuat minimal
50% durasi template fit dan 50% event fit. Minimal 25 press pada tiap
bagian diperlukan untuk menilai flag diagnostik. Pencarian tidak dibatasi
ke metadata creation_time. Metadata itu jam perangkat dengan ketelitian
detik, bukan ground truth waktu ketukan. Offset negatif dan log yang
terpotong pada awal/akhir diperbolehkan dan dilaporkan.

Pilih korelasi tertinggi; jika tepat sama pilih lag terkecil. Simpan puncak
alternatif di luar ±250 ms dari pilihan dan margin keduanya. Tidak memilih
offset lagi setelah melihat validasi. Validasi memakai template bagian
akhir dan offset yang telah dipilih fit, dengan aturan overlap yang sama.
Simpan lag lokal terbaik dalam ±250 ms hanya sebagai diagnostik perubahan
waktu, tanpa menggeser offset hasil fit. Lakukan juga pemeriksaan lokal
empat kuartal log untuk melihat ketidakstabilan; jangan memperbaiki drift
atau offset per blok pada studi ini.

## Kontrol waktu dan batas interpretasi

Untuk fit, buat 19 kontrol dengan urutan inter-press gap yang diacak;
pertahankan press pertama dan terakhir, jumlah event dan histogram gap.
Cari ulang maksimum setiap kontrol dengan ruang lag/overlap yang sama.
Untuk validasi, buat 99 kontrol gap-permuted pada offset fit yang tetap.
Seed per pasangan adalah 17282184 + indeks pasangan (urutan pairing audit).
Laporkan (1 + jumlah kontrol ≥ observasi)/(1 + jumlah kontrol).
Ini kalibrasi diagnostik dengan resolusi terbatas, bukan pengujian
konfirmatori atau p-value yang sudah dikoreksi untuk seluruh rekaman.

Flag **provisional stable match**, bukan sertifikasi label, mensyaratkan:
≥25 press dan ≥80% event tercakup pada masing-masing bagian; korelasi
fit dan validasi ≥0,10; margin puncak fit ≥0,02; tidak ada dari 19 kontrol
fit atau 99 kontrol validasi yang menyamai observasi; lag validasi lokal
berbeda ≤30 ms. Laporkan seluruh pasangan yang lolos maupun gagal.
Kriteria ini ditetapkan sebelum metrik dihitung, tidak dituning ulang.
Ketidaklulusan tidak membuktikan log salah: fitur energi, drift, suara
release, kebisingan, dan potongan rekaman juga dapat menggagalkannya.

Sebagai diagnostik tambahan, cari puncak onset di atas persentil 75 fitur
audio dengan jarak minimal 30 ms. Laporkan median dan p90 jarak absolut
press validasi ke puncak terdekat serta fraksi dalam 30 ms; banyak press
boleh mendekati puncak sama. Ini bukan precision/recall detektor, anotasi
manual, atau bukti setiap huruf memiliki potongan audio yang bersih.

## Keluaran dan langkah sesudahnya

Simpan lock sebelum perhitungan, hasil per pasangan, seluruh kontrol,
kurva korelasi fit, template, fitur dan indeks bin di NPZ; CSV ringkas,
peta offset serta panel overlay awal/akhir untuk peninjauan terpisah.
Bedakan event di luar rekaman dari event valid. Simpan hitungan huruf dan
jarak antar-press hanya sebagai deskripsi, tanpa hasil klasifikasi.

Paper EchoStrike menjelaskan penyesuaian offset manual sebelum segmentasi
(Bagian III.D); arsip yang diperiksa belum menyediakan tabel offset.
Jangan menganggap prosedur otomatis ini reproduksi literal prosedur itu.
Protokol berikutnya harus menetapkan pemeriksaan alignment, kosakata,
jendela, sumber model, split per M4A dan anggaran label sebelum mengevaluasi
klasifikasi. Phase 1 dan Phase 2 adalah rekaman terpisah; metadata belum
menetapkan sesi, mikrofon atau keyboard fisik baru.

Sumber: [SKAID v1](https://zenodo.org/records/17282184),
[paper asli EchoStrike](https://www.researchgate.net/publication/405307424_From_Keystrokes_to_Semantics_Acoustic_Side-Channel_Attacks_on_Typed_Text_Using_Deep_Neural_Networks_and_Large_Language_Models).
