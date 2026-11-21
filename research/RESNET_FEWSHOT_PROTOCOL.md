# Efisiensi label CNN residual: Zoom ke ponsel

Ditentukan 8 November 2026 setelah perbaikan baseline selesai dan sebelum
pretraining/evaluasi baru di bawah ini. Arsitektur tetap compact-resnet
311.892 parameter. Implementasi dan hasil lama tidak diubah. Ini pilot
eksploratif satu arah; seluruh rekaman Harrison pernah dipakai sebelumnya,
sehingga hasil tidak menjadi holdout konfirmatori baru.

## Pertanyaan dan blok berpasangan

Apakah inisialisasi dari pelatihan sumber membantu dibanding scratch ketika
jumlah label target dibatasi? Fokus **Zoom→ponsel**; perbedaan mikrofon,
akuisisi, kanal dan waktu/penekanan belum terisolasi.

- Data, hash, frontend 16 kHz/400 ms/mel 64×64 dan augmentasi sama dengan
  baseline stabil. Tetap 36 kelas dan potongan dari WAV eksternal Harrison.
- Tiga blok: (inisialisasi/pelatihan 42, dukungan 101), (43,102), (44,103).
  Setiap blok memakai seed yang sama untuk pretraining sumber, scratch dan
  adaptasi. SD antarblok mencakup pilihan dukungan dan inisialisasi sekaligus;
  desain ini tidak memisahkan kedua komponen variasi secara faktorial.
- Target K=1/2/3/5/10 per tombol, total 36/72/108/180/360 label per fit.
  Dukungan nested dalam blok, dipilih dari event 0–14, dan uji tetap seluruh
  360 event 15–24. Tidak menggunakan target lain untuk fitting statistik,
  validasi, early stopping atau pemilihan hyperparameter.

## Pretraining sumber dan penguncian

- Tiga model sumber memakai split Zoom stratifikasi seed 42 lama: train
  648 (18/tombol), validasi 72 (2/tombol), test 180 (5/tombol).
- Per model 10.000 update, Adam 5e-4, batch 16 replacement, LR linear ke nol.
  Statistik per-band hanya dari 648 train. Circular shift ±40% waveform dan
  dua mask per sumbu mel, lebar 0–6, seperti baseline.
- Evaluasi validasi sumber setiap 500 update. Pilih checkpoint dengan
  akurasi validasi terbesar; pilih yang paling awal saat seri. Simpan state
  sebagai salinan tensor, termasuk buffer BatchNorm. Test sumber hanya
  dinilai setelah checkpoint dipilih, tanpa mengubah pilihan.
- Jika validasi terbaik suatu model <50% atau ada nilai non-finite, jangan
  lanjutkan fit target dari protokol ini. Simpan kegagalan.
- Kunci hash ketiga checkpoint sumber dan aturan target dalam satu berkas
  sebelum evaluasi ponsel baru. K=0 memakai model sumber serta normalisasi
  sumber, tanpa label target; laporkan ketiganya.
- Biaya sumber: 648 label train +72 validasi **unik**, dipakai ulang pada
  tiga seed, bukan 720 label baru untuk setiap fit. Arsitektur sebelumnya
  dipilih melalui subset data sumber yang tercakup di dalam alokasi ini.

## Scratch dan adaptasi

- **Scratch** dimulai dari inisialisasi acak pada seed blok tersebut.
- **Adaptasi** menyalin seluruh bobot model sumber sesuai blok, termasuk
  kepala 36 kelas dan buffer BatchNorm. Seluruh parameter dilatih kembali;
  BatchNorm beroperasi dalam mode train pada batch dukungan target.
- Optimizer Adam baru untuk keduanya; tidak membawa momentum/counter sumber.
  LR **sama, 5e-4**, dan tepat 10.000 update, batch 16 replacement, linear
  decay ke nol. Tidak memilih LR atau checkpoint dari uji target. Ini
  membandingkan efek inisialisasi sumber pada satu resep fine-tuning tetap.
- K>0: mean/std per-band dihitung hanya dari dukungan K yang sama untuk
  kedua metode. Jadi perubahan K=0→K>0 mencakup normalisasi dan fine-tuning.
- Reset RNG pelatihan setelah konstruksi/pemuatan model. Dalam pasangan
  blok/K, urutan batch dan augmentasi harus sama; simpan fingerprint pada
  update 1/100/1000/3000/6000/10000 untuk diverifikasi.
- Catat loss/gradien, akurasi dukungan pada 1.000/3.000/6.000/10.000 update,
  waktu dan memori. Tidak mengevaluasi target test selama loop latihan.
  Evaluasi bobot terakhir satu kali setelah seluruh update fit tersebut.
- Total: **3 fit sumber, 30 fit target, 33 evaluasi target termasuk K=0**.
  Fit baru seluruhnya; tidak memilih/memakai ulang seed terbaik kontrol lama.
  Seluruh fit GPU hanya RX 6600 LE; audit checkpoint menggunakan CPU.
- Simpan checkpoint pemulihan periodik dengan optimizer, scheduler, RNG dan
  history. Resume hanya dengan signature yang sama; jangan diam-diam memulai
  ulang fit parsial atau menimpa eksperimen lain.

## Analisis yang ditetapkan

Untuk setiap K/metode laporkan semua blok, rerata ± sample SD top-1,
macro-F1/top-5, dukungan dan selisih adaptasi−scratch berpasangan. SD tiga
blok tidak menggambarkan variasi populasi atau sesi; jangan menganggap
360×3 sebagai jumlah event uji independen. Perbedaan tanda antarblok tetap
ditampilkan. Hindari uji signifikansi dengan tiga blok kecil ini.

Laporkan K pertama **pada grid yang diuji** dengan rerata >=80%/90%, atau
belum tercapai sampai K=10. Itu bukan jumlah minimum yang presisi atau
jaminan kebutuhan label pada kondisi lain. Target 90% yang tidak tercapai
tetap dilaporkan. Catat penguasaan dukungan untuk menilai underfitting;
jangan memperpanjang fit atau mengubah resep setelah melihat uji.

Audit CPU seluruh 3 checkpoint sumber dan 30 checkpoint target, prediksi
individual/metrik, ID, statistik input, jumlah label/update, sumber bobot,
optimizer baru dan fingerprint pasangan. Laporan menyertakan biaya
pretraining, bukan hanya biaya kalibrasi target. Hasil hanya mendukung
kesimpulan pada resep/dataset ini, tanpa menetapkan SOTA, generalisasi ke
sesi baru atau efektivitas seluruh strategi adaptasi.
