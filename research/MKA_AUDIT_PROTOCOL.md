# Audit kelayakan MKA v3

Ditentukan 8 November 2026 sebelum decoding audio dan evaluasi model MKA.
Pratinjau direktori ZIP sudah dibaca: ada enam folder platform dan folder
agregat, masing-masing memiliki nama kelas yang cocok dengan 36 tombol
Harrison. Nama folder belum membuktikan identitas keyboard atau sesi.

Sumber: Karwan Mahdi Rawf dan Ayub Abdulrahman,
[Multi-Keyboard Acoustic (MKA) Datasets v3](https://data.mendeley.com/datasets/bpt2hvf8n3/3),
DOI 10.17632/bpt2hvf8n3.3, CC BY 4.0. Unduhan memakai daftar berkas publik
yang dipublikasikan situs dan pin SHA256 penyedia. Arsip asli dipertahankan.

## Pemeriksaan

1. Verifikasi ukuran dan SHA256 seluruh unduhan, lalu CRC saat setiap
   anggota ZIP dibaca. Simpan inventaris dan hash anggota; tidak mengekstrak
   arsip ke path yang diberikan data.
2. Pisahkan enam folder platform dari `All Dataset`. Hitung audio mentah,
   potongan yang disediakan, fitur, serta teks. Kelas bersama ditentukan
   hanya dari nama tombol `a–z,0–9`, tanpa memilih berdasarkan akurasi.
3. Decode audio tanpa resampling, dengan seluruh kanal dan float64.
   Catat rate, kanal, frame, durasi, subtype, nilai non-finite, puncak/RMS,
   SHA256 berkas dan fingerprint PCM. Fingerprint PCM menyertakan rate,
   bentuk array, dan sampel float64 little-endian; ini mencakup seluruh kanal.
4. Periksa duplikasi byte/PCM, salinan agregat, konflik label, dan kecocokan
   persis terhadap audio Harrison lokal. Tidak ditemukannya salinan persis
   tidak membuktikan tidak adanya salinan yang ditransformasi.
5. Periksa isi teks yang disebut metadata: bedakan tabel numerik/objek fitur
   dari keterangan rekaman. Catat bukti eksplisit sesi, partisipan,
   perangkat, mikrofon, atau waktu jika tersedia. Jangan menciptakan ID
   sesi/keyboard dari nama platform atau nomor potongan.
6. Telusuri potongan yang disediakan ke WAV mentah berlabel sama untuk
   seluruh 36 kelas bersama, bila rate memungkinkan. Korelasi ternormalisasi
   pada satu kanal bervariansi terbesar menjadi bukti diagnostik asal
   potongan; simpan lag, korelasi bertanda, galat linear dan interval.
   Ambang kecocokan |r|>=0,999 ditetapkan di sini. Periksa kesamaan penuh
   semua kanal secara terpisah; jangan menyamakan korelasi tinggi dengan
   identitas byte. Interval potongan yang overlap dilaporkan.
7. Hitung kelayakan jumlah potongan unik untuk K=1/2/3/5/10, dengan anggaran
   uji 1/5/10 potongan per kelas. Ini hanya syarat jumlah, bukan bukti
   ketukan atau rekaman independen. Tanpa anotasi event, jumlah puncak
   amplitudo tidak dianggap jumlah ketukan berlabel yang valid.

## Keputusan sebelum pelatihan

Pertanyaan utama berikutnya adalah apakah manfaat adaptasi Harrison
bertahan ketika kalibrasi dan uji target berasal dari rekaman terpisah.
Untuk setiap kelas target, pembagian tersebut harus memiliki kelompok
rekaman yang berbeda dan metadata yang mendukung arti kelompoknya.
Potongan dari WAV induk yang sama, duplikasi, dan salinan `All Dataset`
tidak boleh melintasi kelompok kalibrasi/uji sebagai rekaman independen.

Jika syarat itu tidak didukung data/metadata, simpan keputusan tidak layak
untuk pertanyaan tersebut. Jangan menggantinya diam-diam dengan split
acak dalam rekaman atau memulai kurva akurasi. Kemungkinan evaluasi lintas
dataset tanpa kalibrasi, atau pilot dalam rekaman, dicatat sebagai
pertanyaan berbeda yang memerlukan protokol sebelum evaluasi model.

Keluaran: inventaris berkas/audio/kelas, laporan duplikasi dan provenance,
hasil penelusuran potongan, tabel kecukupan label, keputusan split dan
catatan batas. Tahap audit memakai CPU; tidak menjalankan pelatihan GPU.
