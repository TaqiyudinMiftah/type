# Audit pasangan audio–log SKAID v1

Ditentukan sebelum unduhan/decoding audio SKAID pertama di proyek ini.
Waktu UTC dan hash protokol dicatat dalam lock audit. Audit hitungan
27 log sebelumnya sudah dibaca, termasuk satu log dengan timestamp
tunggal dan kelangkaan huruf pada Phase1. Belum ada prediksi model SKAID.

Sumber: Benjamin Quattrone dan Youakim Badr,
[SKAID v1, Zenodo](https://zenodo.org/records/17282184),
DOI 10.5281/zenodo.17282184, metadata API CC BY4.0. README menjelaskan
audio email sebagai Phase1 dan free-form sebagai Phase2. Klaim sinkronisasi
penulis belum dianggap bukti kecocokan setiap berkas publik.

## Masukan dan integritas

- Unduh arsip Participant Recordings_zip.zip (182.788.356 byte,
  MD5 penyedia e4c9fd0edf4586ac99a886ecaa15c275) dan tabel demografi
  (1.585 byte, MD5 8a3fdb18d7ca72defa0ce6e39d85e4a1) melalui endpoint
  publik record17282184. Verifikasi metadata versi/lisensi/ukuran/checksum,
  hitung SHA256 lokal, pertahankan arsip asli.
- Gunakan log lokal yang sudah diverifikasi, SHA256
  5e84c19209df891b88e8de600f171511e9af370be2de903d85353bd0b2c19eda;
  README SHA256 c51f24e74879a79f36d37e3da56ce7e79a23a5b83b227a5b2948c66f960b93e3.
- Baca CRC seluruh anggota ZIP. Jangan mengekstrak path dari arsip ke
  filesystem. File sementara decoder memakai nama yang dibuat sendiri.
- Jangan membaca CSV full_text atau mengunduh dokumen prompt untuk
  menyusun ulang label dari bahasa. Audio/log asli tetap tidak berubah.
- Seluruh audit CPU, tanpa GPU atau update model. Radeon780M dilarang.

## Pemeriksaan sebelum penetapan split

1. Inventaris seluruh anggota, nama/ID/fase yang benar-benar tercantum,
   ukuran dan hash. Jangan menyamakan indeks folder dengan ID log atau
   identitas partisipan tanpa bukti eksplisit.
2. Probe semua audio memakai ffprobe: codec, sample rate, kanal, durasi,
   timestamp/creation_time, stream/chunk/tag yang tersedia. Simpan versi
   ffmpeg/ffprobe dan seluruh JSON probe. Metadata waktu tanpa timezone
   tidak otomatis diubah menjadi UTC; waktu ZIP bukan anchor audio.
3. Decode audio pada rate/kanal asli ke PCM float32 dengan ffmpeg,
   periksa seluruh sampel finite, frame, RMS, puncak, clipping, dan
   fingerprint PCM yang menyertakan rate serta bentuk array. Periksa
   duplikasi byte/PCM termasuk pasangan email/free-form dan take tambahan.
   Decoder AAC dapat mempunyai priming; durasi container tidak dianggap
   alignment event yang presisi.
4. Simpan envelope RMS10ms dan puncak absolut10ms pada rate native untuk
   diagnostik berikutnya. Jangan menafsirkan jumlah puncak sebagai jumlah
   ketukan berlabel. Tidak menyesuaikan gain atau memilih kanal terbaik.
5. Baca hanya log event: pertahankan nomor baris, session ID, fase,
   timestamp integer dari Decimal, press/release dan nama tombol.
   Laporkan monotonisitas, duplikasi event, jumlah waktu unik, resolusi,
   press/release yang tidak berpasangan, rentang tiap fase, dan seluruh
   alasan masalah. Notasi ilmiah sendiri bukan bukti presisi hilang;
   timestamp identik sepanjang log tidak memberi urutan waktu valid.
6. Cari pairing berdasarkan pengenal/fase eksplisit pada nama berkas,
   folder, log dan tabel demografi. Jangan memakai urutan alfabetis,
   durasi terpanjang, dugaan kemiripan nama atau hasil model untuk
   memilih pasangan. Bila lebih dari satu take memenuhi nama, simpan
   semua kandidat dan tandai ambigu sampai ada bukti pemilihan.
7. Bandingkan rentang waktu log dengan durasi audio dan anchor berzona
   waktu jika ada. Pisahkan bukti file terpisah, tugas/fase terpisah,
   session ID dan partisipan menurut metadata. Dua fase dari satu log
   belum otomatis merupakan sesi atau mikrofon berbeda.
8. Hitung kecukupan kelas a–z dan a–z/0–9 untuk K=1/2/3/5/10 pada Phase1
   dan minimal satu event Phase2. Ini hanya hitungan; potongan valid,
   overlap, ketepatan waktu dan biaya uji belum ditetapkan.

## Keputusan dan kelanjutan

Keluaran tahap ini berupa inventaris/probe, log audit, pairing kandidat,
duplikasi, envelope dan keputusan kelayakan bersyarat. Tidak mengeklaim
alignment hanya dari nama file, jumlah event atau rentang durasi.

Jika metadata cukup untuk pairing tetapi waktu belum terverifikasi,
buat protokol alignment tersendiri sebelum mengestimasi offset/drift
dari audio. Penggunaan timestamp ground truth untuk crop harus disebut
oracle timing, terpisah dari detektor ketukan tanpa log. Jangan mengubah
aturan alignment berdasarkan akurasi classifier.

Jika pairing/timing atau kelompok rekaman belum mendukung kalibrasi/uji
terpisah, simpan keputusan beserta bukti dan cari sumber eksternal lain.
Jangan mengganti pertanyaan menjadi split dalam rekaman tanpa protokol
baru. Dataset/sesi sendiri tetap tahap berikutnya sesuai arahan pengguna.
