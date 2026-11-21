# Pemeriksaan kestabilan: satu aktivasi terakhir

Ditentukan 8 November 2026, setelah autopsi support-only menemukan keluaran
ReLU `0.mlp4.1` nol pada seluruh 360 dukungan model scratch ponsel yang gagal.
Input ke ReLU berada pada rentang negatif; gradien input dan gradien blok
sebelumnya nol pada mode eval dan train-copy. Intervensi LeakyReLU(0,01) pada
salinan checkpoint memulihkan variasi logit dan gradien, tanpa optimizer step.
Ini dasar hipotesis, bukan bukti bahwa pelatihan baru akan berhasil.

Eksperimen ini eksploratif dan ditujukan pada kegagalan ponsel yang sudah
dilihat, bukan preregistrasi holdout baru. Seluruh hasil lama tetap utuh.

## Kondisi yang dibekukan sebelum pelatihan baru

- Target ponsel Harrison, K=10, seed dukungan 101: 360 label dukungan dari
  kandidat event 0–14 dan uji tetap 360 event 15–24. Tidak ada validasi target.
- Dua varian: model asli ReLU dan varian yang hanya mengganti `0.mlp4.1`
  menjadi LeakyReLU(negative_slope=0.01). Bobot awal untuk seed yang sama
  persis sama; tidak memodifikasi kode vendor atau checkpoint lama.
- Seed pelatihan 42, 43, 44; laporkan semuanya, jangan memilih seed dari uji.
  Kontrol ReLU seed 42 memakai hasil diagnostik 10.000 update yang sudah ada,
  dengan checksum dan identitas dukungan/config diperiksa. Lima fit baru:
  ReLU 43/44 serta LeakyReLU 42/43/44.
- Seluruh fit baru dimulai dari inisialisasi, 10.000 update, Adam lr 5e-4,
  batch 16 dengan pengembalian, schedule linear ke nol selama 10.000 update.
  Frontend, normalisasi dukungan dan augmentasi sama dengan diagnostik lama.
- Simpan loss dan norm gradien blok sebelum aktivasi dari backward yang sama
  (tanpa forward tambahan) pada update yang ditetapkan. Akurasi dukungan
  diperiksa pada 1.000/3.000/6.000/10.000. Tidak ada early stopping.
- Uji hanya checkpoint terakhir. Ukur variasi logit pada dukungan, rentang
  input/output aktivasi terakhir dan gradien untuk menandai kegagalan.
- GPU RX 6600 LE, fit berurutan. Kepala linear/audit memakai CPU.

## Pelaporan

Tujuan utama adalah menilai apakah perubahan mencegah keluaran konstan dan
memungkinkan model menguasai dukungan, pada tiga inisialisasi. Akurasi uji
sekunder tetap dilaporkan, bukan dipakai memilih varian atau checkpoint.
Bandingkan setiap seed yang sama dan laporkan rerata ± SD atas inisialisasi;
dukungan dan uji tetap sama, sehingga bukan variasi sampel atau sesi.

Perubahan hanya pada satu aktivasi tidak membuktikan optimalitas arsitektur,
SOTA, atau kestabilan pada semua seed/domain. Kegagalan yang tersisa tetap
dicatat; jangan menambah percobaan sampai mendapatkan hasil yang diinginkan.
Jangan mengklaim kebutuhan label minimal atau keunggulan adaptasi dari kontrol
ini. Konfirmasi pada rekaman independen masih diperlukan.
