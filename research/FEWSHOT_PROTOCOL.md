# Pilot kalibrasi kanal Harrison — protokol v1

Ditentukan sebelum menjalankan evaluasi pilot, 7 November 2026. Ini eksperimen
eksploratif: kedua kanal dan seluruh rekaman Harrison telah diperiksa pada
eksperimen sebelumnya. Pembagian baru bukan holdout konfirmatori yang belum
pernah dilihat. Jangan memilih metode atau hyperparameter dari hasil uji ini.

## Pertanyaan dan data

Berapa contoh berlabel per tombol yang membantu berpindah antara rekaman ponsel
dan Zoom? Bandingkan dua arah, 36 kelas, memakai data eksternal Harrison pada
commit yang sama dengan reproduksi sebelumnya. Tidak memakai data pribadi baru.
Mikrofon, pemrosesan kanal, waktu rekaman, dan kemungkinan perbedaan penekanan
belum dapat dipisahkan sebagai penyebab; ini perpindahan domain rekaman.

Potong ulang WAV asli memakai pusat event dalam manifest reproduksi, bukan
memperpanjang potongan lama dengan padding. Ubah seluruh rekaman mono ke 16 kHz
sebelum memotong 400 ms (50 ms sebelum pusat; 350 ms sesudah). Periksa hash sumber,
batas file, jarak event dan duplikasi. Label isolasi J/phone tetap tidak pasti
karena pencarian ambang tambahan pada reproduksi sebelumnya. Pusat onset masih
berasal dari detektor yang dibantu pengetahuan jumlah ketukan, bukan event log.

Untuk masing-masing target, event urutan 0–14 per tombol merupakan kumpulan
kalibrasi; urutan 15–24 merupakan uji tetap (360 contoh). Dukungan dipilih tanpa
pengembalian memakai seed 101, 102, 103, dengan urutan acak per tombol yang sama
untuk seluruh K sehingga K kecil merupakan subset K besar. K = 1, 2, 3, 5, 10;
anggaran total = 36K label target. Tidak ada validasi target, pemilihan checkpoint
target, penalaan hyperparameter target, atau tambahan data target tanpa label.
K=0 berarti nol label target dan hanya dinilai untuk model yang memiliki kepala
klasifikasi sumber. Pelatihan sumber memakai split stratifikasi 18/2/5, seed 42,
yang dicatat terpisah dari split target. Ketika arah dibalik, data sumber dan
target juga bertukar peran; kedua arah bukan replikasi independen.

## Model dan anggaran komputasi yang ditetapkan

1. **CoAtNet scratch**: backbone kecil dari referensi reproduksi (~1,97 juta
   parameter), inisialisasi acak, hanya dukungan target.
2. **CoAtNet adaptasi**: backbone dan kepala 36 kelas yang sudah dilatih pada
   kanal sumber, kemudian seluruh parameter dilatih menggunakan dukungan target.
3. **BEATs + linear**: BEATs iter3+ AS2M pretrained, encoder dibekukan pada mode
   evaluasi. Rata-rata token keluaran, normalisasi L2 setiap vektor, kemudian
   RidgeClassifier(alpha=1, fit_intercept=True). Untuk K>0, kepala hanya dilatih
   dari dukungan target; K=0 memakai kepala yang dilatih dari 648 contoh sumber.
   Ini pembanding encoder audio pretrained, bukan klaim SOTA keystroke.

CoAtNet: log-mel 64 band, FFT 512, hop 160 pada 16 kHz, dynamic range 80 dB,
resize bilinear 64×64. Statistik per-band mean/std dihitung hanya dari data
latih: sumber-train saat pretraining/K=0; dukungan target yang persis sama untuk
scratch dan adaptasi pada K>0. Karena statistik juga berubah, selisih K=0 ke
K>0 mencakup kalibrasi normalisasi. Tidak melakukan normalisasi dari data uji.

Pelatihan sumber: 600 epoch, Adam lr 5e-4, batch 16, learning rate turun linear
ke nol; validasi sumber setiap 5 epoch dan epoch pertama/terakhir, simpan
checkpoint dengan akurasi validasi tertinggi (yang paling awal saat seri).
Ini anggaran pilot lebih kecil daripada 1100 epoch reproduksi terdahulu.

Scratch/adaptasi: masing-masing tepat 1000 update, batch 16 diambil dengan
pengembalian dari dukungan, Adam, lr scratch 5e-4 dan adaptasi 1e-4, turun linear
ke nol. Seed model/augmentasi 42 untuk seluruh draw; variasi antardraw terutama
menunjukkan sensitivitas pemilihan dukungan, bukan distribusi semua inisialisasi.
Tidak ada early stopping target; gunakan bobot update terakhir. Kedua metode
memakai augmentasi yang sama: circular time shift ±40% panjang waveform dan dua
mask per sumbu log-mel (lebar 0–6 dari 64). Catat loss dan akurasi dukungan tanpa
augmentasi untuk mendeteksi pelatihan yang belum memadai. Anggaran update sama
tidak berarti kedua metode telah mencapai konvergensi; simpulan dibatasi oleh
anggaran komputasi ini dan pilihan learning rate yang berbeda.

BEATs memakai frontend checkpoint aslinya: Kaldi fbank 128 band, frame 25 ms,
hop 10 ms, waveform ×32768, (fbank−15.41663)/(2×6.55582). Audio fisik sama,
representasi/frontend berbeda sesuai masing-masing model. Patch stride BEATs
dapat membuang frame terakhir yang tidak membentuk patch penuh; catat bentuk
fbank/token aktual. Tidak memanjangkan klip agar cocok dengan model.

Komputasi neural hanya RX 6600 LE, tidak Radeon 780M. Fitur BEATs diekstrak
sekali dan disimpan; statistik yang dipelajari dan kepala klasifikasi tetap
dibatasi pada train/dukungan. Ekstraksi individual pada mode eval tidak
mempelajari data uji. Catat waktu ekstraksi, pretraining, adaptasi, inferensi,
dan puncak alokasi memori GPU secara terpisah. Cache dan run memuat checksum
input, checkpoint, sumber vendor, protokol, indeks/ID dan konfigurasi.

## Pelaporan

Untuk setiap K dan arah: akurasi top-1, macro-F1, top-5, jumlah prediksi benar,
kurva rata-rata ± simpangan baku sampel dari tiga draw dukungan, dan selisih
adaptasi–scratch yang dipasangkan pada draw yang sama. Jangan menggabungkan
360×3 sebagai ukuran sampel uji independen. Interval Wilson per evaluasi hanya
deskriptif karena ketukan masih berasal dari rekaman per tombol yang sama.

Laporkan K pertama pada grid yang mencapai rata-rata 80%/90%, atau “belum
tercapai sampai K=10”; bukan estimasi kebutuhan di populasi atau batas minimum
yang presisi. Semua hasil termasuk kegagalan dan kasus sulit tetap dilaporkan.
Sumber biaya/label pretraining dan data umum BEATs berbeda; jangan menyebut
perbandingan sebagai kontrol arsitektur murni. Jangan mengklaim penyebab transfer
telah terisolasi dari pilot ini. Konfirmasi berikutnya memerlukan dataset dengan
sesi/rekaman independen yang belum dipakai untuk pengembangan.
