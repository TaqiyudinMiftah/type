# Diagnostik tambahan: anggaran optimisasi scratch

Ditambahkan setelah keluaran K=1 pertama pada pilot v1 menunjukkan akurasi
dukungan scratch hanya 33,33% dan 22,22% untuk dua draw pertama pada target
Zoom, sedangkan model adaptasi draw pertama menguasai seluruh dukungan. Ini
**diagnostik pascahasil**, bukan bagian dari protokol utama yang sudah dibekukan.
Hasil dan checksum pilot utama tidak diubah. K=10 belum dievaluasi saat
diagnostik ini ditentukan.

Tujuan: memeriksa apakah baseline scratch pada batas 1000 update terlalu
terbatas optimisasi untuk menafsirkan selisih uji sebagai efisiensi label.

Lakukan satu fit tambahan per target (Zoom dan ponsel), K=10, support seed 101,
training seed 42, dari inisialisasi acak. Pilihan K terbesar menguji kontrol
dengan data dukungan terbanyak pada grid, bukan memilih K dari akurasi uji.
Gunakan potongan, dukungan, uji, statistik normalisasi, augmentasi, Adam dan
learning rate awal yang identik dengan scratch pilot. Ubah anggaran menjadi
**10.000 update** dan schedule linear turun ke nol selama 10.000 update.
Ini pelatihan ulang dari awal; berbeda dari melanjutkan bobot dengan LR nol.

Tidak ada early stopping, label validasi target, atau pemilihan dari uji.
Catat loss setiap 500 update, serta akurasi dukungan tanpa augmentasi pada
update 1.000, 3.000, 6.000 dan 10.000. Evaluasi uji hanya pada checkpoint
terakhir. Simpan bobot, skor, indeks/ID, normalisasi, waktu dan memori.
Jalankan setelah seluruh pilot utama selesai, secara berurutan pada RX 6600 LE.

Bandingkan scratch 1.000, scratch 10.000 dan adaptasi 1.000 **pada seed 101
dan dukungan K=10 yang persis sama**. Jangan membandingkan satu draw diagnostik
dengan rata-rata tiga draw tanpa penjelasan. Laporkan sebagai pemeriksaan
optimisasi, bukan kurva few-shot baru, replikasi independen, atau bukti SOTA.
Tambahan komputasi sumber/adaptasi dan learning rate berbeda tetap perlu
dipertimbangkan. Bahkan akurasi dukungan tinggi tidak menjamin konvergensi
generalisasi atau memilih hyperparameter yang optimal.
