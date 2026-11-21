# Pilot lanjutan: kepala linear pada CoAtNet beku

Ditentukan 8 November 2026, sebelum evaluasi metode baru. Pilot Harrison
sebelumnya dan diagnostik optimisasinya tetap utuh. Data target sudah diperiksa
pada tahap sebelumnya, sehingga eksperimen ini tetap eksploratif dan bukan
holdout konfirmatori baru. Tidak memakai dataset pribadi.

## Perbandingan yang ditetapkan

Gunakan potongan asli yang sama: mono 16 kHz/400 ms, 36 kelas, dua domain.
Target event 0–14 menjadi kandidat kalibrasi; 15–24 tetap menjadi uji (360
event). K=1/2/3/5/10, seed dukungan 101/102/103, dukungan nested dari helper
pilot lama. Tidak ada validasi target, checkpoint selection, penalaan alpha
atau tambahan target tanpa label untuk fitting transformasi.

Tiga metode positif-K:

1. `pretrained_source_norm`: checkpoint CoAtNet sumber 600 epoch dari pilot,
   statistik normalisasi sumber tetap, backbone beku.
2. `random_source_norm`: arsitektur sama, inisialisasi seed 42 tanpa pelatihan,
   statistik sumber yang persis sama. Ini kontrol representasi, bukan model
   scratch yang dioptimisasi. Satu inisialisasi tidak mengukur seluruh variasi
   random feature.
3. `pretrained_support_norm`: checkpoint sumber yang sama, mean/std per-band
   dihitung hanya dari dukungan K yang dipilih, seperti pilot fine-tuning.
   Ini kontrol perubahan normalisasi input, bukan adaptasi BatchNorm.

Buang lapisan Linear terakhir setelah global pooling/Flatten, sehingga fitur
768 dimensi. Seluruh parameter `requires_grad=False`, seluruh submodul eval,
termasuk BatchNorm; jangan menghitung statistik BatchNorm dari target. Gunakan
inference_mode saat ekstraksi dan periksa seluruh bobot/buffer tidak berubah.
Tidak memakai augmentasi pada ekstraksi. Fitur dinormalisasi L2 per contoh,
kemudian RidgeClassifier(alpha=1, fit_intercept=True), seperti kepala BEATs
pilot lama. Untuk K>0, kepala hanya memakai 36K label dukungan target.

K=0 pada dua metode source_norm: kepala Ridge dilatih dari 648 contoh train
sumber, lalu uji target. Model pralatih juga membawa 72 label validasi sumber
yang dahulu dipakai memilih checkpoint; kontrol acak berbagi normalisasi dari
source train yang sama. K=0 support_norm tidak didefinisikan. Total 94 kondisi:
90 positif-K dan empat tanpa label target. Evaluasi tambahan domain sumber
(train 648, test 180) memeriksa kemampuan representasi pada sumber; tidak
digunakan memilih hyperparameter.

Fitur source_norm boleh dicache sekali karena transformasi per contoh tetap
dan tidak mengestimasi statistik dari target. Fitur support_norm dicache per
dukungan, dengan indeks/ID dan statistik yang persis dihitung dari dukungan.
Semua file input, checkpoint dan kode diikat checksum. Simpan kepala, skor,
identitas dukungan/uji, normalisasi, waktu fit dan waktu ekstraksi terpisah.
Komputasi GPU hanya RX 6600 LE; fitting Ridge dan audit memakai CPU.

## Audit dan penafsiran

Ulangi kepala tersimpan dari fitur cache pada CPU; confusion matrix dan metrik
harus cocok. Verifikasi pembagian, jumlah label, dan mean/std. Ulangi ekstraksi
dari subset input mentah fitur mel pada CPU, bandingkan dengan GPU, dan periksa
bobot/buffer encoder tidak berubah. Ini tidak mengklaim floating-point bitwise
identik antarbackend. Bandingkan dengan BEATs dari pilot pada dukungan/test yang
sama; jangan melatih ulang atau memilih konfigurasinya dari uji baru.

Laporkan top-1, macro-F1, top-5, rerata ± SD tiga dukungan, serta selisih
berpasangan dari kontrol normalisasi dan kontrol random. Variasi dukungan bukan
variasi sesi, dan 360 event yang dipakai berulang tidak menjadi sampel independen
tambahan. Perbandingan dengan fine-tuning sebelumnya juga mengubah classifier,
optimizer dan anggaran; bukan ablation murni efek pembekuan bobot.

## Pemeriksaan terpisah atas model yang kolaps

Tanpa pelatihan atau evaluasi uji baru, periksa 360 dukungan K=10 seed 101 pada
checkpoint scratch panjang ponsel yang gagal dan Zoom yang berhasil. Tambahkan
inisialisasi seed 42 dan model sumber ponsel sebagai referensi. Ukur variasi
aktivasi setiap tahap dan ReLU serta rentang logit. Ukur gradien dari satu batch
seimbang (satu dukungan per tombol), pada mode eval dan pada salinan mode train.
Tidak ada optimizer step; perubahan buffer BN pada salinan train dibuang.

Jika ditemukan ReLU yang seluruh keluarannya nol meskipun input bervariasi,
uji penggantian ReLU tersebut dengan LeakyReLU(0,01) pada salinan model saja,
dengan bobot dan input sama. Bandingkan variasi logit dan aliran gradien pada
dukungan, tanpa memilih dari akurasi uji. Ini pemeriksaan lokal mekanisme,
bukan perbaikan model terlatih, penetapan SOTA, atau bukti generalisasi.
