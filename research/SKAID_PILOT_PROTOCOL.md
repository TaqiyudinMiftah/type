# Pilot klasifikasi SKAID: anggaran label pada rekaman terpisah

Tanggal lokal: 9 November 2026. Ditulis setelah audit native, offset,
drift dan kelayakan jendela selesai, sebelum mengekstrak atau mengevaluasi
fitur klasifikasi SKAID. Hasil waktu/label sudah diperiksa; akurasi huruf
belum diketahui. Studi eksploratif dengan dataset publik yang dipatok.

## Pertanyaan dan batas anggaran

Ukur pengaruh K=1/2/3/5/10 label per huruf pada klasifikasi crop yang
disediakan timestamp acuan, dengan kalibrasi Phase 1 dan query Phase 2.
Ini efisiensi label **kepala klasifikasi, bersyarat segmentasi/filter
berbasis log**. Timestamp seluruh event, kepemilikan press/release,
identitas kelas untuk stratifikasi dan sampling query memanfaatkan
anotasi dataset. Biaya tersebut tidak tercakup dalam 12×K label dukungan;
hasil bukan kebutuhan label end-to-end untuk rekaman tanpa anotasi.

Gunakan tepat17 ID pada baris audit drift: affine_policy, crop80 ms,
no_other_key_event, exploratory12, K10, query_min_per_class5. Jangan
memilih ID dari akurasi. Urutan kelas alfabetis adalah `adehilnorstu`
(subset `etaoinshrdlu`); hasil tidak diekstrapolasi ke seluruh alfabet.
Fase memiliki M4A terpisah, tetapi metadata belum membuktikan mikrofon,
keyboard fisik atau sesi baru. Tidak membaca `full_text` atau memakai LM.

## Pemilihan event dan penguncian split

Semua candidate adalah initial press yang lulus mask80 ms strict studi
drift. Per ID, pilih tepat5 query unik per kelas Phase 2 dengan
`default_rng(20261109 + indeks_ID)`, per kelas dalam urutan alfabetis.
Query tetap untuk seluruh K, draw dan encoder. Untuk dukungan, pakai
seluruh candidate Phase 1 sebagai pool. Draw101/102/103 masing-masing
memakai `default_rng(draw + 100000×indeks_ID)` dan permutasi per kelas;
prefix K menghasilkan dukungan nested dengan tepat12×K label unik.
Tidak ada validasi target tambahan. Simpan seluruh pool row, rank dan
dukungan; jangan merata-ratakan draw sebagai dataset independen.

Ekstrak hanya union dukungan K10 dari tiga draw dan query tetap. Semua
K lebih kecil adalah subsetnya. Urutkan clip menurut ID, fase, row log;
simpan mapping indeks. Kunci plan sebelum decoding/caching/evaluasi.
Pertahankan seluruh ID jika suatu tahap gagal; laporkan kegagalan dan
berhenti sebelum mengganti split. Reject duplikasi PCM persis, audio
tidak finite, crop tidak lengkap atau overlap train/query pada file yang
sama. Hitung overlap crop dalam fase, tanpa menganggap query satu file
sebagai rekaman independen. Augmentasi tidak digunakan.

## Audio dan frontend

Dekode M4A native dengan helper yang sudah diaudit dan cocokkan hash PCM
keseluruhan file. Ambil crop80 ms, pre15 ms, pada frame native yang sudah
diproyeksikan model jam; tidak menyesuaikan ke peak. Native M4A mono48 kHz.
Resample crop yang telah dipotong ke16 kHz, up1/down3, Kaiserβ5.0,
padtype constant, float32. Panjang output1280 sampel. Resampling setelah
crop mencegah filter mengambil audio dari luar interval yang dipilih.
Tidak melakukan RMS/amplitude normalization, noise reduction atau gain.

Untuk neural encoder, sisipkan audio80 ms dalam buffer nol400 ms/6400
sampel mulai sampel560 (35 ms), sehingga press acuan berada pada50 ms.
Tidak meregangkan waktu atau mengubah pitch. Padding diperlukan untuk
frontend BEATs/model sumber lama. Model pralatih menerima konteks suara
lebih singkat daripada sumber; distribusi input berbeda dan harus
dicatat. MFCC memakai80 ms asli tanpa padding400 ms.

## Representasi, seluruhnya beku

1. **MFCC78**: mel32, n_fft=win_length256, hop80, Hann, center=False,
   power2, fmin20/fmax7600 Hz, Slaney/non-HTK. Log-power ref1, amin1e-10,
   top_db80. DCT-II ortho13 MFCC, lifter0. Delta orde1 dan2, width9,
   mode nearest. Rerata dan population-SD tiap kanal menghasilkan78 fitur.
   Satu clip80 ms menghasilkan13 frame; tidak mempelajari statistik global.
2. **BEATs**: iter3+ AS2M yang sudah dipatok pada riset sebelumnya,
   checkpoint SHA256 `d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34`.
   Mean pooling16 token768 dimensi pada input padded400 ms, model eval,
   requires_gradFalse, tanpa update state. Bobot generik AudioSet, bukan
   model keystroke yang telah dinyatakan terbaik.
3. **ResNet pralatih**: seluruh tiga checkpoint sumber Zoom Harrison
   seed42/43/44 yang sudah diaudit. Frozen embedding128 dimensi sebelum
   head36 kelas, frontend lama64×64 dan mean/std sumber masing-masing.
   Bobot dan seluruh buffer BatchNorm beku. Biaya sumber720 label unik
   (648 fit +72 validation), dipakai ulang dari studi terdahulu.
4. **ResNet acak**: arsitektur sama, seed42/43/44, inisialisasi helper
   asli tanpa checkpoint. Pakai mean/std Harrison yang sama dengan seed
   pasangannya. Tidak melatih bobot; simpan checkpoint random untuk replay.
   Ini kontrol representasi, bukan scratch end-to-end yang dioptimasi.

Seluruh ekstraksi neural GPU hanya RX6000, dengan configure_rx6000 dan
strict select_device('gpu'); jangan jatuh ke780M. ResNet juga memakai GPU.
Rekam perangkat, versi, waktu, puncak memori dan digest state sebelum/
sesudah. Ekstraksi CPU independen pada tahap audit tidak memakai GPU.
Kontrol sumber BEATs pada36 clip Harrison lama dan logits source-test
ResNet180 contoh harus cocok dengan cache sumber dalam atol/rtol1e-3,
dengan prediksi ResNet identik. Ini pemeriksaan numerik, bukan pemilihan
model atau evaluasi tambahan untuk memilih hyperparameter target.

## Kepala dan kontrol yang ditetapkan sebelum akurasi

Semua fitur dikonversi float64, L2 per baris; tidak ada scaler/PCA atau
normalisasi yang dipelajari dari query. Gunakan RidgeClassifier(alpha1,
solver cholesky, fit_intercept=True, class_weight=None). Urutan kelas0–11.
Setiap ID/representasi memakai tiga draw, lima K, dan query identik.
Enam encoder ResNet (3 pretrained +3 random) disilangkan penuh dengan
tiga draw. Total8 encoder ×17 ID ×15 kondisi =2.040 fit normal.

Kontrol negatif hanya K10, semua encoder/ID/draw: permutasi label120
dukungan pada tingkat contoh, seed997+100000×indeks_ID+draw. Jumlah tiap
label tetap10. Query tidak diubah. Tambahan408 fit, total2.448 kepala.
Simpan skor query dan dukungan, koefisien, intercept dan fit labels.
Laporkan akurasi dukungan terhadap fit-label dan label asli untuk kontrol.

Tambahkan K0 pada tiga head sumber ResNet36 kelas yang sudah beku,
membatasi logits ke12 kelas subset. Tanpa fit target. Ini oracle kosakata
yang diketahui sebelumnya, dan bukan baseline K0 bagi MFCC/BEATs.
Semua cache fitur dan split harus selesai/terverifikasi sebelum runner
klasifikasi menghitung metrik. Tidak memilih alpha, checkpoint, seed,
frontend atau panjang crop dari akurasi query.

## Pelaporan dan ketidakpastian

Simpan seluruh2448 kondisi, termasuk kontrol gagal. Metrik utama top1;
tambahan macroF1/top3. Query60 per ID seimbang, peluang uniform1/12.
Rerata ResNet dalam keluarga mencakup tiga model×tiga draw; bukan ensemble
logit atau seed terbaik. MFCC/BEATs mencakup tiga draw. Reratakan per ID
lebih dahulu, lalu laporkan mean dan sample SD antar17 ID.

95% percentile bootstrap10.000 resample **ID** dengan seed7471, termasuk
paired-delta antar keluarga pada K yang sama. Tidak bootstrap crop/draw
seolah independen. CI bersyarat pada17 ID yang lolos filter, bukan CI
mikrofon/sesi/keyboard baru. Laporkan K terkecil pada grid yang mencapai
rerata80%/90% jika ada, dengan seluruh kurva dan ketidakpastiannya;
itu bukan penetapan jumlah label minimum yang pasti.

## Verifikasi

Tes diperlukan untuk split nested/seimbang/disjoint, input padding dan
resampling, penolakan ketidaksesuaian label/provenance, kontrol negatif,
dan Ridge terhadap dual centered solve independen. Semua bobot lama dan
protokol dipertahankan; gunakan modul baru untuk studi ini.

Audit decode/crop ulang semua clip terpilih, hash dan label/split; replay
seluruh2448 kepala dan51 evaluasi K0 dari cache. Bandingkan seluruh kepala
dengan penyelesaian dual ridge independen (toleransi1e-8). Ekstrak ulang
semua embedding/MFCC di CPU, verifikasi terhadap cache GPU atol/rtol1e-3,
dan laporkan selisih/prediksi yang berubah saat semua kepala di-fit serta
dievaluasi pada fitur CPU. Jangan menyembunyikan perbedaan numerik atau
menyesuaikan resep dari hasilnya. Ekspor skor/prediksi, CSV per kondisi,
grafik K dan laporan dengan batas segmentasi/kosakata/seleksi ID.

Referensi teknis: [MFCC librosa0.11](https://librosa.org/doc/0.11.0/generated/librosa.feature.mfcc.html),
[delta](https://librosa.org/doc/0.11.0/generated/librosa.feature.delta.html).
Backbone dan lisensi/provenance mengikuti artefak Harrison/BEATs yang
sudah dipatok dalam studi sebelumnya; bukan klaim reproduksi literal
klasifikasi EchoStrike atau evaluasi detektor bebas timestamp.
