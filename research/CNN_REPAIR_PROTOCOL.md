# Alternatif CNN residual kompak

Ditentukan 8 November 2026 setelah sembilan fit pengembangan sebelumnya.
Belum ada fit/evaluasi ponsel baru pada tahap perbaikan. Stem lebar mencapai
dukungan 100%, 100%, 99,72%, validasi 47,22%, 52,78%, 68,06%; sample SD
10,79 pp sedikit melewati kriteria 10 pp. Keputusan gagal tetap utuh.

Kandidat terakhir pada protokol ini `compact-resnet`: CNN residual kompak
dengan stem Conv3x3 mono→16, BatchNorm, ReLU, empat BasicBlock berlebar
16/32/64/128 (satu per tahap). Setiap blok memiliki dua Conv3x3 tanpa bias,
BatchNorm, ReLU, dan shortcut identity atau Conv1x1+BatchNorm saat bentuk
berubah. Stride 1 di tahap pertama dan 2 pada tiga tahap berikut. Global
average pooling dan Linear128→36. Kaiming normal fan-out untuk konvolusi,
BatchNorm weight=1/bias=0; BN terakhir masing-masing cabang residual
diinisialisasi nol supaya awalnya mengikuti shortcut. Tidak memakai attention.
Ini implementasi residual CNN kompak, bukan ResNet-18 atau CoAtNet kanonis.

Hipotesis: backbone yang lebih kecil dengan residual sejak awal lebih cocok
untuk mengurangi variasi dari inisialisasi pada 10 contoh per kelas daripada
referensi hybrid yang dipakai. Kapasitas dan arsitektur berubah; efeknya tidak
dapat dianggap ablation satu unsur. Seluruh kandidat gagal tetap dilaporkan.

Data, pemilihan 360 fit Zoom dan 72 validasi, tiga seed 42/43/44, 6.000 update,
Adam 5e-4, batch 16 replacement, augmentasi dan linear decay sama persis.
Tidak mengubah kriteria: setiap seed dukungan >=95%, logit tidak konstan,
rerata validasi >=50%, sample SD validasi <=10 pp. Tanpa early stopping atau
pemilihan seed. Jika gagal, jangan evaluasi ponsel dari protokol ini.

Jika lulus, kunci keputusan beserta hash kedua tahap gagal, kode/protokol,
data dan cache, kemudian konfirmasi ponsel K=10 dukungan seed 101,
inisialisasi 42/43/44, 10.000 update dan konfigurasi identik, tanpa validasi
target. Evaluasi setiap checkpoint terakhir saja. RX 6600 LE untuk seluruh
fit; audit CPU semua checkpoint, ID, normalisasi dan keputusan. Tidak
menambah label: 432 label sumber pengembangan dan 360 label target per fit.

Laporkan seluruh 12 fit pengembangan jika tahap ini selesai, termasuk
kegagalan. Ini pencarian arsitektur eksploratif berbasis validasi sumber,
bukan klaim model terbaik. Uji ponsel pernah dilihat pada tahap lama; tanpa
rekaman/sesi baru, hasilnya tidak menetapkan generalisasi populasi atau SOTA.
