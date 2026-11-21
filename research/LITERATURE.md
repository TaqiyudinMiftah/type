# Kajian awal dan pilihan reproduksi

Tanggal pemeriksaan sumber: 7 November 2026. Kajian terarah ini mendasari
eksperimen pada dataset eksternal sebelum pengumpulan data pengguna.

Pembaruan 15 November: [rilis HEAR v1.0](https://github.com/vikrant-vikram/Hear-Dataset/releases/tag/v1)
menyediakan arsip on-device di GitHub Releases. Kanal ponsel dan VoIP
yang dibahas pada paper tidak ada di rilis tersebut. Lihat
[audit HEAR dan Novox NC1](PUBLIC_RECORDING_OBSERVATIONS.md) untuk
bukti langsung dari berkas, termasuk jumlah sesi yang memenuhi syarat.

[Kajian khusus mikrofon](MICROPHONE_EVIDENCE.md) memeriksa pemisahan sesi,
anggaran label, dan asal data. HEAR lengkap dan hasil VoIP KEYAC tidak
dihitung sebagai dua kohort independen: tesis Maurya mengaitkan keduanya
dan melaporkan tabel hasil yang sama.

## Penelitian terdahulu

| Penelitian | Pendekatan dan hasil yang relevan | Implikasi eksperimen |
|---|---|---|
| [Asonov & Agrawal, 2004](https://research.ibm.com/publications/keyboard-acoustic-emanations) | Jaringan saraf mempelajari perbedaan bunyi tombol pada beberapa perangkat. | Menetapkan kelayakan klasifikasi bunyi; kondisi perangkat perlu dicatat. |
| [Zhuang, Zhou & Tygar, 2009](https://people.eecs.berkeley.edu/~tygar/papers/Keyboard_Acoustic_Emanations_Revisited/TISSEC.pdf) | Pemodelan akustik dipadukan dengan HMM dan statistik bahasa; pemulihan karakter mencapai 96% dalam eksperimen mereka. | Hasil yang dibantu bahasa perlu dipisahkan dari pengenalan tombol akustik. |
| [Harrison, Toreini & Mehrnezhad, 2023](https://arxiv.org/abs/2308.01074) | CoAtNet, mel-spectrogram, dan augmentasi; top-1 95% pada ponsel dan 93% pada Zoom. | Dipilih untuk percobaan reproduksi karena rekaman asli tersedia. |
| [Taheritajar & Rahaeimehr, 2024](https://arxiv.org/abs/2403.08740) | Statistik interval antartombol dan kamus, dengan 20 pengguna; keberhasilan kata rata-rata sekitar 43%. | Mengukur kata dan pola waktu; metriknya berbeda dari top-1 tombol terisolasi. |
| [Fürst & Aßmuth, 2025](https://arxiv.org/html/2503.16719) | Clustering tanpa label, cross-correlation, dan pencocokan kamus untuk passphrase. | Menunjukkan pentingnya gaya mengetik, pemisahan tekan/lepas, serta unit evaluasi. |
| [Park dkk., 2025 — ditarik](https://arxiv.org/abs/2502.09782) | Versi publik ditarik karena kesalahan prapemrosesan yang memengaruhi validitas hasil. | Klaim peningkatan akurasinya tidak digunakan sebagai benchmark. |
| [Maurya dkk., DECKER, 2026](https://arxiv.org/html/2605.03384) | HEAR mencakup 53 peserta, 37 keyboard laptop, dan tiga kanal; normalisasi karakteristik keyboard serta pembelajaran domain digunakan untuk transfer. | Split lintas pengguna/keyboard lebih bermakna daripada hanya split acak dalam satu rekaman. Paper menyebut subset publik, tetapi repositori yang diperiksa baru memuat README. |
| [Choudhury dkk., KEYAC, 2026](https://arxiv.org/html/2606.21210) | Menguji representasi audio pralatih dan adapter KAN pada perubahan perangkat/codec VoIP. Dataset tersedia atas permintaan. | Pergantian codec perlu diuji terpisah; model pralatih tidak otomatis tahan perubahan domain. |
| [Okada dkk., 2026, v3](https://arxiv.org/html/2607.22094) | Clustering, BERT karakter, koreksi LLM, dan self-training; rerata skor Levenshtein 0,941 pada 150 ketukan untuk 12 kombinasi perangkat–teks. | Skor rekonstruksi teks berbantuan bahasa berbeda dari top-1 akustik; gunakan versi terbaru saat membaca klaim. |
| [Quattrone & Badr, EchoStrike, 2026](https://pure.psu.edu/en/publications/from-keystrokes-to-semantics-acoustic-side-channel-attacks-on-typ/) | SKAID dengan uji lintas peserta: akurasi tombol 76,34%; juga melaporkan kemiripan semantik teks. | Akurasi karakter dan kemiripan semantik harus dilaporkan terpisah. |
| [Hama Rawf, 2026](https://link.springer.com/article/10.1007/s10207-025-01194-x) | Improved CoAtNet pada MKA melaporkan 99,8% akurasi dalam abstrak penerbit; terbit 12 Januari 2026. | Dataset dan protokol berbeda dari Harrison. Teks lengkap berbayar belum diperiksa; angka ini tidak menetapkan peringkat head-to-head untuk pilot. |
| [Vranješ dkk., 2026](https://www.mdpi.com/2079-9292/15/16/3509) | Pelabelan OCR–audio dan CNN; transfer ke pengguna/perangkat berbeda sekitar 85% top-1 dengan 13 contoh per tombol. Angka 98,1% adalah model dengan data besar, bukan hasil 13-shot. | Few-shot dan pembanding scratch sudah diteliti; kontribusi baru membutuhkan kontrol anggaran label dan pengujian independen yang lebih kuat. |
| [Chen dkk., BEATs, ICML 2023](https://proceedings.mlr.press/v202/chen23ag.html) | Encoder audio umum memakai target token akustik diskret dan pretraining iteratif. [Checkpoint dan kode resmi](https://github.com/microsoft/unilm/tree/master/beats) tersedia. | Pembanding representasi pralatih yang dapat diuji melalui encoder beku; keberhasilan di benchmark audio umum tidak menjamin pengenalan tombol. |

Angka di atas merupakan hasil masing-masing penulis. Akurasi tombol, persentase
karakter yang dipulihkan, dan keberhasilan kata harus dibaca sesuai definisinya.

## Sintesis dan pertanyaan riset

Literatur awal menunjukkan bahwa ciri akustik membawa informasi tentang tombol.
Penelitian setelahnya menambahkan bahasa, clustering, model dalam, dan transfer
domain. Namun, peningkatan angka sering diikuti perubahan tugas: tombol
terisolasi, kata dalam kamus, teks yang diperbaiki model bahasa, atau kemiripan
semantik. Perbandingan langsung membutuhkan unit evaluasi dan asumsi yang sama.

Tahap ini menjawab tiga pertanyaan terbatas:

1. Seberapa dekat implementasi independen dengan parameter publik Harrison
   dapat mendekati hasil ponsel/Zoom yang dilaporkan?
2. Seberapa kuat baseline MFCC–SVM ketika data, split, dan pemilihan melalui
   validasi dibuat eksplisit?
3. Bagaimana skor berubah saat ketukan uji berasal dari akhir rekaman atau
   kanal perekaman berbeda?

Harrison dipilih berdasarkan ketersediaan audio langsung, lisensi jelas, tugas
36 kelas yang terukur, dan ukuran eksperimen yang dapat dijalankan lokal pada
RX 6600 LE. Pemilihan ini tidak menyatakan Harrison sebagai metode terbaik
untuk semua kondisi. Kebutuhan generalisasi dalam DECKER/KEYAC dan evaluasi
lintas peserta dalam EchoStrike menjadi dasar tahap dataset selanjutnya.

Tahap akustik ini tidak menambahkan koreksi bahasa. Jika kelak koreksi bahasa
diuji, metrik sebelum dan sesudah koreksi harus dipisahkan; kemiripan makna
tidak menggantikan ketepatan karakter.

Sebagian benchmark rekonstruksi memakai teks serta statistik bahasa Inggris.
Kinerja pada teks Indonesia perlu dinilai tersendiri jika koreksi bahasa kelak
ditambahkan; klasifikasi tombol pada tahap ini tidak memakai asumsi kosakata.

## Kandidat dataset

| Sumber | Isi dan ketersediaan | Penggunaan |
|---|---|---|
| [Harrison 2023](https://github.com/JBFH-Dev/Keystroke-Datasets) | Dua kanal rekaman, 36 tombol alfanumerik; repositori berlisensi MIT. | Dataset utama percobaan ini. |
| [MKA 2024](https://data.mendeley.com/datasets/bpt2hvf8n3/3) | Beberapa keyboard/platform, audio mentah, segmen, dan fitur; CC BY 4.0. | Kandidat berikutnya untuk variasi perangkat. |
| [KAD](https://github.com/spatam/kad) | Keyboard mekanis; audio publik dan sitasi paper tersedia; kode penuh memerlukan permintaan kepada penulis. | Kandidat perluasan; jumlah segmen perlu diaudit sebelum digunakan. |
| [SKAID 2025](https://zenodo.org/records/17282184) | Rekaman pengetikan alami dan log tombol tersinkronisasi. | Kandidat tahap berikutnya untuk penilaian pada teks alami. |
| [HEAR 2026](https://github.com/vikrant-vikram/Hear-Dataset) | Paper menyebut sebagian data tersedia, tetapi repositori saat pemeriksaan hanya berisi README tanpa audio/tautan unduhan. Data lengkap perlu permintaan akademik. | Belum dapat direproduksi dari repositori yang ditautkan; ketersediaan audio perlu dikonfirmasi kelak. |

## Target reproduksi Harrison 2023

Penulis menyediakan audio dan fungsi isolasi, sedangkan kode pelatihan lengkap
tersedia atas permintaan. Referensi [36] paper menunjuk implementasi CoAtNet
[External-Attention-pytorch](https://github.com/xmu-xiaoma666/External-Attention-pytorch).
Percobaan menggunakan backbone itu dengan pooling dan head 36 kelas.

Parameter acuan: 1.100 epoch, batch 16, Adam, learning rate awal 0,0005 dengan
penurunan linear, 64 mel band, FFT 1.024, timeshift 40%, dan dua mask per sumbu
dengan lebar maksimum 10%. Tabel 3 mencantumkan hop 225, sedangkan uraian
menyebut 500 dan 255 bergantung data; percobaan memilih nilai tabel dan mencatat
penyesuaian bentuk fitur menjadi 64×64.

Paper membedakan split `Random` dari `Stratified`, dan hasil akhirnya memakai
`Random`. Percobaan ini memakai **acak terstratifikasi**, sehingga setiap kelas
memiliki 18/2/5 contoh. Ini perbedaan protokol yang disengaja untuk kontrol
seimbang, bukan salinan literal metode split final paper. Keberhasilan mendekati
angka headline tidak membuktikan kesetaraan seluruh konfigurasi penulis.

Rasio validasi, seed, normalisasi, kanal input, dan modifikasi backbone lengkap
tidak seluruhnya ditentukan dalam artefak publik. Asumsi implementasi disimpan
dalam konfigurasi hasil. Ini percobaan reproduksi independen, dengan kesetaraan
implementasi yang harus diperiksa saat menafsirkan selisih hasil.

Audit saat persiapan menemukan bahwa ambang adaptif asli tidak konvergen untuk
`MBPWavs/J.wav`. Langkah yang mengecil secara geometris membatasi kenaikan
ambang dari 0,06 hingga mendekati 0,56. Pada file ini masih terdapat 26 transien
terpisah pada ambang 0,56. Ekstensi deterministik mencari 4.096 ambang pada
rentang energi dan menemukan 25 segmen pada 4,060634. Ini **modifikasi isolasi**,
bukan bukti semua transien tersebut benar-benar tombol `J`. Manifest menandai
satu file ini sebagai `extended-grid-4096-count-assisted`; 71 file lainnya
menggunakan pencarian adaptif asli. Tanpa log tombol asli, ketidakpastian label
tetap ada. Pencarian hanya memakai audio dan jumlah yang didokumentasikan,
tanpa hasil klasifikasi atau pemilihan timestamp secara manual.

Ponsel tetap pada 44.100 Hz, Zoom pada 32.000 Hz sesuai berkas asli. Potongan
14.400 sampel berarti durasi berbeda: sekitar 327 ms dan 450 ms. Fitur mel juga
memakai batas frekuensi native yang berbeda. Uji perpindahan kanal karenanya
mencakup perbedaan perangkat, akuisisi, dan frontend; hasilnya belum mengisolasi
pengaruh codec saja. Kontrol dengan grid frekuensi dan durasi yang disamakan
merupakan eksperimen lanjutan, bukan penyesuaian setelah melihat skor uji ini.

Kode sumber isolator diturunkan dari README dataset berlisensi MIT; pemberitahuan
lisensinya dipertahankan di [licenses/harrison-MIT.txt](licenses/harrison-MIT.txt).
Backbone rujukan, dependensinya, lisensi, commit, dan hash tersimpan di
`research/vendor/external_attention/`.

## Protokol dan observasi yang akan dicatat

1. Audit file asli, hash, sample rate, kelas, jumlah segmen, dan lisensi.
2. Jalankan isolasi berdasarkan kode publik penulis. Jumlah 25 ketukan diketahui
   dari protokol; kecocokan jumlah bukan pengukuran precision/recall deteksi.
3. Pisahkan segmen asli sebelum augmentasi: per kelas 18 latihan, dua validasi,
   dan lima uji. Simpan identitas seluruh segmen dan seed.
4. Jalankan MFCC–SVM sebagai kontrol, lalu CoAtNet rujukan. Pilih model hanya
   berdasarkan validasi; buka metrik uji setelah pelatihan selesai.
5. Bandingkan split acak dengan split berurutan dalam rekaman, serta uji
   perpindahan kanal. Rekaman per tombol tidak menyediakan sesi independen;
   hasil dalam rekaman tidak mengukur generalisasi ke sesi/keyboard baru.
6. Laporkan top-1, top-5, macro-F1, confusion matrix, interval Wilson untuk
   akurasi, kurva validasi, durasi, hardware, dan kegagalan numerik bila ada.

Isolasi dilakukan pada rekaman lengkap sebelum split menggunakan jumlah ketukan
yang diketahui. Karena satu tombol hanya memiliki satu file per kanal,
identitas tombol berimpit dengan identitas rekaman. Uji acak ini tidak dapat
memisahkan pembelajaran bunyi tombol dari kemungkinan ciri khas file/lingkungan.
Audit juga memeriksa duplikasi segmen serta tumpang tindih jendela; pemeriksaan
tersebut tetap tidak menggantikan uji pada sesi yang direkam terpisah.

Data eksternal memiliki 36 kelas. Data pengguna `a–z` dan pembagian enam sesi
tetap menjadi eksperimen terpisah pada tahap berikutnya.

## Pilot efisiensi label dan posisi kebaruan

[Protokol pilot](FEWSHOT_PROTOCOL.md) ditetapkan sebelum mengevaluasi kurva
K=1/2/3/5/10. CoAtNet kecil dari repositori yang dikutip Harrison dibandingkan
dengan scratch pada dukungan yang sama dan BEATs iter3+ AS2M dengan encoder beku.
Backbone CoAtNet yang dipakai berparameter sekitar 1,97 juta; nama keluarga
arsitektur tidak menjamin kesetaraan dengan CoAtNet-0/1/3 dari implementasi lain.
BEATs berparameter sekitar 90,31 juta; ia pembanding pretrained yang tersedia,
bukan klaim model keystroke terbaik atau encoder audio terbaru.

Menurut [Vranješ dkk.](https://www.mdpi.com/2079-9292/15/16/3509), uji transfer
13-shot memakai pengguna/keyboard berbeda dan juga dibandingkan dengan CNN
scratch. Maka gagasan "sedikit contoh untuk adaptasi" sendiri bukan kebaruan.
Nilai tahap ini ialah memastikan biaya label dihitung menyeluruh, tidak memakai
validasi target tersembunyi, mengulang pilihan dukungan, dan memeriksa apakah
pretraining membantu dibanding scratch. Harrison tetap pilot dalam rekaman yang
sama; konfirmasi pada sesi independen masih diperlukan. Hasil dari dua paper
tersebut dan pilot ini tidak boleh diurutkan hanya berdasarkan persen akurasi.

## Dasar perbaikan optimisasi, 8 November 2026

[Xiong dkk., ICML 2020](https://arxiv.org/abs/2002.04745) menganalisis letak
LayerNorm di Transformer. Pre-LN menempatkan normalisasi di dalam cabang
residual; analisis dan eksperimen mereka menunjukkan gradien awal yang lebih
terkendali pada model yang diteliti. Paper ini menjadi dasar rancangan kontrol
residual/pre-norm, bukan bukti hasil pada suara keyboard. Efek pada proyek ini
harus diukur dengan fit dan validasi sendiri.

[Dokumentasi PyTorch 2.14](https://docs.pytorch.org/docs/2.14/nn.init.html#torch.nn.init.xavier_uniform_)
mendefinisikan Xavier uniform menurut fan-in/fan-out. Kontrol minimal
mengganti delapan matriks q/k/v/o std=0,001 dengan Xavier gain=1; hanya bobot
proyeksi tersebut berubah dibanding varian Leaky, diverifikasi melalui tes.
Kontrol cadangan menambah residual dan pre-LayerNorm jika kontrol minimal
belum stabil. Rancangan serta aturan pemilihan ditetapkan dalam
[protokol perbaikan](REPAIR_PROTOCOL.md), memakai Zoom untuk pengembangan dan
menunda evaluasi ponsel sampai keputusan terkunci.

Backbone kecil repositori referensi dan varian perbaikan ini tetap berbeda
dari keluarga [CoAtNet asli](https://arxiv.org/abs/2106.04803). Nama CoAtNet
tidak cukup untuk menyamakan parameter, blok, pelatihan atau hasil; eksperimen
ini membenahi implementasi yang telah digunakan, bukan menetapkan SOTA baru.

Ketika tiga modifikasi hybrid belum melewati kriteria variasi validasi,
[protokol CNN](CNN_REPAIR_PROTOCOL.md) menguji backbone residual kompak.
[He dkk.](https://arxiv.org/abs/1512.03385) mendasari penambahan fungsi cabang
residual pada shortcut. Kandidat ini menggunakan empat BasicBlock kecil,
bukan arsitektur ResNet-18 standar. Pola inisialisasi konvolusi dan BN nol
di akhir cabang juga dapat diperiksa pada
[sumber resmi Torchvision](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py).
Tidak memakai bobot ImageNet atau dependensi Torchvision; implementasi lokal
memakai PyTorch yang sudah terpasang. Klaim efektivitas pada keyboard tetap
bergantung pada pengukuran, bukan pada hasil pengenalan gambar paper tersebut.

## Kandidat validasi yang lebih luas, 8 November 2026

[Repositori resmi HEAR](https://github.com/vikrant-vikram/Hear-Dataset)
memuat abstrak DECKER (ASIA CCS 2026, DOI
[10.1145/3779208.3805984](https://doi.org/10.1145/3779208.3805984)).
Menurut abstrak penulis, pendekatannya mencakup normalisasi karakteristik
keyboard, disentanglement adversarial domain, alignment contrastive lintas
keyboard, dan randomisasi gaya akustik. Evaluasinya membahas lintas keyboard,
pengguna, lingkungan dan ketikan beruntun. Hal itu relevan sebagai pembanding
arah riset generalisasi, tetapi kurva Harrison dalam WAV yang sama tidak
setara dengan rancangan evaluasi tersebut.

Halaman penerbit mengembalikan 403 saat diperiksa; catatan ini hanya merujuk
abstrak publik penulis, belum mereview detail eksperimen naskah lengkap.
Repositori HEAR yang ditemukan hanya berisi README tanpa audio/tautan unduh
data. DECKER belum direproduksi dan belum dijadikan pembanding numerik;
tidak menetapkan urutan model terbaik dari abstrak. Status kandidat data
MKA/SKAID dan audit log nyata dicatat dalam
[kelayakan validasi eksternal](EXTERNAL_VALIDATION_FEASIBILITY.md).

## Sinkronisasi SKAID dan EchoStrike, 9 November 2026

Naskah asli [EchoStrike, Bagian III.D](https://www.researchgate.net/publication/405307424_From_Keystrokes_to_Semantics_Acoustic_Side-Channel_Attacks_on_Typed_Text_Using_Deep_Neural_Networks_and_Large_Language_Models)
menjelaskan penyelarasan manual dengan satu offset sebelum segmentasi.
Arsip audio/log, README dan metadata SKAID yang diperiksa belum memberikan
tabel offset tersebut. Menyamakan creation_time M4A dengan onset pertama
tidak cukup untuk mereproduksi segmentasi paper.

[Audit lokal](SKAID_OBSERVATIONS.md) memeriksa seluruh audio dan kemudian
mencocokkan pola waktu press dengan energi tanpa nama huruf atau akurasi
classifier. Sebanyak 44/50 pasangan lolos kriteria waktu sementara,
termasuk validasi pola log bagian akhir; prosedur belum mensertifikasi
setiap onset. Drift, crop bertumpang tindih, kosakata dan split perlu
ditetapkan sebelum mengukur model. Temuan ini bukan reproduksi akurasi
EchoStrike atau pembandingan SOTA.
