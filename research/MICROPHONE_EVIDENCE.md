# Apa yang sudah diketahui tentang pergantian mikrofon?

Pemeriksaan sumber primer, 15 November 2026. **HEAR lengkap mendokumentasikan
ketukan yang direkam serentak lewat laptop dan ponsel, tetapi kanal ponsel
belum ada di rilis publik yang diperiksa.** Dari sumber yang ditinjau, belum
ada tabel yang sekaligus membandingkan mikrofon fisik yang cocok dan berbeda,
menahan hari uji, serta menunjukkan akurasi untuk beberapa jumlah contoh
berlabel per tombol pada target.

Untuk menilai klaim tersebut, kajian ini memisahkan keberadaan audio
berpasangan dari rancangan evaluasi. Rekam satu kejadian melalui dua perangkat
dan kita dapat membandingkan apa yang sampai di dua jalur. Untuk mengukur
kebutuhan kalibrasi, kita juga perlu sesi target yang terpisah dari uji,
jumlah **ketukan unik** yang dipakai untuk kalibrasi dan validasi, serta akurasi
per kejadian pada kedua kondisi. Dipakainya perangkat yang belum muncul saat
pelatihan awal tidak membuktikan hari kalibrasi target dan hari uji berbeda.

| Sumber | Perubahan yang benar-benar didokumentasikan | Apa yang bisa disimpulkan untuk riset ini |
|---|---|---|
| [Harrison dkk., 2023, §4 dan Tabel 4–5](https://arxiv.org/html/2308.01074) | Keyboard MacBook yang sama; ponsel dan mikrofon laptop melalui Zoom. Akurasi 95% dan 93% berasal dari model yang dievaluasi pada kanalnya masing-masing. | Selisih dua skor itu tidak mengukur penurunan saat satu model berpindah mikrofon. [Kurva lokal Zoom→ponsel](RESNET_FEWSHOT_OBSERVATIONS.md) memang menguji perpindahan jalur, tetapi menggunakan kumpulan WAV yang sama dan perubahan jalur mencakup pemrosesan Zoom. |
| [Maurya dkk., DECKER, §4.6, §4.8 dan §6](https://arxiv.org/html/2605.03384v1); [tesis Maurya v1, §5.6 dan §6.2](https://arxiv.org/html/2611.13922v1) | HEAR lengkap memakai laptop, ponsel, dan VoIP. Tesis menyebut tiga tampilan dari ketukan yang sama. Hasil utama DECKER membandingkan keyboard yang pernah dan belum pernah dilihat model. | Pasangan laptop–ponsel berpotensi memisahkan jalur rekam dari variasi antarketukan. Belum ada kurva kebutuhan label dengan hari target ditahan pada kedua jalur. [Rilis HEAR v1.0](https://github.com/vikrant-vikram/Hear-Dataset/releases) hanya memuat jalur laptop. |
| [Choudhury dkk., KEYAC v5, §3–4](https://arxiv.org/html/2606.21210v5) | Menyebut rekaman laptop, ponsel dan VoIP. Tabel membedakan pengujian keyboard baru dan jalur VoIP; tidak memberikan perbandingan laptop–ponsel berpasangan per jumlah label. | Bukti pembanding untuk perubahan kanal, dengan akses data atas permintaan menurut naskah. Tesis mengaitkan hasil yang sama dengan HEAR; jangan hitung sebagai replikasi pada kohort kedua. |
| [Vranješ dkk., 2026, §3.5–3.7 dan Gambar 8](https://www.mdpi.com/2079-9292/15/16/3509) | Melaporkan sekitar 85% top-1 pada 13 contoh per tombol. Peserta dan keyboard target berbeda dari sumber pralatih. Data uji dipertahankan sama di beberapa jumlah contoh. | Ada pembanding penelitian untuk adaptasi dengan sedikit contoh. Gambar 8 tidak mengisolasi pergantian mikrofon. Keterangan sesi yang independen dari prapelatihan belum menetapkan bahwa kalibrasi dan uji **target** diambil pada hari berbeda. |
| [He dkk., HERO, abstrak penerbit](https://doi.org/10.1016/j.knosys.2026.116689) | Abstrak melaporkan pengujian variasi jenis mikrofon, jarak, posisi, dan kebisingan. | Kandidat terdekat untuk bukti langsung. Dari abstrak yang dapat diperiksa belum bisa ditentukan apakah rute kalibrasi diuji silang pada ketukan yang sama, apakah hari uji ditahan, atau berapa label per kelas dipakai. Ini status verifikasi, bukan pernyataan bahwa naskah lengkap tidak mengujinya. |
| [Tietz dkk., 2019, §III dan §V](https://hpi.de/fileadmin/user_upload/fachgebiete/meinel/teaching/ws2020/keyboard_acoustics_CSNet.pdf) | Dari 26 peserta dengan data layak, 18 menyediakan audio laptop dan ponsel pada studi yang sama. Evaluasi memperlakukan audio laptop dan ponsel sebagai dua dataset. | Contoh pengumpulan dua jalur fisik. Hasilnya adalah galat autentikasi pengguna, tidak mengukur top-1 huruf atau label per tombol. Tidak ditemukan rilis rekaman beserta metadata pemisahan hari pada sumber resmi yang diperiksa. |
| [Rzemieniuk dkk., §2.1](https://www.mdpi.com/2076-3417/16/2/563) dan [audit lokal NC1](../reports/public-transfer-evidence/laporan.md) | Semua rekaman memakai Novox NC1; audit lokal membandingkan pemrosesan codec offline pada ketukan yang sama dan tanggal kalibrasi/uji yang terpisah. | Bukti paling terkontrol di proyek ini untuk efek pemrosesan kanal. Selisih +31,90 dan +39,22 poin di K10 berlaku untuk dua arah di satu kelompok uji; belum ada pembanding mikrofon kedua. |

[Audit pembagian Harrison](HARRISON_PROVENANCE.md) mengonfirmasi bahwa
seluruh 36 WAV induk di sisi kalibrasi ponsel juga muncul di sisi uji.
Arsip dan penanda relatifnya belum menetapkan identitas fisik ketukan
yang sama antar kanal.

Pada HERO, angka **91,48% pada 18 peserta** dari
[abstrak jurnal](https://www.sciencedirect.com/science/article/pii/S0950705126014152)
merujuk pada uji bergiliran menahan satu peserta. Abstrak
[pracetak SSRN](https://doi.org/10.2139/ssrn.5943496) melaporkan 90,5%
pada 10 peserta untuk uji lintas peserta. Jangan tafsirkan salah satunya
sebagai hasil mengganti mikrofon, atau bandingkan selisih dua angka itu
tanpa memeriksa perubahan kelompok uji. Berkas pracetak tidak dapat
dibuka dari lingkungan ini; pemeriksaan terbatas pada abstrak terindeks
dan cuplikan yang ditampilkan penerbit. Tabel yang memisahkan efek jenis
mikrofon, hari uji, dan anggaran kalibrasi belum dapat diverifikasi.

## HEAR dan KEYAC: satu sumber bukti untuk hasil VoIP ini

[Tesis Maurya, §6.2–6.3.3](https://arxiv.org/html/2611.13922v1)
menyebut 12.480 ketukan pada subset VoIP HEAR dan menunjuk karya Choudhury
dkk. sebagai publikasi eksperimen representasi audionya. Pemeriksaan
Tabel 6.1 tesis terhadap Tabel 1 KEYAC menemukan **216 dari 216 angka sama**,
baik pada [v1 KEYAC](https://arxiv.org/html/2606.21210v1) maupun pada
[v5](https://arxiv.org/html/2606.21210v5). Pencocokan berdasarkan 18 baris
model dan 12 kolom metrik dengan kondisi yang bersesuaian. Ini bukti
dokumenter tentang hasil yang sama; identitas ketukan belum diverifikasi
dari arsip pasangan. Jangan jumlahkan peserta atau ukuran sampel dua kali.

Ada perbedaan deskripsi yang berpengaruh: §3.4 v1 menyebut latihan pada
rekaman standar untuk pengujian keyboard baru melalui VoIP; §3.4 v5
menyebut latihan dan uji sama-sama memakai VoIP untuk kolom tersebut.
Angka tabel tetap identik. Untuk merujuk hasil, pakai deskripsi v5 dan
sebutkan edisinya. Pemisahan aktual per rekaman masih perlu manifest
untuk diverifikasi. Kedua deskripsi tetap tidak memberi efek mikrofon
fisik murni atau kurva biaya label per hari yang kita cari.

[Daftar rilis resmi](https://github.com/vikrant-vikram/Hear-Dataset/releases)
masih menandai v1.0 sebagai rilis terbaru saat diperiksa. [Audit arsip lokal](PUBLIC_RECORDING_OBSERVATIONS.md)
menemukan 36 ID peserta dan tidak ada ID dengan lebih dari satu token
perangkat. Token perangkat adalah metadata pada rilis; token itu sendiri
tidak mengidentifikasi dua transduser atau rute audio berpasangan. Tidak ada
pasangan laptop–ponsel yang bisa diuji dari rilis tersebut.

## Apakah merilis kanal ponsel saja cukup?

**Untuk 26 huruf, belum cukup apabila kanal baru hanya menambah rekaman
ponsel dari ketukan yang sama.** Pada tier paling longgar, yaitu klip
ekspor yang cocok dengan log, kanal laptop HEAR v1 memiliki **nol pasangan**
dengan satu contoh kalibrasi dan lima contoh uji di setiap huruf. Jumlah
ketukan yang tersedia pada kedua kanal tidak dapat melebihi jumlah pada
kanal laptop. Rilis yang juga menambah ketukan atau sesi perlu diperiksa
ulang.

Kami menghitung batas atas untuk jadwal satu kalibrasi dan dua uji memakai
daftar pasangan dari audit. Untuk setiap sesi kalibrasi, dua sesi uji harus
sama-sama memiliki lima klip per huruf, serta berurutan menurut metadata.
Untuk 12 huruf di K10, hasilnya **17 susunan sesi dari dua ID** pada tier
terverifikasi. Tujuh susunan punya jarak mulai minimal 24 jam di kedua
selang berurutan. Ini hitungan waktu berlalu dan batas atas, belum verifikasi
tanggal lokal, ketersediaan kanal ponsel, atau pengukuran akurasi. Beberapa
susunan memakai sesi yang sama. [Hitungan semua K dan tier](results/hear-session-coverage-2026-11-15.json).

## Bukti yang akan mengubah keputusan

Hasil paling berguna berikutnya adalah kurva dari ketukan uji independen
yang dapat dipasangkan antar mikrofon. Untuk membuktikan data eksternal
layak, perlu asal sesi/hari, pemetaan ketukan antara kanal laptop dan
ponsel, dan cakupan label yang cukup di hari kalibrasi **serta** hari uji.
Untuk menafsirkan hasil, pisahkan biaya label sumber, kalibrasi/validasi
target, serta evaluasi. Jika perangkat juga berbeda pada posisi atau
pemrosesan audio, kesimpulannya berlaku untuk keseluruhan jalur rekam.

Sampai bukti itu tersedia, hasil publik tersebut memberi konteks risiko
kebocoran dan nilai pembanding, tetapi tidak menentukan kebutuhan label
untuk mikrofon baru pada [tujuan riset](RESEARCH_GOAL.md). Pemeriksaan ini
memakai dokumen dan catatan evaluasi yang sudah ada. Tidak ada pelatihan
baru dan tidak ada klaim bahwa literatur secara keseluruhan belum pernah
melakukan pengujian sejenis.

[Catatan pemeriksaan tabel dan sumber](results/microphone-evidence-2026-11-15.json).

Pemeriksaan ulang pada 15 November masih menemukan HEAR v1.0 sebagai
[rilis publik terbaru](https://github.com/vikrant-vikram/Hear-Dataset/releases/latest).
Di workspace ada enam manifest sesi contoh sintetis serta nol manifest
studi dua mikrofon. Karena itu, tahap pengukuran efek mikrofon tetap
menunggu data baru atau hasil agregat yang melaporkan hitungan benar dan
total uji per hari, jalur rekam, serta anggaran label. Mengulang hitungan
atas korpus saat ini tidak menambah perbandingan fisik itu.

## Pemeriksaan arsip pada 21 November 2026

[Rilis HEAR terbaru](https://github.com/vikrant-vikram/Hear-Dataset/releases/tag/v1)
masih v1.0 dan memuat rekaman perangkat tempat peserta mengetik. Kanal ponsel
dan VoIP disebut sebagai rilis berikutnya. Publikasi
[HetKey](https://ieeexplore.ieee.org/abstract/document/11761744) membandingkan
dua keyboard mekanis dengan switch merah dan biru. Variabel itu berbeda dari
pergantian mikrofon pada keyboard yang sama.

Arsip [Multi-Pressure](https://zenodo.org/records/19453177) menyediakan 114
WAV mono 48 kHz. Pemeriksaan isi versi v8 menemukan 38 tombol dengan satu WAV
per tombol pada masing-masing kondisi tekanan. Ketersediaan satu klip dalam
tiap sel tombol–tekanan membatasi pemisahan dukungan dan query pada kondisi
yang sama. Deskripsi eksperimen menjelaskan variasi tekanan dengan posisi
mikrofon tetap. [Artikel dataset](https://www.mdpi.com/2306-5729/11/7/168)

Sumber ini memberikan informasi tambahan tentang perubahan keyboard dan
tekanan. Perbandingan dua mikrofon dengan ketukan yang sama dan sesi target
terpisah masih membutuhkan data berpasangan.
