# Ketik Audio

Riset lokal pengenalan tombol dari suara keyboard. Riset menggunakan
**dataset eksternal Harrison 2023, MKA dan SKAID**, kajian penelitian terdahulu, baseline
MFCC–SVM, percobaan reproduksi CoAtNet, dan pilot kalibrasi dengan CoAtNet/BEATs
serta CNN residual pada **RX 6600 LE**. Pengumpulan data sendiri dilakukan setelah hasil tahap
dataset eksternal dianalisis.

**[Tujuan riset](research/RESEARCH_GOAL.md):** ukur berapa contoh
berlabel per tombol yang diperlukan pada rekaman baru, dan bagaimana
akurasi berubah saat mikrofon atau kanal berubah. Protokol memisahkan
kalibrasi dan uji, serta biaya label sumber, target dan segmentasi.

[Uji rekaman terpisah pada HEAR dan Novox NC1](research/PUBLIC_RECORDING_OBSERVATIONS.md)
memakai ulang resep MFCC+Ridge SKAID. Pada K10, HEAR mendapat **11,48%**
pada tiga ID. NC1 Fnatic tenang lintas tanggal mencapai **44,42%**;
ketika query diproses µ-law, hasilnya 6,43% dengan kalibrasi asli dan
**45,65%** jika kanal kalibrasi cocok. Ini kontrol codec offline;
pengaruh pergantian mikrofon fisik belum diukur.

[Audit dataset MKA](research/MKA_OBSERVATIONS.md) juga sudah selesai:
manifest bersih 36 kelas tersedia, tetapi struktur satu WAV per
tombol/platform belum mendukung kalibrasi dan uji lintas rekaman pada
platform yang sama. Evaluasi terpisah
[Harrison → MKA tanpa kalibrasi](research/MKA_ZEROSHOT_OBSERVATIONS.md)
menguji seluruh 1.296 potongan unik pada tiga model sumber: rerata akurasi
per platform hanya **2,62–4,63%**, dekat peluang uniform **2,78%**.
Audit crop dan replay seluruh prediksi cocok. Backbone belum dilatih ulang pada MKA.

Kelanjutan [BEATs vs ResNet lintas folder MKA](research/MKA_LOPO_OBSERVATIONS.md)
melatih kepala linear pada lima folder dan menguji folder keenam, dengan
WAV induk terpisah. Pada anggaran penuh, BEATs **2,16%**, ResNet
**2,65% ± 0,66 pp**: belum ada transfer yang berguna. Semua 336 kepala dan
embedding diaudit ulang. Backbone tetap beku; kepala kini memakai label MKA.

[Validasi bersarang regularisasi](research/MKA_NESTED_OBSERVATIONS.md)
juga selesai: alpha dipilih hanya dari folder latih. Akurasi BEATs
2,16%→**1,77%**, rerata ResNet tetap **2,65%**. Akurasi latihan meningkat,
tetapi tuning ini belum memperbaiki transfer. Seluruh 2.160 kandidat
internal, 48 model akhir dan 48 baseline cocok pada audit CPU.

[Audit audio dan sinkronisasi SKAID](research/SKAID_OBSERVATIONS.md)
kini selesai: 53 audio didekode, 52 M4A dipasangkan menurut ID/fase,
50 pasangan memiliki jam log yang dapat diperiksa. **44/50** lolos kriteria
kecocokan waktu sementara; ada 20 ID dengan kedua fase lolos. Seluruh hasil
timing cocok pada replay CPU. Ini belum akurasi huruf. Kelangkaan huruf
dan ketukan beruntun memerlukan kosakata serta aturan segmentasi tersendiri
sebelum uji kalibrasi lintas rekaman.

Kelanjutan [koreksi drift dan audit jendela](research/SKAID_DRIFT_OBSERVATIONS.md)
menaikkan kecocokan waktu sementara menjadi **48/50 pasangan** atau
23 ID dengan kedua fase lolos. Crop 80 ms dengan filter press/release
tetangga menyisakan **17 ID** yang cukup untuk 12 huruf, K hingga 10 dan
≥5 query per kelas. Seluruh seleksi clock dan hitungan di-replay;
akurasi klasifikasi SKAID belum diukur.

[Pilot klasifikasi SKAID](research/SKAID_PILOT_OBSERVATIONS.md) kini
sudah selesai. Pada 12 huruf dan 17 ID, K10/120 label dukungan per ID
menghasilkan **52,29% MFCC**, **43,12% ResNet pralatih**, dan
**26,31% BEATs**; kontrol label acak dekat peluang 8,33%. Semua fitur
dihitung ulang di CPU dan seluruh 2.448 kepala cocok pada audit.
Rerata 80%/90% belum tercapai. Budget ini bersyarat pada crop/filter
berbasis anotasi log; metadata belum membuktikan pergantian mikrofon/sesi.

[Pemilihan normalisasi/regularisasi dari ID lain](research/SKAID_SELECTION_OBSERVATIONS.md)
meningkatkan K10 menjadi **75,78% MFCC**, **69,96% ResNet pralatih**,
dan **69,80% BEATs** pada split yang sama. Semua 408 pilihan mengecualikan
ID yang ditahan; pemilihan K10 memakai 5.749–5.807 label sumber unik di
luar 120 label target. Seluruh 44.064 kandidat dan 2.448 refit diaudit.
Mean 80% belum tercapai. Hasil ini validasi silang kohort yang sudah
diperiksa; batas timestamp/kosakata dan biaya sumber tetap berlaku.

[Uji SVM linear/RBF](research/SKAID_SVM_OBSERVATIONS.md) menghasilkan
**76,73% MFCC+RBF** pada K10, dibandingkan 75,78% Ridge. Selisih
+0,95 pp memiliki CI berpasangan −1,37 hingga +3,27 pp; belum meyakinkan.
BEATs dan ResNet pralatih justru turun. Seluruh 146.880 kandidat dan
4.896 kepala final diaudit; prediksi normal tetap sama pada fitur CPU.
Ada 65 kandidat dengan selisih skor di atas toleransi yang dijelaskan
dalam diagnosis numerik. Sebanyak 241/340 pilihan RBF normal menyentuh
batas C tertinggi, sehingga optimum SVM belum ditetapkan.

[Perluasan grid C](research/SKAID_SVM_C_OBSERVATIONS.md) kemudian
menaikkan MFCC+linear dari 65,49% ke **76,63%** pada K10; RBF berubah
dari 76,73% ke **76,83%**. Delta keduanya terhadap Ridge masih mencakup
nol pada CI berpasangan. Biaya label dan split sama, 88.128 kandidat
baru serta seluruh 4.896 refit diaudit. Satu dari 340 pilihan RBF
normal berada di batas baru C=100.000. Langkah berikut berfokus pada
uji rekaman independen, dengan MFCC+Ridge sebagai pembanding utama.

Hasil kalibrasi Harrison: [kurva adaptasi CNN Zoom→ponsel](research/RESNET_FEWSHOT_OBSERVATIONS.md).
Pada K=5 contoh target per tombol, adaptasi mencapai **83,52% ± 2,05 pp**;
pada K=10 mencapai **94,07% ± 1,67 pp**. Pretraining menguntungkan semua
15 pasangan yang diuji. Ini tiga blok gabungan inisialisasi/pilihan dukungan,
masih dalam WAV yang sama, dengan biaya sumber 720 label unik.

Kajian dan protokol ada di [research/LITERATURE.md](research/LITERATURE.md).
Hasil terukur ada di [research/OBSERVATIONS.md](research/OBSERVATIONS.md):
CoAtNet **95,56% ponsel** dan **93,33% Zoom**, masing-masing dari 180 contoh uji.
Ini percobaan reproduksi independen dengan perbedaan protokol yang dicatat.
Pilot kalibrasi: [research/FEWSHOT_OBSERVATIONS.md](research/FEWSHOT_OBSERVATIONS.md).
Pada 10 label target per tombol, BEATs + linear mencapai rerata **47,22% Zoom**
dan **52,96% ponsel**. CoAtNet pada 1000 update masih underfit; hasil tersebut
tidak menetapkan keunggulan dibanding scratch yang dilatih memadai.
Diagnostik satu draw dengan 10.000 update mencapai 54,72% pada Zoom, sementara
run ponsel kolaps ke 2,78%; temuan tersebut menjadi dasar perbaikan baseline
sebelum mengukur kurva label CNN berikutnya.
Kelanjutan 8 November: [kepala beku dan kestabilan](research/FROZEN_OBSERVATIONS.md).
Autopsi menemukan gate ReLU yang memutus gradien pada run gagal; kontrol
LeakyReLU melaporkan ketiga inisialisasi, dengan variasi akurasi yang masih besar.
Perbaikan berikutnya: [CNN residual kompak](research/REPAIR_OBSERVATIONS.md)
lulus pengembangan pada Zoom, lalu mencapai **89,07% ± 0,85 pp** pada ponsel
K=10 untuk tiga inisialisasi, dengan dukungan 100% pada semuanya. Seluruh
15 checkpoint baru cocok pada replay CPU. Hasil ini kontrol scratch pada
dukungan tetap. Pengukuran adaptasi berikutnya dilakukan terpisah; generalisasi
sesi baru belum diuji.
Perekam terminal Linux, CNN kecil, dan evaluasi enam sesi tersedia untuk tahap
dataset sendiri (`a–z`); eksperimen eksternal memakai 36 kelas `0–9, a–z`.

## Instalasi

Gunakan [uv](https://docs.astral.sh/uv/getting-started/installation/).
Python 3.12 dipilih otomatis; versi dependensi dikunci dalam `uv.lock`.

```bash
uv sync --frozen --extra dev
uv run ketik --help
uv run ketik doctor
```

Pada laptop Ubuntu/Debian, perekaman memerlukan pustaka sistem PortAudio:

```bash
sudo apt-get install libportaudio2
```

`doctor` memeriksa input mikrofon, menjalankan operasi CPU, dan menguji operasi
CNN GPU sampai backward dan langkah optimizer. Tes GPU berjalan dalam proses
terpisah dengan batas waktu 45 detik. Exit code `1` berarti perekaman atau
pelatihan belum siap; JSON tetap menjelaskan kemampuan masing-masing.
Komputer pelatihan boleh tidak mempunyai mikrofon.

```bash
uv run ketik doctor --json --output runs/pemeriksaan.json
```

Instalasi utama menggunakan PyTorch CPU. Panduan lingkungan GPU AMD ada di bawah.

## Web pengumpulan dataset sendiri

Web publik: [Ketik Audio](https://ketik-audio-riset.crisppelican16.chatgpt.site).

Versi 12 memakai tampilan gelap dua kolom. Area mengetik dan kuota ada di kiri.
Pemilih mikrofon, level suara, **Periksa mikrofon**, dan **Mulai sesi** ada di
panel **Siapkan sesi** di kanan.

Kolektor ada di `web/collector`. Protokol kuota baru memakai **27 kelas: a–z
dan Spasi**, masing-masing **25 event terpilih**, total **675 per sesi**.
Peserta memakai Chrome atau Edge desktop dan mikrofon laptop. Tiga tahap
tetap dilakukan pada hari berbeda: kalibrasi, uji pertama, dan uji kedua.

Isi nama perangkat, centang persetujuan, dan klik **Mulai sesi**. Ikuti tiga
detik hening dan delapan ketukan Spasi yang dipandu. Ketukan itu dipakai
untuk pemeriksaan suara sekaligus sebagai penanda awal. Setelah lulus, kolom
mengetik langsung aktif. Gunakan **Periksa mikrofon** setelah mencentang
persetujuan jika ingin menguji suara sebelum mulai merekam. Mikrofon berhenti
setelah penilaian dan hasilnya muncul di **Rincian pemeriksaan**.

Latihan dimulai dengan satu kalimat Indonesia yang mencakup semua kelas. Tombol
yang masih perlu ditekan tampil dalam dua baris yang menyesuaikan lebar area
mengetik. Saat kursor berpindah baris, baris berikutnya langsung muncul. Setelah
kuota lengkap, peserta menekan Spasi delapan kali untuk pemeriksaan akhir.
Audio dibatasi 128 MiB PCM sebagai pengaman penyimpanan.
Jika pengaman tercapai atau peserta berhenti lebih awal, paket menjadi parsial
dan tidak memajukan hari studi.

25 keydown pertama yang memenuhi aturan dipilih untuk setiap kelas. Label
berasal dari tombol aktual, termasuk salah ketik. Backspace tidak membatalkan
ketukan yang sudah terjadi. Repeat saat tombol ditahan, shortcut, dan IME tidak
memenuhi kuota. Semua audio, event tambahan, Shift, Backspace, dan perubahan
input tetap disimpan. Spasi pemeriksaan tidak menjadi contoh kelas Spasi.

Metode `key-signal-v3` memeriksa delapan ketukan awal terhadap latar dan
batas clipping. Metode ini menghitung RMS jendela
5 ms setelah komponen DC dikurangi hanya untuk penilaian. Median RMS hening
menjadi referensi perangkat, dengan ambang
`max(1e-6, median × 10^(6/20), persentil-95 × 1.2)`.
Ini ambang awal rekayasa untuk keterdengaran relatif, bukan hasil optimasi
akurasi model. Pantulan dan suara pelepasan dicatat melalui
`timing_ambiguous` tanpa menggagalkan keterdengaran. Analisis timing tidak
diubah dan masih dapat menyatakan `needs_review`.

Pemeriksaan suara mengambil snapshot `AudioContext.currentTime` pada handler
tombol, dalam koordinat yang sama dengan frame PCM worklet. Penilaian ini
tidak lagi bergantung pada lolosnya anchor jam performa dengan interval
maksimal 5 ms; event dan anchor asli tetap dipakai oleh audit timing.
Snapshot menilai jendela suara di sekitar handler, bukan memastikan onset
mekanik atau ketepatan timestamp event. Fallback ke interpolasi jam performa
tetap tersedia untuk event lama tanpa snapshot.

Panel **Rincian suara** menampilkan hasil tiap ketukan saat pemeriksaan. Jika
pemeriksaan gagal, alasannya muncul di panel **Siapkan sesi**.
**Unduh hasil pemeriksaan** di panel **Siapkan sesi** mengekspor JSON lokal berisi
ringkasan sinyal dan posisi waktu ketukan Spasi.

Batang hijau dan meter relatif menggunakan latar perangkat sebagai 0 dB;
level dBFS asli juga ditampilkan. Latar −46 dBFS dapat mencakup derau
mikrofon/gain dan tidak langsung berarti ruangan bising. Mikrofon mengikuti
default sistem. **Pilih mikrofon lain** di panel **Siapkan sesi** memuat daftar
melalui izin browser. Satu kolom **Nama perangkat**
menentukan judul sesi dan nama ZIP; tahap Kalibrasi/Uji tetap tampil terpisah.
Field metadata `laptop` dan `keyboard` dipertahankan untuk kompatibilitas,
sedangkan nama gabungan disimpan sebagai `setup.device_name`; informasi
keyboard terpisah yang tidak diisi ditandai tidak dirinci.

Backspace, Enter, Tab, Delete, panah, serta Shift/Ctrl/Alt kiri dan kanan
memiliki hitungan tambahan berdasarkan kode fisik, di luar kuota 675.
Hitungan `additional_keys` dihitung ulang oleh importer dari event aktual.
Repeat ditolak dari hitungan; modifier pada pintasan tetap terekam mentah.
Tab di area mengetik tidak memindahkan fokus selama perekaman. Browser/OS
tetap dapat menyerap pintasan tertentu; hindari kombinasi selama pengumpulan.

Audio asli disimpan tanpa normalisasi atau filter. Setelah pemeriksaan lulus,
audio awal, ketukan, dan profil disimpan bersama. Paket menyertakan bagian
awal ini di `audio.wav`, dengan `preflight.source` bernilai `session-sync-start`.
Importer mencocokkan posisi dan urutan ketukannya dengan event sesi. Paket
lama dengan pemeriksaan terpisah masih dapat diimpor.

Pemeriksaan timing awal/akhir tetap memakai delapan Spasi masing-masing.
Setelah Spasi dilepas, hitung mundur visual 3–2–1 menandai jeda; tunggu tulisan
“Tekan Spasi sekarang”. Ketukan terlalu cepat tetap dicatat sebagai event
mentah dan tidak menjadi penanda kalibrasi. Pemeriksaan suara lulus dan
kuota lengkap belum menjamin semua contoh akustik atau timing layak.

**Mulai sesi** membuka kalibrasi studi kuota untuk profil lama secara
otomatis. Studi kuota mendapat ID dan jadwalnya sendiri dengan ID peserta
yang sama. Rekaman lama tersedia di **Sesi saya**.

Audio dan event disimpan bertahap di IndexedDB dan diekspor sebagai ZIP lokal.
Tidak ada unggahan otomatis atau akun peserta. Simpan ZIP setiap sesi dan
kirim melalui kanal studi yang disepakati. Browser yang sama menyimpan ID
peserta; ZIP terbaru dapat memulihkan profil jika data browser terhapus.
Tanggal dan identitas setup adalah keterangan perangkat/peserta, bukan
verifikasi identitas atau jam server.

```bash
cd web/collector
npm ci
npm run build
npm run dev
```

Buka `http://127.0.0.1:4173`. Pengembangan di localhost dapat meminta mikrofon;
penerbitan ke internet memerlukan HTTPS. Rekaman dimulai setelah blok input
yang kontinu tersedia; frame awal dan sampel praperekaman yang dibuang saat
inisialisasi tercatat. Perpindahan tab, mikrofon berhenti, atau gangguan
penyimpanan mempertahankan data yang tersedia sebagai sesi terputus.

Paket `ketik-web-session/v2` berisi `audio.wav` (PCM float32 mono),
`events.jsonl`, `timing.json`, dan `session.json` dengan hash berkas. Impor dari
root proyek ke direktori baru:

```bash
uv run ketik web-import /path/ke/sesi.zip --output data/collected/sesi-pertama
```

Paket v1 tetap dapat diimpor dengan aturan aslinya. Paket v2 menambahkan
konfigurasi dan hitungan `collection`, `selected_sequences`, ringkasan
`preflight`, serta flag `quota_selected` pada event keydown. Kelas Spasi
bernama `space`, sementara event mentah mempertahankan `key: " "` dan
`code: "Space"`. Jenis latihan dicatat sebagai `sentence` atau `topup`.
Importer menghitung ulang pemilihan 25 event pertama per kelas; manipulasi
flag, jumlah kuota, pemeriksaan suara, atau klaim sesi lengkap ditolak.

Importer menghitung ulang offset/drift dan pemeriksaan timing dari waveform,
memeriksa integritas paket, lalu menulis `review.json`. Status `checked`
menyatakan konsistensi terhadap transien pemeriksaan, bukan onset mekanik
yang pasti. `needs_review` mempertahankan audio mentah dan mencatat alasannya.
Hitungan kuota di web adalah ketukan tercatat; hitungan kandidat potongan bebas
press/release tetangga dilaporkan terpisah, dibatasi event kuota terpilih pada v2.
Format web tidak dimasukkan ke
format enam sesi terminal atau dilatih otomatis.

Audit alignment robust dapat dijalankan ke direktori baru:

```bash
uv run ketik web-align /path/ke/sesi.zip --output data/collected/sesi-alignment-v2 --evaluation-role development
```

Perintah ini mempertahankan audio, event, `timing.json`, dan laporan importer
`review.json`, lalu menambahkan `alignment-v2.json`. Metode v2 memakai
konsensus penanda fit untuk menahan pengaruh puncak keliru. Penanda validasi
tidak ikut memilih model dan tidak dibuang saat residualnya besar. Sinyal
lemah, beberapa hipotesis dengan dukungan setara, atau dukungan fit yang
kurang tetap menghasilkan `needs_review`. Label `development` menyatakan
bahwa sesi telah dilihat selama pengembangan; ini bukan validasi independen.

Protokol dan amandemen fixture sintetis ada di
`research/WEB_ALIGNMENT_V2_PROTOCOL.json` dan
`research/WEB_ALIGNMENT_V2_1_PROTOCOL.json`. Reproduksi kontrol di CPU:

```bash
uv run python research/audit_web_alignment.py --protocol research/WEB_ALIGNMENT_V2_1_PROTOCOL.json --output reports/web-alignment-run-baru
```

Konsistensi penanda awal/akhir belum membuktikan timing di tengah rekaman.
Kontrol sintetis menyertakan contoh perubahan timing di tengah sesi yang
tidak terdeteksi dari kedua ujungnya; hasil ini dicatat sebagai batas metode.

Tes kolektor: `npm test` dan `npm run test:browser` dari `web/collector`.
Tes browser menggunakan perangkat audio sintetis; jalankan pilot mikrofon
nyata sebelum memakai data untuk eksperimen. Tidak ada pelatihan GPU dalam
alur pengumpulan atau impor ini.

## Uji pertama Lenovo LOQ

Rekaman 22 November 2026 berlangsung 293,2 detik tanpa clipping dan memuat
25 ketukan untuk tiap huruf a–z serta Spasi. Sebanyak 571 dari 675 ketukan
lolos seleksi potongan 80 ms, dengan 14–24 potongan per kelas. Sebanyak
65 potongan tersaring karena press atau release tetangga dan 39 karena
transien lokal lemah.

Model dari kalibrasi 21 November dievaluasi langsung pada mikrofon Realtek
yang sama dengan anggaran label target nol. Potongan menggunakan timestamp
dari log tombol. [Protokol transfer](research/LOQ_TRANSFER_PROTOCOL.md)

| Model | Benar | Akurasi top-1 | Rata-rata per kelas | Akurasi top-3 |
| --- | ---: | ---: | ---: | ---: |
| MFCC + Ridge | 82/571 | 14,36% | 14,75% | 28,37% |
| MFCC + SVM RBF | 77/571 | 13,49% | 13,54% | 31,35% |
| CNN residual log-mel | 46/571 | 8,06% | 8,52% | 16,64% |

Ridge mencapai 25,71% pada potongan uji dari rekaman kalibrasi. Pada Uji pertama,
Ridge mencapai 12,59% untuk huruf a–z saja. Peluang uniform untuk 27 kelas
adalah 3,70%, sedangkan tebakan kelas terbanyak memberi 4,20%.

Skor lintas hari bersifat eksploratif. Audit timing ketat memerlukan tiga
penanda fit konsisten di setiap ujung dan sesi ini hanya memiliki dua di
akhir. Ketujuh penanda validasi yang dapat diukur memiliki residual absolut
maksimal 11,0 ms. Pemeriksaan eksploratif memakai ambang yang telah
ditetapkan untuk audit retrospektif. Pada potongan yang sama, top-1 Ridge
adalah 12,08%, 14,36%, dan 6,83% pada pergeseran −12,5, 0, dan +12,5 ms.
[Metrik agregat](research/results/loq-test1-2026-11-22.json)

## Reproduksi dengan dataset eksternal

Unduh rekaman publik asli dan backbone rujukan pada commit yang dikunci:

```bash
mkdir -p data/external
git clone https://github.com/JBFH-Dev/Keystroke-Datasets data/external/harrison-2023
git -C data/external/harrison-2023 checkout 2ecdc09b6d1868aae50813f6a6c842bdd52dab06
uv run python research/fetch_reference.py
uv run python -m ketik_audio.harrison prepare --output data/external/harrison-prepared
.venv-rocm/bin/python -u -m ketik_audio.harrison run --prepared data/external/harrison-prepared --output runs/harrison-reproduction --device gpu --epochs 1100
uv run python research/replay_harrison.py --run runs/harrison-reproduction --prepared data/external/harrison-prepared --output reports/harrison-cases
uv run python research/report_harrison.py --run runs/harrison-reproduction --output reports/harrison --audit reports/harrison-cases/audit.json
```

Perintah pelatihan memerlukan lingkungan ROCm di bawah. `--skip-neural --device
cpu` menjalankan kontrol SVM saja. Gunakan direktori keluaran baru untuk setiap
eksperimen. Audio sumber, hash, sample rate, hasil isolasi, identitas split,
konfigurasi, checkpoint, kurva, dan metrik disimpan untuk audit.
Perintah riset dijalankan dari checkout proyek agar direktori `research/` dan
backbone vendor tersedia. Keluaran laporan: `reports/harrison/laporan.md`.
Audit checkpoint di CPU mengekspor prediksi per contoh, skor, dan WAV kesalahan
di `reports/harrison-cases/`; confusion matrix harus cocok dengan hasil asli.

Split acak: 18/2/5 ketukan per kelas untuk latihan/validasi/uji. Augmentasi dan
normalisasi dipelajari setelah split. Kontrol SVM juga menguji split berurutan
dan perpindahan kanal ponsel↔Zoom. Ini evaluasi dalam kumpulan rekaman yang sama;
dataset tidak menyediakan sesi independen untuk mengukur generalisasi lintas
sesi. Perbedaan preprocessing dan ketidakpastian isolasi dicatat dalam kajian
serta manifest; hasilnya merupakan percobaan reproduksi independen.

## Pilot kalibrasi kanal dengan sedikit label

Protokol sebelum evaluasi: [research/FEWSHOT_PROTOCOL.md](research/FEWSHOT_PROTOCOL.md).
Pilot membandingkan CoAtNet scratch, CoAtNet sumber→target, dan BEATs iter3+
AS2M dengan encoder beku dan kepala linear. Semua menerima potongan WAV asli
yang dipotong ulang menjadi mono 16 kHz, 400 ms. CoAtNet dilatih ulang untuk
frontend baru; checkpoint reproduksi sebelumnya tetap menjadi eksperimen terpisah.

Untuk tiap target, 15 event pertama per tombol menjadi kandidat kalibrasi dan
10 event terakhir menjadi uji tetap. K=1/2/3/5/10 label per tombol, tiga pilihan
dukungan nested, tanpa validasi target tambahan. K=0 menguji kepala sumber.
Setiap percobaan tetap memakai 36 kelas, sehingga K=1 berarti 36 label total.
Ini pilot eksploratif: seluruh rekaman Harrison sudah diperiksa sebelumnya.

Jalankan dari root proyek setelah menyiapkan raw Harrison dan lingkungan ROCm:

```bash
uv run python research/fetch_beats.py
uv run python -m ketik_audio.fewshot prepare --raw data/external/harrison-2023 --original-manifest data/external/harrison-prepared/manifest.json --output data/external/harrison-fewshot
.venv-rocm/bin/python -u -m ketik_audio.fewshot cache --prepared data/external/harrison-fewshot --checkpoint data/external/beats/BEATs_iter3_plus_AS2M.pt --output data/external/harrison-fewshot-features
.venv-rocm/bin/python -u -m ketik_audio.fewshot run --prepared data/external/harrison-fewshot --cache data/external/harrison-fewshot-features --output runs/fewshot-pilot
uv run python research/audit_fewshot.py --run runs/fewshot-pilot --prepared data/external/harrison-fewshot --cache data/external/harrison-fewshot-features --checkpoint data/external/beats/BEATs_iter3_plus_AS2M.pt --output reports/fewshot-audit
uv run python research/report_fewshot.py --run runs/fewshot-pilot --output reports/fewshot-pilot --audit reports/fewshot-audit/audit.json
```

`prepare`, `cache`, audit, dan laporan meminta direktori baru. Perintah `run`
dapat melanjutkan kondisi yang sudah selesai apabila hash kode, protokol, data,
dan cache sama. Kondisi yang terputus sebelum `result.json` dilatih ulang dari
awal dengan seed yang sama; tidak melanjutkan langkah optimizer yang terputus.
Progres tersimpan pada `source-*/history.json` dan keluaran terminal.

Checkpoint BEATs ~361 MB diunduh dari mirror publik pada revisi tetap karena
tautan OneDrive resmi mengembalikan 403 saat pilot. Digest SHA256 diverifikasi
terhadap objek LFS mirror, bukan digest Microsoft. Kode upstream, lisensi MIT,
implementasi Kaldi PyTorch murni berlisensi BSD-2, dan dua patch import tercatat
di `research/vendor/beats/provenance.json`. Ekstensi native TorchAudio tidak
diperlukan; fbank dibandingkan dengan rilis resmi TorchAudio v0.13.0 pada delapan
klip nyata dan menghasilkan nilai identik, tercatat di
`research/results/beats-kaldi-compatibility.json`.

CoAtNet sumber: 600 epoch per kanal. Scratch/adaptasi: tepat 1000 update per
dukungan, 60 pelatihan target total. Audit mengulang seluruh 94 predictor pada
CPU dan memeriksa identitas dukungan, anggaran label, normalisasi, dan skor.
Waktu head BEATs dilaporkan terpisah dari biaya ekstraksi fitur. Hasil pilot
tidak mengukur generalisasi ke sesi, keyboard, atau pengguna baru.

Diagnostik tambahan ditentukan setelah scratch pada K=1 terlihat belum
menguasai dukungan. [Protokol diagnostik](research/FEWSHOT_DIAGNOSTIC_PROTOCOL.md)
dan snapshot pemicu dibedakan dari hasil utama. Ia menguji scratch 10.000
update, K=10 dan seed dukungan 101 saja, pada kedua target. Evaluasi uji
dilakukan hanya pada update terakhir; tidak memilih checkpoint dari uji.

```bash
.venv-rocm/bin/python -u research/diagnose_fewshot_optimization.py --primary runs/fewshot-pilot --prepared data/external/harrison-fewshot --cache data/external/harrison-fewshot-features --output runs/fewshot-optimization-diagnostic
uv run python research/report_fewshot_diagnostic.py --primary runs/fewshot-pilot --diagnostic runs/fewshot-optimization-diagnostic --prepared data/external/harrison-fewshot --cache data/external/harrison-fewshot-features --output reports/fewshot-diagnostic
```

## Backbone beku dan pemeriksaan aktivasi

[Protokol kepala beku](research/FROZEN_HEAD_PROTOCOL.md) membandingkan
representasi CoAtNet pralatih dan acak dengan kepala Ridge yang sama seperti
BEATs. Kumpulan dukungan dan uji tetap sama. Seluruh bobot dan statistik
BatchNorm dibekukan; kontrol normalisasi membedakan statistik sumber tetap
dari statistik yang dihitung hanya dari dukungan target.

```bash
.venv-rocm/bin/python -u -m ketik_audio.frozen --primary runs/fewshot-pilot --prepared data/external/harrison-fewshot --cache data/external/harrison-fewshot-features --output runs/frozen-head
uv run python research/audit_frozen.py --run runs/frozen-head --primary runs/fewshot-pilot --prepared data/external/harrison-fewshot --cache data/external/harrison-fewshot-features --output reports/frozen-head-audit
uv run python research/autopsy_fewshot.py --primary runs/fewshot-pilot --diagnostic runs/fewshot-optimization-diagnostic --prepared data/external/harrison-fewshot --cache data/external/harrison-fewshot-features --output reports/fewshot-autopsy
uv run python research/report_frozen.py --run runs/frozen-head --primary runs/fewshot-pilot --audit reports/frozen-head-audit --autopsy reports/fewshot-autopsy --output reports/frozen-head
```

Autopsi memakai dukungan saja, tanpa optimizer step atau evaluasi uji baru.
Ia memeriksa variasi aktivasi dan gradien pada checkpoint gagal/berhasil,
inisialisasi acak dan model sumber. Jika gate ReLU nol pada semua dukungan,
intervensi LeakyReLU pada salinan memeriksa apakah aliran gradien kembali.

[Protokol kestabilan](research/STABILITY_PROTOCOL.md) menguji perubahan satu
gate dari awal pelatihan: ReLU vs LeakyReLU(0,01), seed 42/43/44, K=10 dan
10.000 update. Kontrol ReLU seed 42 dipakai ulang dari diagnostik lama;
lima fit lain dilatih baru pada RX 6600 LE. Laporkan semua seed, bukan yang
paling tinggi. Ini varian riset, bukan reproduksi literal arsitektur paper.

```bash
.venv-rocm/bin/python -u -m ketik_audio.stability --primary runs/fewshot-pilot --diagnostic runs/fewshot-optimization-diagnostic --prepared data/external/harrison-fewshot --cache data/external/harrison-fewshot-features --autopsy reports/fewshot-autopsy --output runs/stability-gate
uv run python research/report_stability.py --run runs/stability-gate --prepared data/external/harrison-fewshot --cache data/external/harrison-fewshot-features --output reports/stability-gate
```

Seluruh perintah baru meminta direktori keluaran baru dan menjaga hasil lama.
Kepala beku memakai feature cache terpisah dan diperiksa terhadap encoder CPU;
kontrol aktivasi menyimpan checkpoint, history gradien, indeks, normalisasi,
waktu, memori dan skor untuk replay CPU. Perbandingan tetap dalam rekaman
Harrison yang telah diperiksa sebelumnya; konfirmasi pada sesi independen
belum dilakukan.

## Perbaikan aliran sinyal attention

[Protokol perbaikan](research/REPAIR_PROTOCOL.md) memisahkan pemilihan
konfigurasi dari uji ponsel. Proyeksi attention referensi memakai bobot awal
std=0,001 dan tidak memiliki residual; pengukuran sebelum fit menunjukkan
variasi sinyal dan gradien yang sangat kecil. Kandidat minimal memakai
Xavier uniform pada delapan matriks attention dan LeakyReLU terakhir.
Kandidat cadangan menambah residual dan pre-LayerNorm jika kandidat minimal
gagal kriteria pengembangan yang telah ditetapkan.

```bash
.venv-rocm/bin/python -u -m ketik_audio.repair develop --output runs/repair-development
uv run python research/autopsy_repair_bn.py --output reports/repair-bn-autopsy
.venv-rocm/bin/python -u -m ketik_audio.stem_repair develop --output runs/stem-repair-development
.venv-rocm/bin/python -u -m ketik_audio.cnn_repair develop --output runs/cnn-repair-development
.venv-rocm/bin/python -u -m ketik_audio.cnn_repair confirm --development runs/cnn-repair-development --output runs/cnn-repair-confirmation
uv run python research/audit_repair.py --output reports/repair-audit
uv run python research/report_repair.py --output reports/repair --snapshot research/results/repair-2026-11-08.json
```

Pengembangan memakai Zoom saja: 360 contoh fit dan 72 validasi sumber,
tiga inisialisasi, 6.000 update. Keputusan dan hash dikunci sebelum konfirmasi
ponsel K=10, seed dukungan 101, inisialisasi 42/43/44, 10.000 update tanpa
validasi target. Konfirmasi menolak perubahan kode/data/protokol setelah
keputusan dibuat. Kode vendor dan hasil lama tetap utuh. Semua perintah
memerlukan direktori keluaran baru; jangan menimpa eksperimen tersimpan.
Perubahan inisialisasi memerlukan pelatihan ulang. Dua kandidat attention
gagal kriteria kestabilan validasi; [kontrol stem lebar](research/STEM_REPAIR_PROTOCOL.md)
juga belum lulus. [Alternatif CNN residual kompak](research/CNN_REPAIR_PROTOCOL.md)
menguji backbone yang lebih kecil dengan shortcut di setiap blok. Seluruh
kandidat dilaporkan, tanpa melonggarkan kriteria dari hasil uji ponsel. Ini
pencarian arsitektur berbasis validasi Zoom, bukan klaim CoAtNet kanonis atau SOTA.
Hasil dan batas: [research/REPAIR_OBSERVATIONS.md](research/REPAIR_OBSERVATIONS.md).

## Efisiensi label CNN residual: Zoom ke ponsel

[Protokol kurva label](research/RESNET_FEWSHOT_PROTOCOL.md) membandingkan
scratch dan adaptasi full-model pada K=1/2/3/5/10. Tiga blok memasangkan seed
pelatihan/dukungan (42,101), (43,102), (44,103); SD mencakup kedua faktor.
Setiap pasangan menerima dukungan, batch/augmentasi dan LR 5e-4 yang sama,
10.000 update, tanpa validasi target. Tiga model sumber memakai 648 train
dan 72 validasi Zoom; checkpoint dipilih melalui validasi lalu dikunci
sebelum evaluasi ponsel baru. Anggaran sumber unik 720 label, target 36K/fit.

Hasil: [observasi](research/RESNET_FEWSHOT_OBSERVATIONS.md) dan
[laporan lengkap](reports/resnet-fewshot/laporan.md).
Pada grid yang diuji, adaptasi pertama melewati 80% di K=5 (180 label target)
dan 90% di K=10 (360 label), pada seluruh blok. Scratch pertama melewati
80% di K=10, dengan rerata 87,22%; ambang rerata 90% belum tercapai.
Seluruh 33 checkpoint dan tiga evaluasi K=0 cocok pada replay CPU.

```bash
.venv-rocm/bin/python -u -m ketik_audio.resnet_fewshot --output runs/resnet-fewshot
uv run python research/audit_resnet_fewshot.py --output reports/resnet-fewshot-audit
uv run python research/report_resnet_fewshot.py --output reports/resnet-fewshot --snapshot research/results/resnet-fewshot-2026-11-08.json
```

Eksperimen memerlukan artefak baseline stabil sebelumnya pada lokasi yang
tercantum di atas. Checksum model, data, frontend dan protokol diperiksa;
tidak mengubah kode eksperimen yang sudah selesai. Pemulihan eksperimen
yang sama memakai optimizer/scheduler/RNG dan update terakhir tersimpan:

```bash
.venv-rocm/bin/python -u -m ketik_audio.resnet_fewshot --output runs/resnet-fewshot --resume
```

Kandidat validasi pada rekaman lain diperiksa dalam
[catatan kelayakan dataset eksternal](research/EXTERNAL_VALIDATION_FEASIBILITY.md).
Audit log SKAID dapat dijalankan dengan
`uv run python research/catalog_skaid.py --output reports/skaid-feasibility`.
Hitungan log ini belum merupakan evaluasi model atau verifikasi alignment audio.

Total tiga fit sumber, 30 fit target dan tiga evaluasi K=0. Semua fit memakai
RX 6600 LE; audit CPU memeriksa checkpoint, ID, statistik fit, inisialisasi,
counter update dan fingerprint batch pasangan. Kurva ini eksploratif dalam
WAV Harrison yang sama; ambang 80%/90% pada grid bukan kebutuhan label
minimum yang pasti atau hasil pada sesi/keyboard baru.

## Audit dataset eksternal MKA v3

[Protokol](research/MKA_AUDIT_PROTOCOL.md) dan
[hasil audit](research/MKA_OBSERVATIONS.md) memeriksa audio mentah,
potongan, duplikasi, metadata, serta kecukupan kelompok untuk evaluasi.
Arsip asli sekitar 2 GB disimpan tanpa ekstraksi penuh; pin SHA256
penyedia diperiksa. Semua tahap berikut memakai CPU.

```bash
uv run python research/fetch_mka.py --output data/external/mka-v3
uv run python research/audit_mka.py --output reports/mka-audit --snapshot research/results/mka-audit-2026-11-08.json
uv run python research/inspect_mka_headers.py
uv run python research/audit_mka_praat.py --output reports/mka-praat-audit --snapshot research/results/mka-praat-2026-11-08.json
uv run python research/curate_mka.py --output reports/mka-36 --snapshot research/results/mka-clean-36-2026-11-08.json
```

Unduhan yang sama dapat dilanjutkan dengan perintah fetch yang sama.
Audit parsial memakai `--resume` dengan data/kode/protokol yang sama;
audit selesai diverifikasi tanpa menimpa hasilnya. Audit/kurasi baru
memerlukan direktori dan snapshot baru. Keluaran utama:
`reports/mka-audit/laporan.md`, `audio.csv`, `excerpt-provenance.json`,
dan `reports/mka-36/manifest.json`. ID kelompok adalah fingerprint WAV,
bukan ID sesi; label berasal dari nama berkas penulis, tanpa anotasi event.
Audit pelengkap juga mendecode 120 objek teks Praat Sound pada folder fitur;
semuanya salinan PCM WAV, tanpa tambahan contoh unik.

## Transfer langsung Harrison ke MKA

Evaluasi langsung memakai protokol tersendiri yang dikunci sebelum inferensi:
[protokol zero-shot](research/MKA_ZEROSHOT_PROTOCOL.md),
[observasi](research/MKA_ZEROSHOT_OBSERVATIONS.md), dan
[laporan lengkap](reports/mka-zeroshot/laporan.md).
Checkpoint sumber Zoom seed42/43/44, normalisasi sumber, seluruh parameter
dan buffer tetap beku; tidak ada label/statistik target untuk fit. Crop utama
400 ms di sekitar puncak amplitudo dibandingkan dengan sensitivitas rata-rata
logit seluruh jendela 400 ms. Tidak memilih kebijakan/seed terbaik dari MKA.

```bash
.venv/bin/python -u -m ketik_audio.mka_zeroshot --output runs/mka-zeroshot
.venv/bin/python -u research/audit_mka_zeroshot.py --output reports/mka-zeroshot-audit
.venv/bin/python research/report_mka_zeroshot.py --output reports/mka-zeroshot --snapshot research/results/mka-zeroshot-2026-11-08.json
```

Seluruh tahap zero-shot memakai CPU. Run dengan hasil lengkap hanya dibaca
ulang jika semua checksum tetap cocok; audit/laporan memerlukan direktori
keluaran baru. MKA memberi enam potongan per WAV induk, bukan enam ketukan
independen. Evaluasi ini belum mengukur adaptasi ke sesi baru atau decoding teks.

## Representasi beku lintas folder MKA

[Protokol LOPO](research/MKA_LOPO_PROTOCOL.md) membandingkan BEATs 768 fitur
dengan tiga ResNet Harrison 128 fitur. Tiap fold menahan seluruh WAV dari
satu folder; lima folder lain memasok K=1/2/3/5/6 contoh per tombol/folder
untuk kepala Ridge alpha1. K=6 memakai 1.080 label, bukan nol label MKA.
Label/indeks dan crop peak400ms identik antar encoder. Encoder/BatchNorm
tetap beku; tidak ada validasi atau kalibrasi dari folder uji.

```bash
.venv-rocm/bin/python -u -m ketik_audio.mka_lopo cache --output data/external/mka-lopo-features
.venv/bin/python -u -m ketik_audio.mka_lopo run --cache data/external/mka-lopo-features --output runs/mka-lopo
.venv/bin/python -u research/audit_mka_lopo.py --output reports/mka-lopo-audit
.venv/bin/python research/report_mka_lopo.py --output reports/mka-lopo --snapshot research/results/mka-lopo-2026-11-08.json
```

Ekstraksi BEATs memakai RX 6600 LE; ekstraksi ResNet, kepala dan audit CPU.
Cache/audit/laporan baru meminta direktori baru; run lengkap hanya
diverifikasi ulang tanpa menimpa jika checksum sama. Hasil ada pada
[observasi](research/MKA_LOPO_OBSERVATIONS.md) dan
[laporan](reports/mka-lopo/laporan.md). Semua 312 fit normal dan 24 kontrol
pengacakan label tingkat WAV dilaporkan. Nama folder bukan ID sesi fisik;
hasil belum mengukur efisiensi kalibrasi ke sesi baru.

## Seleksi regularisasi melalui validasi bersarang

[Protokol](research/MKA_NESTED_PROTOCOL.md) memakai embedding LOPO yang
sudah diaudit dan anggaran K6. Dalam setiap outer fold, alpha dipilih
melalui lima inner fold pada lima folder latih. Grid1e-6 sampai100
ditetapkan sebelumnya; seri memilih alpha terbesar. Seluruh pilihan
dikunci sebelum prediksi luar baru, termasuk kontrol permutasi label.
Tidak menghitung outer score untuk kandidat alpha yang tidak dipilih.

```bash
.venv/bin/python -u -m ketik_audio.mka_nested select --output runs/mka-nested
.venv/bin/python -u -m ketik_audio.mka_nested evaluate --run runs/mka-nested
.venv/bin/python -u research/audit_mka_nested.py --output reports/mka-nested-audit
.venv/bin/python research/report_mka_nested.py --output reports/mka-nested --snapshot research/results/mka-nested-2026-11-08.json
```

Semua tahap memakai CPU. Seleksi, audit dan laporan memerlukan direktori
baru; evaluasi yang sudah selesai hanya diverifikasi tanpa menimpa
apabila semua checksum cocok. Hasil dan batas ada pada
[observasi](research/MKA_NESTED_OBSERVATIONS.md) serta
[laporan lengkap](reports/mka-nested/laporan.md). Ini studi eksploratif
setelah hasil LOPO dibaca, bukan final test yang belum pernah diperiksa.

## Sinkronisasi dan kelayakan segmentasi SKAID

[Audit native](research/SKAID_AUDIT_PROTOCOL.md),
[offset tunggal](research/SKAID_ALIGNMENT_PROTOCOL.md) dan
[offset+drift](research/SKAID_DRIFT_PROTOCOL.md) memakai sumber checksum
tetap dari [SKAID v1](https://zenodo.org/records/17282184). Dua fase
memiliki M4A terpisah, tetapi bukan bukti pergantian mikrofon/sesi fisik.
Audit dan koreksi waktu tidak membaca `full_text` atau melatih classifier.
Semua perintah berikut memakai CPU, ffmpeg/ffprobe, dan direktori keluaran baru.

```bash
.venv/bin/python research/fetch_skaid.py
.venv/bin/python research/audit_skaid.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/align_skaid.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/audit_skaid_alignment.py
.venv/bin/python research/report_skaid.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/run_skaid_drift.py select
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/run_skaid_drift.py evaluate
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/audit_skaid_drift.py
.venv/bin/python research/report_skaid_drift.py
```

Memerlukan katalog log awal `data/external/skaid-catalog/` seperti dalam
[catatan kelayakan](research/EXTERNAL_VALIDATION_FEASIBILITY.md).
Hasil yang sudah tersedia tidak perlu ditimpa atau diunduh ulang;
manifest mematok isi beserta provenance unduhan.

## Pilot klasifikasi SKAID dengan rekaman kalibrasi/uji terpisah

[Protokol](research/SKAID_PILOT_PROTOCOL.md) mengunci crop 80 ms,
17 ID, 12 huruf, query Phase 2 seimbang 60/ID, tiga draw nested serta
K1/2/3/5/10 dukungan Phase 1. MFCC, BEATs, tiga ResNet pralatih dan
tiga ResNet acak mendapat split yang sama. Kepala Ridge alpha 1 tetap,
tanpa target validation. Dataset dan sumber neural memerlukan artefak
SKAID/Harrison/BEATs sebelumnya pada path yang dipatok.

```bash
.venv/bin/python -m ketik_audio.skaid_pilot_data
.venv-rocm/bin/python -m ketik_audio.skaid_pilot_features
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python -m ketik_audio.skaid_pilot_probe
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python research/audit_skaid_pilot.py
.venv/bin/python research/report_skaid_pilot.py
.venv/bin/python research/review_skaid_pilot_report.py
```

Semua tahap meminta direktori hasil baru. Ekstraksi neural hanya RX 6000;
kepala dan replay semua fitur memakai CPU. Hasil pada
[laporan](reports/skaid-pilot-reviewed/laporan.md) belum mengukur alfabet
penuh, detektor tanpa timestamp, atau kebutuhan label end-to-end.

## Pemilihan resep kalibrasi dari ID sumber

[Protokol antar-ID](research/SKAID_SELECTION_PROTOCOL.md) mempertahankan
cache/split pilot. Dua normalisasi dan sembilan alpha divalidasi pada
16 ID selain ID yang ditahan. Pilihan bersama per keluarga/K dikunci
sebelum refit luar, termasuk kontrol label acak. Semua tahap memakai CPU.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m ketik_audio.skaid_selection validate
.venv/bin/python -m ketik_audio.skaid_selection select
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m ketik_audio.skaid_selection evaluate
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/audit_skaid_selection.py
.venv/bin/python research/inspect_skaid_selection_numeric.py
.venv/bin/python research/report_skaid_selection.py
```

Gunakan direktori baru dan pertahankan artefak pilot yang dipatok.
[Laporan](reports/skaid-selection/laporan.md) mencatat seluruh biaya label
sumber, hasil normal/kontrol, dan tujuh variasi prediksi ketika fitur
diganti dengan cache CPU lama; akurasi top1 kondisi normal tetap sama.

## Perbandingan kepala SVM linear dan RBF

[Protokol SVM](research/SKAID_SVM_PROTOCOL.md) mempertahankan cohort,
cache dan budget. Seleksi memakai 16 ID lain, dengan kebijakan linear/RBF
terpisah dan kontrol yang mendapat seleksi sendiri. Jalankan pada CPU:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m ketik_audio.skaid_svm validate
.venv/bin/python -m ketik_audio.skaid_svm select
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m ketik_audio.skaid_svm evaluate
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/audit_skaid_svm.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/inspect_skaid_svm_numeric.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/inspect_skaid_svm_bounds.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/report_skaid_svm.py
```

Semua tahap meminta direktori hasil baru. Artefak SKAID/Harrison/BEATs
terdahulu harus tersedia pada path yang dipatok. [Laporan](reports/skaid-svm/laporan.md)
memuat semua hasil, biaya sumber, variasi ID dan sensitivitas intercept
SVM; diagnosis tersebut tidak mengubah model terkunci.

## Pemeriksaan batas C SVM

[Protokol C](research/SKAID_SVM_C_PROTOCOL.md) memakai ulang kandidat
dan audit lama, lalu menambahkan C=1.000, 10.000 dan 100.000. Tahap CPU:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m ketik_audio.skaid_svm_c validate
.venv/bin/python -m ketik_audio.skaid_svm_c select
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m ketik_audio.skaid_svm_c evaluate
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/audit_skaid_svm_c.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/inspect_skaid_svm_c_numeric.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/report_skaid_svm_c.py
```

Jalankan dengan hasil SVM awal yang dipatok. Gunakan tujuan baru untuk
replay dan pilih path snapshot baru untuk laporan. [Laporan](reports/skaid-svm-c/laporan.md)
mencatat perbandingan semua K, profil C di sumber, biaya dan audit.

## Baseline pada rekaman publik terpisah

[Audit arsip](research/PUBLIC_RECORDING_AUDIT_PROTOCOL.md) memeriksa WAV
dan duplikasi, lalu [baseline terkunci](research/PUBLIC_RECORDING_BASELINE_PROTOCOL.md)
menguji HEAR lintas sesi dan NC1 lintas tanggal. Jalankan pada CPU:

```bash
.venv/bin/python research/fetch_hear.py
.venv/bin/python research/fetch_nc1.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/audit_public_recordings.py hear
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/audit_public_recordings.py nc1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/run_public_recording_baseline.py prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/run_public_recording_baseline.py evaluate
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python research/audit_public_recording_baseline.py
.venv/bin/python research/report_public_recording_baseline.py
```

Tahap setelah unduhan memakai direktori hasil baru. Baseline memerlukan
artefak SKAID yang dipatok. [Laporan](reports/public-recording-baseline/laporan.md)
memisahkan biaya label fit, klasifikasi lintas sesi, dan kontrol kanal.

## Pengumpulan data pada laptop

Pakai satu keyboard, pengguna, posisi laptop, dan pengaturan mikrofon yang sama.
Nonaktifkan Caps Lock. Ketik satu huruf sesuai prompt dengan jeda minimal
0,8 detik; pemakaian jari mengikuti gaya mengetik Anda. Awali tiap sesi dengan
tiga detik diam. Perekam menerima tombol hanya dari terminalnya sendiri.

```bash
uv run ketik record --data data/real --session 1
```

Jika perlu memilih mikrofon, tambahkan `--device ID` sesuai daftar dari `doctor`.
Satu sesi berisi 25 ketukan per huruf, total 650. Ulangi perintah dengan nomor
sesi **2, 3, 4, dan 5**, dipisahkan istirahat atau waktu rekaman berbeda.
Alokasi keseluruhan enam sesi sekitar 1–2 jam, bergantung kecepatan dan istirahat.

- Sesi 1–4: latihan, target 2.600 contoh.
- Sesi 5: validasi, target 650 contoh.
- Sesi 6: uji akhir, target 650 contoh; rekam setelah konfigurasi model ditetapkan.

Spasi menjeda prompt dan Enter melanjutkan. Waktu kontrol dan istirahat disimpan
sebagai interval protokol dan dikeluarkan dari analisis audio. Salah tekan
dicatat dengan huruf aktual, lalu prompt diulang. Salah tekan tetap masuk
referensi rangkaian pada evaluasi agar prediksi yang benar tidak dihitung sebagai
huruf tambahan. Potongan latihan yang tumpang tindih dengan kontrol atau ketukan
lain ditolak dan jumlahnya dilaporkan.

Ctrl+C menyimpan sesi sebagai `aborted`. Buffer audio hilang menandai sesi
`invalid`. Ulangi sesi gagal dengan:

```bash
uv run ketik record --data data/real --session 1 --retry
```

Sesi gagal dipindahkan ke `data/real/failed/`; sesi lengkap tidak ditimpa.
Untuk pilot singkat, `--per-key 2` mengurangi ukuran tiap sesi. Gunakan direktori
data terpisah untuk pilot dan eksperimen utama.

Setiap sesi menghasilkan:

```text
data/real/raw/session-01/
  audio.wav       # mono float32 pada sample rate perangkat
  events.csv      # key, expected, sample_index, accepted, reason
  session.json    # identitas, peran, perangkat, kualitas, dan interval jeda
```

Penanda tombol memakai clock stream PortAudio dan awal pengambilan sampel ADC,
bukan jam kalender. Jangan mengubah sample rate WAV tanpa mengonversi penandanya.
Salin folder proyek dan `data/real` melalui media lokal jika komputer pelatihan
berbeda dari laptop perekam. Kunci dependensi kedua perangkat dengan `uv.lock`.

## Siapkan dan latih model

```bash
uv run ketik prepare --data data/real --output data/real/processed
uv run ketik train svm --dataset data/real/processed --output runs/svm
uv run ketik train cnn --dataset data/real/processed --device cpu --output runs/cnn-cpu
```

`prepare` **hanya membaca sesi 1–5**, memilih ambang deteksi dari sesi 5, dan
menyimpan fitur serta checksum sumber. Potongan untuk klasifikasi menggunakan
penanda tombol yang diketahui. Deteksi tanpa penanda diukur terpisah saat evaluasi.

Pemrosesan: resampling 44,1 kHz, segmen 300 ms (50 ms sebelum penanda), 64 mel
band, FFT 1.024, hop 256, dan 20 MFCC. Fitur SVM adalah rata-rata dan simpangan
baku MFCC serta delta orde 1/2, sebanyak 120 nilai.

SVM membandingkan `C={1,10,100}` dan `gamma={scale,0.01}` berdasarkan macro-F1
validasi. Scaler hanya dipelajari dari sesi 1–4. CNN menggunakan kanal
16/32/64, Conv–ReLU–MaxPool, global average pooling, dropout 0,2, dan 26 keluaran.
Normalisasi per mel band juga hanya dipelajari dari sesi 1–4.

CNN memakai Adam, learning rate 0,001, batch 32, maksimal 50 epoch, early
stopping delapan epoch, dan seed 42/43/44. Bobot terbaik setiap seed dan model
pilihan berdasarkan validasi disimpan. Jangan memakai sesi 6 untuk menyetel model.
`training.json` menyimpan kurva, waktu, versi perangkat lunak, dan backend.

Direktori dataset, model, dan evaluasi yang sudah ada tidak ditimpa. Gunakan
nama baru untuk eksperimen baru, misalnya `--output runs/cnn-cpu-v2`.

## Uji akhir, prediksi, dan laporan

Setelah konfigurasi kedua model dikunci, rekam sesi 6 pada laptop:

```bash
uv run ketik record --data data/real --session 6
```

Salin sesi tersebut ke komputer pelatihan jika berbeda. Dataset fitur latihan
tidak perlu dibuat ulang.

```bash
uv run ketik evaluate --model runs/svm --data data/real
uv run ketik evaluate --model runs/cnn-cpu --data data/real
uv run ketik report --models runs/svm runs/cnn-cpu --output reports/awal
uv run ketik predict --model runs/cnn-cpu --audio data/real/raw/session-06/audio.wav --output runs/prediksi.json
```

Keluaran utama:

- `<model>/evaluation/evaluation.json`: metrik, provenance, waktu, dan statistik seed.
- `<model>/evaluation/classification.csv`: label dan prediksi tiap potongan bersih.
- `<model>/evaluation/detections.csv`: hasil deteksi dan klasifikasi dari audio saja.
- `reports/awal/laporan.md`, `results.json`, dan grafik confusion matrix/kurva latih.

Jika memakai keluaran evaluasi khusus, berikan direktori sesuai urutan model:

```bash
uv run ketik report --models runs/svm runs/cnn-cpu --evaluations runs/uji-svm runs/uji-cnn --output reports/uji-khusus
```

Metrik utama adalah akurasi, macro-F1 seluruh 26 kelas, F1 deteksi dengan
toleransi 75 ms, dan CER. CER mencakup huruf salah, hilang, dan tambahan; nilainya
bisa melebihi 1. Prediksi menerima WAV tanpa label, dengan jeda ketukan terpandu.
Untuk WAV bernama `audio.wav` hasil perekam, metadata `session.json` digunakan
hanya untuk interval kontrol; `events.csv` tidak dibaca oleh `predict`.
WAV mandiri sebaiknya hanya berisi huruf a–z tanpa tombol kontrol.

Target eksploratif macro-F1 adalah 0,70 pada sesi uji. Laporan tetap menyajikan
hasil ketika target belum tercapai. Hasil v1 mengukur satu pengguna/keyboard
dengan ketukan terpisah. Pengujian kalimat alami, tombol Shift/spasi/Backspace,
lintas pengguna, keyboard, dan kebisingan merupakan eksperimen lanjutan.
Skor keluaran model belum dikalibrasi sebagai probabilitas.

## GPU AMD lokal

PyTorch CPU dan ROCm menggunakan versi dasar 2.14.0 yang sama. Pasang build ROCm
di lingkungan terpisah agar `uv sync` tidak menggantinya dengan build CPU.
Jalur di bawah mengikuti [instalasi resmi AMD](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)
untuk `gfx1032` (RX 6600 LE) dengan ROCm 10.1.0. Kecocokan pada workspace ini
diverifikasi langsung dengan forward, backward, dan langkah optimizer.

```bash
mkdir -p runs
uv venv --python 3.12 .venv-rocm
uv export --frozen --no-emit-project --no-emit-package torch --output-file runs/requirements-base.txt
uv pip install --python .venv-rocm/bin/python -r runs/requirements-base.txt
uv pip install --python .venv-rocm/bin/python 'torch[device-gfx1032]==2.14.0+rocm10.1.0' --index https://stable.repo.amd.com/rocm/whl-next/
uv pip install --python .venv-rocm/bin/python --no-deps --editable .
```

Proyek memilih ordinal ROCm `0`, yaitu RX 6600 LE pada komputer ini, dan
mengunci `ROCM_SDK_TARGET_FAMILY=gfx1032`. Mask lama ditimpa sebelum inisialisasi
GPU; perangkat selain seri RX 6000 ditolak sebelum alokasi model. Radeon 780M
tidak digunakan. Jika urutan perangkat berbeda pada komputer lain, tetapkan
`KETIK_ROCM_DEVICE` sesuai ordinal RX 6600 LE yang diperiksa. Profil SDK ini
khusus arsitektur `gfx1032`.

```bash
.venv-rocm/bin/ketik doctor --json --output runs/pemeriksaan-rocm.json
.venv-rocm/bin/ketik train cnn --dataset data/real/processed --device gpu --output runs/cnn-gpu
.venv-rocm/bin/ketik evaluate --model runs/cnn-gpu --data data/real --device gpu
uv run ketik report --models runs/svm runs/cnn-cpu runs/cnn-gpu --output reports/perbandingan
```

Jalankan `.venv-rocm/bin/ketik` langsung, bukan `uv run`, untuk mempertahankan
build ROCm. Gunakan data, seed, batch, dan batas epoch sama untuk perbandingan
CPU/GPU; periksa juga jumlah epoch aktual jika early stopping berbeda.
Inferensi diukur setelah pemanasan dan sinkronisasi GPU. Waktu per ketukan yang
dilaporkan adalah throughput batch, bukan latensi aplikasi langsung.

`--device gpu` gagal dengan penjelasan jika tes GPU tidak lolos. `--device auto`
mencoba GPU lalu mencatat alasan ketika beralih ke CPU. `--device cpu` selalu
memakai CPU. Jika GPU belum kompatibel, hasil CPU dan kendalanya tetap dapat
diselesaikan tanpa mengunggah dataset.

Pada verifikasi workspace ini, **RX 6600 LE lulus forward/backward/optimizer**.
Hasil pemeriksaan terbaru ada di `runs/doctor-rx6600.json`. Berkas diagnosis
sebelumnya merupakan catatan historis dan tidak menjadi konfigurasi aktif.

## Verifikasi perangkat lunak

```bash
uv run pytest -q
uv run ruff check src tests research
uv run python -m ketik_audio.demo --output runs/smoke --epochs 2
```

Demo membuat enam rekaman **sintetis** dengan bunyi terkontrol, melatih kedua
model, mengevaluasi, memprediksi WAV tanpa label, dan menghasilkan
`runs/smoke/report/laporan.md`. Demo tidak membutuhkan mikrofon. Hasilnya
ditandai sebagai uji perangkat lunak, **bukan pengukuran akurasi keyboard nyata**.

## Rancangan pengumpulan dengan perekam terminal

Rancangan 14 hari berikut memakai alur enam sesi dengan perekam terminal.

| Hari | Kegiatan |
|---|---|
| 1–2 | Instalasi pada laptop/komputer pelatihan, doctor, dan uji perekaman pilot. |
| 3–5 | Rekam sesi 1–5, periksa kualitas, dan siapkan fitur. |
| 6–9 | Latih SVM/CNN, tetapkan konfigurasi memakai validasi. |
| 10–12 | Rekam sesi 6, evaluasi akhir, dan bandingkan CPU/GPU bila tersedia. |
| 13–14 | Tinjau confusion matrix, tulis laporan dan batasan hasil. |

Rekaman nyata memerlukan partisipasi Anda. Folder `data/`, `runs/`, dan
`reports/` diabaikan Git agar data audio dan artefak eksperimen tersimpan lokal.
