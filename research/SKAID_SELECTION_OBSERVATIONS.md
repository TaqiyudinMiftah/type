# SKAID: pemilihan regularisasi dan normalisasi dari ID lain

Tanggal lokal: 9 November 2026. **MFCC K10 naik dari 52,29% menjadi
75,78%**, BEATs dari 26,31% menjadi 69,80%, dan ResNet pralatih dari
43,12% menjadi 69,96%. Setiap model memilih resep dari 16 ID lainnya;
query ID yang ditahan tidak ikut kriterianya. Kenaikan memakai label
sumber tambahan dan tetap belum mencapai mean 80%/90% pada grid K1–10.

## Desain dan urutan kerja

[Protokol](SKAID_SELECTION_PROTOCOL.md) ditulis setelah hasil pilot
alpha 1 diketahui, sebelum kandidat baru dihitung. Pertahankan 17 ID,
12 huruf, crop 80 ms, query tetap Phase 2, dukungan nested Phase 1,
tiga draw dan delapan encoder beku. Semua tahap ini memakai CPU pada
cache yang sudah diaudit; tidak mengekstrak atau memperbarui backbone.

Dua normalisasi—row L2 dan z-score dukungan lalu L2—disilangkan dengan
sembilan alpha 1e-6 hingga 100. Statistik scaler hanya berasal dari
dukungan percobaan, dengan aturan fitur near-constant yang ditetapkan
sebelumnya. Satu resep per keluarga/K/held ID dipakai bersama untuk
semua seed/draw; tidak memilih seed terbaik. Seri memilih alpha lebih
besar, lalu row L2.

Setiap ID menjadi validasi sumber untuk 16 fold lain. Kandidat pada
semua query memang dihitung sebagai validasi; untuk model h, baris h
dikeluarkan sebelum fungsi seleksi menerima kriterianya. **Ini validasi
silang pada kohort yang sudah diperiksa**, bukan test baru yang belum
pernah dilihat. Seluruh 408 pilihan (340 normal + 68 kontrol) dikunci
sebelum refit/evaluasi luar. Mengubah hanya baris h tidak mengubah
resep model h pada audit counterfactual.

## Seluruh hasil

Mean antar 17 ID setelah merata-ratakan tiga draw, serta tiga model
pada setiap keluarga ResNet. Ada 60 query per ID, seimbang pada 12 kelas;
peluang uniform 8,33%. Ini rerata akurasi, bukan ensemble logits.

| K / label target | MFCC | BEATs | ResNet pralatih | ResNet acak |
|---|---:|---:|---:|---:|
| 1 /12 | 43,40% | 21,11% | 29,28% | 15,90% |
| 2 /24 | 55,78% | 33,56% | 40,59% | 18,66% |
| 3 /36 | 61,60% | 42,75% | 47,85% | 20,92% |
| 5 /60 | 69,02% | 56,44% | 58,16% | 24,95% |
| 10 /120 | 75,78% | 69,80% | 69,96% | 29,51% |

![Kurva baseline dan terpilih](../reports/skaid-selection/kurva-baseline-vs-seleksi.png)

| Fitur K10 | Mean query | CI95% bersyarat ID | Dukungan fit-label | Kontrol query |
|---|---:|---:|---:|---:|
| MFCC | 75,78% | 72,16–79,25% | 97,66% | 7,88% |
| BEATs | 69,80% | 64,93–74,54% | 100,00% | 8,50% |
| ResNet pralatih | 69,96% | 65,10–74,74% | 99,96% | 8,10% |
| ResNet acak | 29,51% | 25,98–33,20% | 73,24% | 8,47% |

Peningkatan K10 terjadi pada semua 17 ID untuk setiap keluarga. CI memakai
10.000 bootstrap ID seed 7471, dengan draw/model tetap di dalam ID.
Fold berbagi data sumber; interval bersifat deskriptif dan bersyarat pada
kohort/kebijakan, bukan CI populasi mikrofon atau sesi baru. MFCC top-3
mencapai 91,27%, tetapi top-1 tetap 75,78%; jangan mengganti kriteria 80%
dengan metrik ranking tersebut.

Di antara 340 pilihan normal, 288 memakai row L2 dan 52 support z-score.
Tidak ada yang memilih batas alpha terbawah; dua memilih batas atas 100.
Pada MFCC, seluruh held ID memilih row L2/alpha 0,001 untuk semua K.
Grid tidak diperluas setelah hasil luar diketahui.

## Biaya supervisi tambahan

Budget target per percobaan tetap 12×K label classifier. Pemilihan resep
menggunakan union dukungan tiga draw pada 16 ID lain ditambah 960 query
sumber unik. Label tidak dikalikan karena encoder/resep memakai ulang
contoh sama.

| K | Label target | Label sumber unik per fold |
|---:|---:|---:|
| 1 | 12 | 1.526–1.528 |
| 2 | 24 | 2.073–2.078 |
| 3 | 36 | 2.601–2.608 |
| 5 | 60 | 3.595–3.618 |
| 10 | 120 | 5.749–5.807 |

960 query sumber termasuk dalam setiap total. Union untuk seluruh K
sama dengan K10 karena dukungan nested. ResNet pralatih juga memakai
ulang 720 label Harrison; generic pretraining BEATs tidak dianggap 0 label
total. Anotasi log untuk timestamp/filter/stratifikasi dan query held ID
untuk evaluasi berada di luar budget classifier target. Kenaikan ini
tidak boleh disebut perbaikan gratis pada jumlah anotasi keseluruhan.

## Audit dan variasi numerik

Seluruh 44.064 kandidat diperiksa terhadap sklearn; 2.448 di antaranya
memakai ulang baseline pilot persis, 41.616 kandidat baru. Ada 4.896
faktorisasi Gram untuk path alpha. Semua 408 pilihan dan 2.448 refit
diperiksa, termasuk scaler support-only, union label budget, pemisahan
held ID, dan tie rule independen. Kontrol negatif memilih resep sendiri
dengan fleksibilitas yang sama, menggunakan permutasi support-label lama.

Prediksi semua kandidat cocok dengan sklearn; replay final identik.
Selisih skor kandidat maksimal 3,32e-6, dalam atol 1e-6/rtol 1e-5 yang
ditetapkan sebelumnya. Koefisien/skor final dibandingkan dual ridge
independen berbeda maksimal 4,89e-8, tanpa perubahan prediksi.

Kontrol memakai ulang cache fitur CPU lengkap dari audit pilot, tanpa
ekstraksi baru. Ada 7 perubahan prediksi dari 146.880 prediksi berulang:
satu BEATs normal yang tetap salah, dan enam pada kontrol ResNet acak.
Akurasi top-1 seluruh kondisi normal tetap sama; MFCC dan ResNet pralatih
tidak berubah prediksinya. Satu kontrol berubah benar→salah. Pilihan
tidak diperbarui dari hasil CPU ini.

Selisih skor terbesar 1,424 ada pada kontrol `random_s44` dengan support
z-score/alpha 1e-5; condition number Gram sekitar 7,1×10⁶. Mask fitur
near-constant tidak berubah. Pola konsisten dengan amplifikasi perubahan
input kecil pada sistem berkondisi buruk; ini inferensi diagnostik,
bukan bukti penyebab tunggal. Skor Ridge bukan probabilitas.

Seluruh 112 tes, Ruff lint/format, lock dan build wheel/sdist lulus.

## Implikasi dan kelanjutan

Fit dukungan neural kini hampir 100% sementara query sekitar 70%.
Keterbatasan sekarang tidak dapat diselesaikan hanya dengan menyatakan
bahwa model sudah menghafal dukungan. Kepala nonlinear layak diuji dengan
seleksi pada ID sumber yang sama, sambil mempertahankan query dan budget.
Jangan memilih C/gamma dari query ID yang sedang dievaluasi.

Hasil tetap 12 kelas dengan timestamp/filter acuan. Metadata belum
membuktikan mikrofon, keyboard atau sesi fisik baru; onset belum dianotasi
manual. Studi ini bukan detektor bebas log, biaya label end-to-end,
reproduksi literal EchoStrike atau penetapan SOTA. Data sendiri nantinya
perlu jumlah per tombol seimbang dan identitas kondisi rekaman yang jelas.

- [Laporan dan grafik lengkap](../reports/skaid-selection/laporan.md).
- [Pilihan terkunci](../runs/skaid-selection/selection.json),
  [CSV pilihan](../reports/skaid-selection/selected-recipes.csv), dan
  [biaya label](../reports/skaid-selection/label-budgets.csv).
- [Audit](../reports/skaid-selection-audit/audit.json),
  [prediksi final](../reports/skaid-selection-audit/classifications.csv),
  serta [diagnostik numerik](../reports/skaid-selection-numeric/summary.json).
- [Snapshot](results/skaid-selection-2026-11-09.json).
