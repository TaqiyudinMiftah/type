# Seberapa konsisten anggaran label pada SKAID?

15 November 2026. Pada **10 contoh per huruf**, MFCC+Ridge mencapai
80% pada setidaknya satu pilihan kalibrasi untuk **9 dari 17 ID**.
Hanya **4 dari 17** mencapai 80% pada ketiga pilihan. Rerata seluruh
kohort adalah 75,78%. Jadi rerata dan hasil dari satu pilihan contoh
menjawab pertanyaan yang berbeda dari konsistensi per ID.

Audit ini menghitung ulang hasil yang sudah ada untuk 12 huruf, 60 ketukan
uji per ID dari Phase 2, dan tiga pilihan contoh Phase 1. Data uji tetap
sama di tiap pilihan. Untuk tiap ID, 48 jawaban benar berarti 80%; 54
berarti 90%. Hitungan memakai bilangan bulat sebelum pembulatan. Tidak ada
model, prediksi, atau rekaman baru.

## Kurva Ridge per ID

| Contoh per huruf | Label target per fit | Rerata kohort | ≥80% pada minimal satu pilihan | ≥80% pada rerata tiga pilihan | ≥80% pada ketiganya |
|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 43,40% | 0/17 | 0/17 | 0/17 |
| 2 | 24 | 55,78% | 1/17 | 0/17 | 0/17 |
| 3 | 36 | 61,60% | 1/17 | 1/17 | 0/17 |
| 5 | 60 | 69,02% | 5/17 | 2/17 | 1/17 |
| 10 | 120 | 75,78% | 9/17 | 6/17 | 4/17 |

Satu ID mulai memenuhi 80% di semua pilihan pada K5 dan masih memenuhi
di K10. Tiga ID lain baru memenuhi di K10. Untuk 13 ID, pencapaian di
semua pilihan belum terlihat pada grid yang diuji. Pada K10, selisih
pilihan terbaik dan terburuk per ID memiliki median 5,00 poin persentase
dan maksimum 16,67 poin. Meski rerata per ID tidak turun saat K5→K10,
tiga ID mengalami penurunan di setidaknya satu pilihan contoh yang sama.

## Pembanding pada ketukan uji yang sama

| Metode pada K10 | Rerata kohort | ≥80% pada ketiga pilihan | Masih ≥80% jika tiap pilihan mendapat satu kesalahan tambahan | ≥90% pada ketiga pilihan |
|---|---:|---:|---:|---:|
| Ridge | 75,78% | 4/17 | 4/17 | 0/17 |
| SVM linear | 76,63% | 4/17 | 4/17 | 0/17 |
| SVM RBF | 76,83% | 6/17 | 4/17 | 0/17 |

Dua ID tambahan pada RBF berada tepat di batas 48/60 pada pilihan
terlemahnya. Uji dengan satu kesalahan tambahan menunjukkan sensitivitas
pada ukuran uji ini; ini bukan interval keyakinan. Hanya **3 dari 17 ID**
mencapai 80% di semua pilihan pada ketiga metode. Untuk 90%, ada satu ID
pada linear dan dua ID pada RBF yang mencapai ambang setidaknya sekali,
tetapi tidak ada satu pun yang bertahan pada tiga pilihan di metode mana
pun. Pada RBF, satu ID memenuhi 80% di semua pilihan pada K3 tetapi tidak
di K5. Ambang yang pertama terlewati dapat berbeda dari ambang yang
bertahan pada semua K yang lebih besar di grid.

## Batas hasil

Ketiga metode memakai pembagian dukungan dan uji yang sama. Pada K10,
120 label target adalah biaya fit per ID, di samping 5.749–5.807 label
sumber per fold yang dipakai pemilihan resep. Ketiga pilihan kalibrasi
berbagi 60 ketukan uji per ID. Totalnya 1.020 ketukan uji unik dalam
kohort, dan fase SKAID tidak memverifikasi hari atau mikrofon fisik yang
berbeda. Angka ini belum menentukan jumlah label minimum di antara titik
grid, pada mikrofon baru, atau untuk 26 huruf. Pada uji mikrofon berikutnya,
laporkan jumlah perangkat atau peserta yang melewati ambang di setiap
pilihan kalibrasi bersama rerata akurasi.

Hash pembagian dan anggaran sama di kedua run sumber. Audit memeriksa
765 catatan akurasi, pemisahan dukungan dan uji, serta 255 rerata per ID
terhadap CSV lama. Hitungan ambang dan ambang yang bertahan juga
diperiksa dengan agregasi terpisah.

[Ringkasan hitungan](results/skaid-budget-reliability-2026-11-15.json) ·
[skrip audit](audit_skaid_budget_reliability.py) ·
[seleksi Ridge](SKAID_SELECTION_OBSERVATIONS.md) ·
[pembanding SVM](SKAID_SVM_C_OBSERVATIONS.md).
