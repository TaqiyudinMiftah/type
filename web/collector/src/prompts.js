// Separate, versioned text sets for calibration and the two held-out days.
const BANK = {
  calibration: [
    'Pagi ini saya membuka jendela dan merapikan meja kerja. Sebuah buku, segelas air, dan laptop sudah siap. Saya membaca catatan singkat sebelum memulai kegiatan. Di luar, angin menggerakkan daun dan cahaya masuk perlahan. Saya mengetik dengan kecepatan yang nyaman, lalu memperbaiki kata jika ada yang keliru.',
    'Di ruang pamer sains, kami melihat contoh xilem, gas xenon, dan foto sinar x. Petugas bernama xenia menjelaskan setiap benda dengan sabar. Pada bagian komputer, ada gambar papan qwerty dan kartu kuis. Kami mencatat jawaban di buku sambil berdiskusi tentang cara kerja alat yang menarik itu.',
    'Sore itu, zaki dan zahra memilih menu di kafe kecil. Mereka membicarakan perjalanan ke qatar, jadwal pesawat qantas, dan foto dari kawasan wisata. Musik jazz terdengar pelan. Di halaman majalah ada profil seorang qari, latihan kuis, dan cerita tentang bazar yang ramai pada akhir pekan.',
    'Kelompok belajar kami menyusun katalog sederhana. Folder bernama qwerty berisi catatan tombol, sedangkan folder xenon memuat gambar percobaan. Xenia menambahkan indeks dan zaki memeriksa urutan. Kami menyimpan versi akhir setelah membaca setiap halaman dan memastikan semua berkas dapat dibuka.',
    'Peta perjalanan qatar diletakkan di samping brosur qantas. Dalam kuis, zahra menulis qatar sebagai tujuan, lalu mengecek kembali ejaannya. Ia menyukai teka-teki tentang xenon, xilem, dan musik jazz. Setelah itu kami berjalan ke bazar untuk membeli sayur, roti, dan beberapa buah segar.'
  ],
  test1: [
    'Hujan berhenti menjelang siang. Saya membawa payung ke teras dan mendengarkan suara burung di halaman. Tetangga sedang menata tanaman dalam pot baru. Setelah menonton video singkat, saya kembali ke meja untuk menulis rencana kegiatan. Ada beberapa pekerjaan kecil yang ingin saya selesaikan hari ini.',
    'Perpustakaan mengadakan diskusi bersama xenia tentang xilem dan xenon. Buku panduan qwerty tersusun di rak komputer. Zaki mencatat pertanyaan untuk kuis, sementara zahra membaca artikel musik jazz. Mereka saling bertukar pendapat sebelum memilih buku yang akan dibawa pulang.',
    'Cerita perjalanan kali ini membahas qatar dan rute qantas. Seorang qari berbagi pengalaman tentang bazar buku di sana. Kami melihat foto perahu, gedung, dan taman. Dalam kuis penutup, xenia mendapat pertanyaan tentang sinar x, sedangkan zaki menjelaskan contoh gas xenon di ruang pamer.',
    'Penyusun kuis memberi nama berkas qwerty agar mudah ditemukan. Topiknya beragam, mulai dari qatar hingga xilem pada tumbuhan. Zahra memeriksa ejaan xenon dan xenia. Di bagian akhir ada pertanyaan tentang jazz, bazar, serta jadwal qantas. Kami mengoreksi catatan itu dengan tenang.',
    'Di bazar sekolah, zaki menampilkan poster qatar. Xenia menyiapkan percobaan xenon dan gambar xilem. Zahra mengurus kuis di dekat meja qwerty, sementara kelompok musik memainkan jazz. Semua pengunjung mendapat kesempatan bertanya dan mencoba kegiatan sesuai minat masing-masing.'
  ],
  test2: [
    'Kami berangkat ke pasar saat udara masih sejuk. Penjual menyusun barang di atas meja dan menyapa pembeli yang datang. Saya memilih bahan untuk makan malam, lalu merekam video singkat di dekat taman. Sepanjang perjalanan pulang, saya memikirkan resep baru yang bisa dicoba bersama keluarga.',
    'Dalam lokakarya, xenia mengenalkan simbol sinar x dan penggunaan xenon. Seorang qari menjadi pembawa acara, lalu zaki membacakan kuis tentang qatar. Peserta menulis jawaban dengan papan qwerty. Zahra memeriksa catatan sambil mendengarkan musik jazz yang diputar di ruang sebelah.',
    'Majalah bulanan memuat cerita bazar, perjalanan qantas, dan peta qatar. Pada halaman sains ada gambar xilem serta percobaan xenon. Xenia dan zaki menggunting gambar untuk bahan kuis. Mereka menyusun semuanya dalam urutan yang mudah dibaca sebelum memasukkannya ke folder qwerty.',
    'Zahra menyusun kuis untuk teman-temannya. Pertanyaan tentang qatar dan qantas ada di bagian perjalanan. Pertanyaan xenon, xilem, dan sinar x masuk bagian sains. Xenia menambahkan soal qwerty, sedangkan zaki menulis soal jazz dan bazar. Mereka ingin setiap peserta menikmati proses belajar.',
    'Ketika acara selesai, xenia merapikan kartu xenon dan xilem. Zaki mengembalikan papan qwerty ke lemari. Zahra menyimpan kuis qatar, qantas, dan jazz dalam satu map. Di dekat pintu bazar, seorang qari berbincang dengan panitia. Kami pulang setelah memastikan ruang sudah bersih.'
  ]
};
export const PROMPT_VERSION = 'id-natural-1';
export function choosePrompt(role, counts, used) {
  const bank = BANK[role], available = bank.map((text, i) => ({ id: `${PROMPT_VERSION}:${role}:${i}`, text }));
  const remaining = available.filter(p => !used.includes(p.id));
  const choices = remaining.length ? remaining : available;
  return choices.map(p => ({ ...p, score: [...p.text].reduce((s, c) => s + (c in counts ? 1 / (1 + counts[c]) ** 2 : 0), 0) })).sort((a, b) => b.score - a.score || a.id.localeCompare(b.id))[0];
}
export { BANK };
