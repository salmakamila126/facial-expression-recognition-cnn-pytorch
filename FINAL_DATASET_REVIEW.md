# Final dataset review — clean_grouped_v2

**Keputusan: pertahankan versi beku ini secara provisional; belum memberi
persetujuan visual final atau persetujuan training.** Tidak ditemukan kegagalan
integritas struktural. Kebijakan karantina konservatif layak dipertahankan bila
review manusia menerima pengecualiannya. Perlu perhatian khusus pada 12 grup
(86 gambar) yang meluas melewati satu komponen reliable. Jika pemeriksaan visual
menemukan false positive yang tidak dapat diterima, usulkan protokol versi baru;
jangan mengubah label, keputusan, split atau hash versi ini.

## Sumber dan hal yang sudah diverifikasi

Artefak ditemukan di [folder audit Google Drive](https://drive.google.com/drive/folders/1LAm30m4_fAfYOKP9yMDQbe3NvpJxHyYv).
Salinan file diunduh untuk pemeriksaan lokal; file Drive tidak ditulis ulang.

- `summary.json`, `configuration.json`, `decision_manifest.csv`,
  `split_manifest.csv`, `candidate_pairs.csv`, `conflicting_groups_manifest.csv`,
  serta `audit_manifest.csv` dari folder run induk dibaca/diperiksa.
- SHA-256 byte file split dihitung ulang dan cocok:
  `7e9742da12ed2caa4652e23af7a78215f57fd1e44e37cec2e7c70cd664b66ec8`.
- 59.099 path/label/hash piksel cocok dengan manifest pemindaian awal. Tidak ada
  gambar invalid tercatat di `invalid_images.csv` (header saja).
- Seluruh 28.775 baris eligible sama dengan manifest split, termasuk label,
  target, pixel hash dan kedua jenis group ID. Train 20.143, validation 4.316,
  test 4.316. Lima kelas ada di setiap split.
- Seluruh **30.324 pengecualian** tetap tercatat: **29.951 redundant exact copies**
  dan **373 gambar karantina konflik**. Tidak ada yang masuk split. Setiap redundant
  exact copy mempunyai representasi dengan pixel hash sama yang tetap eligible.
- Seluruh 65 grup campuran dikarantina utuh. Manifest khusus konflik cocok persis
  dengan subset karantina di manifest keputusan.
- Komponen isolation dan reliable direkonstruksi dari exact hashes dan edges
  tersimpan; hasil cocok dengan membership manifest. Tidak ada group isolation,
  reliable atau exact pixel hash yang melintasi split.
- Hash implementasi dalam konfigurasi audit sama dengan `clean_grouped_v2.py`
  lokal: `63e12c648a0fb3fd83a7a83d6e972009069ac0d9a4f1f42c39f54b7415030228`.
- Byte seluruh artefak input lokal diperiksa sebelum/sesudah review; tidak berubah.

Pencocokan ini membuktikan konsistensi artefak dan pencatatan pengecualian.
Keberadaan/keutuhan seluruh file gambar sumber tidak diverifikasi dari manifest
saja. Tidak dilakukan pemeriksaan visual foto asli dalam sesi ini.

## Kebijakan karantina dan alasan

Kandidat memakai dHash <=4 dan grayscale MAE <=3 pada 48x48. Edge reliable
memerlukan dimensi asli sama, dHash <=1, RGB MAE <=1, p99 RGB difference <=4,
dan standar deviasi grayscale >=10 pada kedua gambar. Semua edge kandidat
membentuk **isolation component**; reliable component dicatat terpisah.
Komponen isolation dengan lebih dari satu label dikarantina seluruhnya.
Tidak ada voting label, prediksi model, pelabelan ulang, atau penghapusan sumber.

Kategori berikut saling eksklusif menurut bukti terkuat dalam tiap grup. Jumlah
gambar meliputi seluruh anggota grup, termasuk anggota tambahan dari edge lemah.

| Bukti terkuat di dalam grup | Grup | Gambar |
|---|---:|---:|
| Konflik label dengan pixel hash identik | 49 | 268 |
| Konflik di komponen reliable, tanpa konflik exact | 8 | 57 |
| Konflik hanya melalui komponen kandidat tidak pasti | 8 | 48 |
| Total | 65 | 373 |

Konflik exact mendukung keputusan untuk menahan sampel ambigu, tetapi tidak
menunjukkan label mana yang benar. Kategori reliable tetap merupakan aturan
heuristik, bukan konfirmasi manusia. Daftar per grup beserta alasan dan metrik
tersedia di keluaran `conflict_group_review.csv` dan `conflict_edges_review.csv`.

## Dampak per kelas

Persentase dihitung terhadap gambar readable asli kelas tersebut, termasuk
salinan exact. Banyaknya salinan dapat membuat dampak pada foto unik berbeda.

| Kelas | Original | Retained | Karantina konflik | % original |
|---|---:|---:|---:|---:|
| Angry | 10.148 | 4.947 | 101 | 1,00% |
| Fear | 9.732 | 4.807 | 114 | 1,17% |
| Happy | 18.439 | 9.254 | 11 | 0,06% |
| Sad | 12.553 | 6.301 | 72 | 0,57% |
| Surprise | 8.227 | 3.466 | 75 | 0,91% |
| Total | 59.099 | 28.775 | 373 | 0,63% |

Fear, Angry dan Surprise terdampak lebih besar daripada Happy. Ini bukan bukti
filter salah, tetapi alasan untuk meninjau seluruh 65 grup dan mencatat dampak
pada foto unik, bukan hanya jumlah file.

## Risiko over-grouping yang masih memerlukan visual

- **12 grup / 86 gambar** terdiri atas lebih dari satu reliable component.
  Artinya ada perluasan melalui edge tidak pasti; bukan otomatis false positive.
- **1 grup / 20 gambar** memiliki graph kandidat tidak lengkap setelah salinan
  exact digabung. Beberapa anggota terhubung transitif tanpa pasangan langsung.
  Prioritas tertinggi: `isolate_0c7d7a51e5cc95b589e7ba0efe1fa9746ebc2c8b42e5dabd9c09694233572e2c`
  (18 Angry, 2 Sad, 5 pixel hash unik, 4 reliable components).
- Salah satu edge lemah dapat menyeret foto berbeda ke karantina. Subjek sama
  dengan ekspresi berbeda tidak boleh dianggap salinan otomatis. Resize/grayscale
  dapat menyembunyikan perbedaan asli. Tinjau crop, detail wajah, ekspresi dan warna.
- Grup besar dari salinan exact tidak sendirinya menunjukkan over-grouping;
  periksa jumlah pixel hash unik dan graf edge, bukan hanya jumlah file.
- Tidak ada jaminan identitas subjek terpisah antar split, atau semua near duplicate
  telah terdeteksi. Integritas komponen teridentifikasi tidak menghilangkan risiko ini.

Tidak ditemukan kebutuhan revisi wajib dari validasi struktural saja. Keputusan
apakah edge tertentu terlalu agresif tetap menunggu pemeriksaan visual.

## Prosedur Colab tanpa membuat dataset ulang

1. Buka `Review_clean_grouped_v2.ipynb` yang baru di Colab; gunakan CPU.
   Unggah `review_dataset_v2.py` ke `/content/`. Tidak perlu Run all notebook training.
2. Mount Drive. Isi `AUDIT_DIR` dengan folder audit tersimpan yang memiliki hash
   di atas, atau biarkan pencarian otomatis memilih satu folder yang cocok.
   Jika lebih dari satu cocok, notebook berhenti agar lokasi dipilih eksplisit.
3. Jika runtime audit lama masih hidup, isi `DATA_DIR` dengan lokasi folder `Data`.
   Jika sudah hilang, biarkan `DATA_DIR=None`, sesuaikan `ZIP_PATH`. Kode memverifikasi
   SHA arsip lalu mengekstrak **hanya 373 sumber konflik** ke folder sementara.
   Tidak ada deteksi duplikat, penghitungan split, atau preprocessing ulang dataset.
4. Jalankan notebook review. Output masuk folder baru
   `RUN_DIR/final_conflict_review_<timestamp>_<uuid>/` di Drive, terpisah dari audit.
5. Buka `index.html` beserta folder PNG yang menyertainya, atau lihat PNG langsung
   di Drive. Satu halaman hanya menampilkan satu group ID lengkap, maksimum
   9 gambar, panel gambar hingga 320x320, label asli dan path. Pembesaran untuk
   tampilan tidak menambah detail asli; full path juga ada di CSV anggota grup.
6. Periksa `conflict_image_status.csv`: semua 373 harus `verified` sebelum
   menyelesaikan pemeriksaan visual. File hilang/berubah ditandai, tidak diganti
   dengan gambar baru dan tidak dianggap lulus visual.
7. Mulai dari 12 `priority=first` pada `conflict_group_review.csv`, terutama grup
   rantai di atas. Periksa setiap anggota dan edge pada `conflict_edges_review.csv`.
   Lanjutkan seluruh 65 grup, bukan seluruh dataset.
8. Isi salinan `review_decisions_template.csv`: reviewer, status
   pending/reviewed/needs_second_review; hubungan visual berupa exact copy,
   near copy, same subject/different expression, different images, atau uncertain;
   recommendation retain_quarantine/review_future_protocol/uncertain; dan catatan path.
   Tidak perlu menentukan label benar. Untuk grup ambigu minta pemeriksaan kedua.
9. Selesaikan hanya setelah semua grup memiliki keputusan dan dampak per kelas
   sudah diterima. Rekomendasi bersifat catatan; kode tidak mengaplikasikannya.
   Jika perlu membership berbeda, buat usulan versi baru terpisah dengan hash baru.

Output tambahan: `conflict_class_impact.csv`, `conflict_members_review.csv`,
`all_exclusions_ledger.csv`, `review_report.md`, `review_integrity.json`,
`input_hashes.json`. Kode memeriksa ulang hash split setelah selesai. Tidak ada
setting yang menyetujui training dan tidak membaca prediksi/performa test.

## Batas pelaksanaan dan perubahan file

`clean_grouped_v2.py` dan `CNN_resnet18.ipynb` sengaja **tidak diubah** dalam review
ini, sehingga konstruksi dan checksum modul tetap sama. Notebook SimpleCNN,
checkpoint, label, split dan file audit Drive juga tidak diubah.

Ditambahkan: `review_dataset_v2.py`, `Review_clean_grouped_v2.ipynb`, dokumen ini,
serta tes review baru. Tes lama diperbarui agar menerima kedua pilihan protokol
valid pada konfigurasi notebook yang sudah dipilih pengguna; training tetap
wajib default False. Validasi JSON/sintaks dan 8 tes lokal lulus, termasuk review
sintetis dengan gambar besar, penolakan split berubah, pengecualian hilang,
ekstraksi ZIP selektif dan deteksi gambar sumber berubah.

Contact sheet foto asli belum dibuat dalam sesi ini karena sumber gambar tidak
tersedia di filesystem lokal. Notebook Colab sudah disiapkan untuk membuatnya
langsung dari audit tersimpan dan sumber asli. Contact sheet baru belum ditinjau
manusia. Tidak ada training, final test, commit, push atau merge.
