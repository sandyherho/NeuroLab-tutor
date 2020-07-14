# Pendahuluan

<p align=”justify”>Pada tutorial ini saya akan membahas tentang aspek praktis dari algoritma jaringan saraf tiruan menggunakan pustaka NeuroLab yang berbasis pada bahasa pemrograman Python. Jaringan saraf tiruan saat ini sedang menjadi buah bibir dalam perkembangan teknologi dunia semenjak Alex Krizhevsky (pendiri laboratorium Dessa), Ilya Sutskever (ilmuwan kepala di laboratorium OpenAI), dan Geoffrey E. Hinton (profesor di Departemen Ilmu Komputer, Universitas Toronto) menerbitkan makalah tentang penerapan pemelajaran mendalam (<i>deep learning</i>) untuk mengklasifikasikan 1,2 juta citra menggunakan arsitektur yang nantinya dikenal sebagai AlexNet (Krizhevsky dkk., 2012) pada <i>Conference and Workshop on Neural Information Processing Systems</i> (NeurIPS) yang diselenggarakan di Stateline, Nevada, Amerika Serikat tahun 2012 silam. Semenjak itu, hampir setiap perusahaan teknologi raksasa, seperti Google; Facebook; Baidu; dll melakukan ekspansi besar - besaran untuk mendanai tim riset algoritma ini.</p>

<p align=”justify”>Algoritma jaringan saraf tiruan saat ini merupakan bagian penting dari riset dan pengembangan teknologi buatan yang menjadi sendi utama dalam revolusi industri ke-4, seperti yang diterangkan oleh ekonom dunia Karl Schwab (Schwab, 2016). Presiden kita, Ir. H. Joko Widodo, yang dalam hemat saya, beserta staf -staf kepresiden juga tidak mengerti apa - apa tentang algoritma jaringan saraf tiruan, sempat mewacanakan perampingan struktur jabatan aparatur sipil negara (ASN) dengan memanfaatkan dengan menggunakan aplikasi kecerdasan buatan (Tempo, 2019), yang saya anggap sebagai suatu utopia kosong yang umum digaungkan orang - orang bodoh yang belum belajar sulitnya penerapan algoritma jaringan saraf tiruan di dunia nyata.</p>

<p align=”justify”>Daripada membicarakan politik yang bukan level-nya saya, maka baiknya kita kembali ke isi buku singkat ini, kita tidak akan membahas hal - hal yang tinggi - tinggi juga muluk - muluk seperti penerapan <i>Generative Adversarial Network</i> (GAN) (Goodfellow dkk., 2014), untuk membuat foto <i>bohongan</i> seperti yang saat ini menjadi kontroversi. Kita akan membahas hal - hal sederhana, seperti:</p>
<ul>
    <li>konsep - konsep dasar dalam algoritma jaringan saraf tiruan,</li>
    <li>penerapan klasifikasi menggunakan perseptron tunggal,</li>
    <li>penerapan algoritma jaringan saraf buatan berlapis tunggal (<i>single-layer neural network</i>),</li>
    <li>penerapan algoritma jaringan saraf buatan berlapis banyak (<i>multi-layer neural networks</i>),</li>
    <li>pengantar analisis data sikuensial menggunakan algoritma <i>Recurrent Neural Networks</i> (RNNs), dan</li>
    <li>konstruksi mesin pengenalan karakter optis (<i>Optical Character Recognition</i>: OCR) sederhana.</li>
</ul>

<p align=”justify”>Seluruhnya akan diterapkan pada pustaka <a href="https://github.com/zueve/neurolab">NeuroLab versi 0.3.5</a> (alih - alih pustaka <i>deep learning</i> Python lainnya yang lebih populer seperti: TensorFlow (Abadi dkk., 2016), Keras (Chollet dkk., 2015), dan PyTorch (Paszke dkk., 2019)), yang dikembangkan oleh tiga orang pengembang Rusia dengan nama samaran: maksimovVva; zueve; dan solarjoe,untuk mempermudah para akademisi Indonesia di luar bidang ilmu komputer yang lebih terbiasa dengan <i>Neural Network Toolbox</i> (NNT) yang dijalankan pada piranti lunak berbayar: MATLAB. Saya juga pada dasarnya akan mengikuti struktur buku Prateek Joshi (Joshi, 2017) dalam alur pembagian materi di tutorial ini. <i>Oh, ya</i> satu hal lagi yang perlu saya ingatkan karena tutorial ini bersifat praktis, saya tidak akan membahas tentang konsep matematis algoritma ini. Disarankan pembaca menunggu buku yang saya dan Dr. Dasapta Erwin Irawan (Teknik Geologi ITB) akan siapkan pada pertengahan tahun depan (jika saya masih hidup dan sehat) untuk pembahasan lebih mendasar tentang konsep <i>back-propagation</i>, fungsi aktifasi, dll. Atau kalau sudah tidak sabar, <i>monggo</i> untuk melihat buku - buku seperti: Goodfellow, dkk. (2016), Burkov (2019), dan (mungkin yang lebih <i>asik</i> dan praktis bagi saya pribadi yang <i>tolol</i> dalam matematika) Géron (2019). Sebagai tambahan lagi, di sini saya juga mengharapkan pembaca punya sedikit pengalaman dengan bahasa pemrograman Python sampai tingkatan pemrograman ber-<i>gaya</i> fungsional dan mahir dalam penggunaan pustaka NumPy (Oliphant, 2006) serta matplotlib (Hunter, 2007) karena saya tidak akan membahas lagi apa itu vektorisasi <i>array</i>, apa itu <code>if __name__ == '__main__':</code>, dan tata cara visualisasi sederhana menggunakan modul pyplot di pustaka matplotlib.</p>

<p style="text-align:justify">Untuk melakukan instalasi ikuti prosedur sebagai berikut:</p>

1. Kunjungi <url>https://docs.conda.io/projects/conda/en/latest/user-guide/install/</url> untuk instalasi Miniconda versi 3.
2. Ikuti prosedur instalasi (jangan lupa atur PATH -nya).
3. Ikuti prosedur instalasi git pada situs:
<url>https://git-scm.com/book/en/v2/Getting-Started-Installing-Git</url>.
4. Buku *Command Line Interface* (CLI), jalankan perintah:
```bash
    git clone git@github.com:sandyherho/NeuroLab-tutor.git
```
5. Lakukan instalasi lingkungan virtual conda dengan menjalankan perintah di folder hasil *cloning* tersebut:
```bash
    conda env create -f environment.yml
```
6. Aktifkan lingkungan virtual dengan perintah:
```bash
    conda activate neuroLabs
```
7. Untuk mengaktifkan Jupyter Notebook, jalankan perintah:
```(bash)
    jupyter notebook
```
8. Jika sudah terbuka di *browser* masing - masing sesi Python interaktif dapat dimulai.
9. Untuk mengakhiri sesi tekan tombol `<CTRL> + C` di CLI, dan jalankan perintah sebagai berikut untuk menonaktifkan lingkungan virtual conda:
```(bash)
    conda deactivate
```
