# Implementasi algoritma jaringan saraf tiruan menggunakan NeuroLab

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![made-with-Markdown](https://img.shields.io/badge/Made%20with-Markdown-1f425f.svg)](http://commonmark.org)<br>
[![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](http://ansicolortags.readthedocs.io/?badge=latest)
[![GitHub issues](https://img.shields.io/github/issues/Naereen/StrapDown.js.svg)](https://GitHub.com/Naereen/StrapDown.js/issues/)
[![No Maintenance Intended](http://unmaintained.tech/badge.svg)](http://unmaintained.tech/)<br>
![GitHub watchers](https://img.shields.io/github/watchers/sandyherho/NeuroLab-tutor?style=social)<br>
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


oleh:

[Sandy H.S. Herho](mailto:sandy.herho@igdore.org) 

<p align=”justify”>Algoritma jaringan saraf tiruan (arificial neural networks : ANN) merupakan bagian penting dari riset dan pengembangan teknologi kecerdasan buatan yang menjadi sendi utama dalam revolusi industri ke-4. Presiden kita Ir. H. Joko Widodo, berkali – kali menyerukan untuk meningkatkan daya saing sumber daya manusia (SDM) Indonesia dalam rangka menyongsong revolusi industri 4.0 ini, namun yang disayangkan hingga kini tutorial yang membahas tentang ANN (yang mana merupakan batu penjuru dari revolusi industri 4.0) dalam bahasa Indonesia masih sangat langka.</p>

<p align=”justify”>Tutorial singkat ini mencoba menjawab tantangan tersebut dengan membahas dasar – dasar implementasi algoritma jaringan saraf tiruan menggunakan pustaka NeuroLab yang berbasis pada bahasa pemrograman Python. Penyusun sengaja menggunakan NeuroLab (alih – alih pustaka Python untuk implementasi ANN yang lebih populer seperti TensorFlow, Keras, dan PyTorch) untuk menyesuaikan dengan kondisi iklim komputasi ilmiah di sistem akademik Indonesia yang menurut hemat penyusun masih didominasi oleh penggunaan MATLAB<sup>®</sup>. Kemiripan sintaks dan alur logika pada NeuroLab dengan <i>Neural Network Toolbox</i> (NNT) diharapkan dapat menarik minat sidang pembaca untuk beralih dari sistem komputasi komersial ke sistem komputasi bersumber terbuka.</p>

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
Penerbit:

<img src="wcpl.png" alt="wcpl-itb" width="200"/>
