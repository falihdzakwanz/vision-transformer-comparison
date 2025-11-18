# Vision Transformer Comparison: Swin vs DeiT

Proyek perbandingan performa dua model Vision Transformer (Swin Transformer dan DeiT) pada dataset Indonesian Food Classification dengan 5 kelas.

## 📋 Deskripsi

Tugas eksplorasi ini membandingkan:

- **Swin Transformer**: Hierarchical transformer dengan shifted window attention
- **DeiT (Data-efficient Image Transformer)**: ViT dengan knowledge distillation

Kedua model menggunakan pre-trained weights dari ImageNet dan di-fine-tune pada dataset Indonesian Food.

## 🗂️ Struktur Project

```
transformer_explore/
├── dataset/                   # Dataset Indonesian Food
│   ├── train.csv              # Label file
│   └── train/                 # Image files
├── src/                       # Source code
│   ├── data_preparation.py    # Data loading dan preprocessing
│   ├── train_swin.py          # Training Swin Transformer
│   ├── train_deit.py          # Training DeiT
│   ├── evaluate.py            # Evaluasi model dan metrics
│   └── visualize.py           # Visualisasi hasil
├── models/                    # Saved model weights
├── outputs/                   # Hasil eksperimen
│   ├── figures/              # Visualisasi dan plot
│   └── results/              # Metrics dan logs
├── requirements.txt           # Dependencies
└── README.md                 # Dokumentasi ini
```

## 🚀 Setup dan Instalasi

### 1. Buat Virtual Environment

```powershell
# Buat virtual environment
python -m venv venv

# Aktivasi virtual environment
.\venv\Scripts\Activate.ps1

# Jika ada error execution policy, jalankan:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Verifikasi Dataset

Pastikan struktur dataset sudah benar:

```
dataset/
├── train.csv          # File dengan kolom: filename, label
└── train/            # Folder berisi semua gambar
```

## 📊 Cara Menjalankan

### Step 1: Eksplorasi Data

```powershell
python src/data_preparation.py
```

Output:

- Distribusi kelas
- Sample visualisasi
- Statistik dataset

### Step 2: Training Swin Transformer

```powershell
python src/train_swin.py
```

Model yang digunakan: `swin_tiny_patch4_window7_224` (pre-trained ImageNet)

### Step 3: Training DeiT

```powershell
python src/train_deit.py
```

Model yang digunakan: `deit_tiny_patch16_224` (pre-trained ImageNet)

### Step 4: Evaluasi dan Perbandingan

```powershell
python src/evaluate.py
```

Output:

- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Inference Time
- Parameter Count

### Step 5: Visualisasi Hasil

```powershell
python src/visualize.py
```

Output:

- Learning curves
- Confusion matrices
- Comparison tables
- Sample predictions

## 📈 Metrik Evaluasi

Setiap model dievaluasi berdasarkan:

1. **Jumlah Parameter**

   - Total parameters
   - Trainable parameters
   - Model size (MB)

2. **Performance Metrics**

   - Accuracy
   - Precision (per-class & average)
   - Recall (per-class & average)
   - F1-Score (per-class & average)
   - Confusion Matrix

3. **Inference Time**
   - Average time per image (ms)
   - Throughput (images/second)
   - Total test set time

## ⚙️ Konfigurasi Training

### Hyperparameters

- **Input Size**: 224x224
- **Batch Size**: 32
- **Epochs**: 20
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4
- **Weight Decay**: 0.01
- **Scheduler**: CosineAnnealingLR

### Data Augmentation

- Random Horizontal Flip
- Random Rotation (±10°)
- Color Jitter
- Normalization (ImageNet stats)

## 💻 Hardware

Spesifikasi yang digunakan:

- GPU: [Sesuaikan dengan GPU Anda]
- CPU: [Sesuaikan dengan CPU Anda]
- RAM: [Sesuaikan dengan RAM Anda]

## 📝 Hasil

Hasil lengkap tersimpan di folder `outputs/`:

- `outputs/figures/`: Semua visualisasi
- `outputs/results/`: Metrics dalam format CSV dan JSON
- `models/`: Model weights (best & last checkpoint)

## 🔍 Reproducibility

Untuk hasil yang reproducible:

- Random seed: 42 (set di semua script)
- Gunakan dependencies dari `requirements.txt`
- Jalankan dengan configuration yang sama

## 📚 Referensi

1. **Swin Transformer**: [Liu et al., 2021] - Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
2. **DeiT**: [Touvron et al., 2021] - Training data-efficient image transformers & distillation through attention
3. **PyTorch Image Models (timm)**: https://github.com/huggingface/pytorch-image-models

## 👨‍🎓 Author

- **Nama**: [Nama Anda]
- **NIM**: 122140132
- **Mata Kuliah**: Deep Learning
- **Semester**: Ganjil 2025/2026

## 📄 License

Proyek ini dibuat untuk keperluan tugas akademik.
