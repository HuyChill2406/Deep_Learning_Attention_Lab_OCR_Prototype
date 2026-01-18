# Deep Learning Assignments: Attention (LLMs) & OCR with ResNet50 + Transformer

Repo này chứa notebook **`DL_522h0131_FINAL.ipynb`**, bao gồm hai phần chính:

1. **Bài 1 – Tìm hiểu chuyên sâu cơ chế Attention trong Large Language Models (LLMs)**
2. **Bài 2 – Xây dựng hệ thống OCR kết hợp CNN Encoder và Transformer Decoder**
   - EasyOCR cho bước phát hiện vùng chữ (text detection)
   - ResNet-50 làm encoder trích xuất đặc trưng ảnh
   - Transformer Decoder để nhận dạng chuỗi ký tự (text recognition)
   - Pipeline huấn luyện, đánh giá và suy luận hoàn chỉnh

---

## Tổng quan Notebook

### Phần 1 – Attention trong LLMs (Lý thuyết)

Phần này tập trung trình bày và phân tích các cơ chế Attention được sử dụng trong Transformer và LLM hiện đại, bao gồm:
- Self-Attention: trực giác, công thức tính toán, ưu và nhược điểm
- Phân tích độ phức tạp tính toán và khả năng mở rộng
- So sánh Attention cơ bản với Attention trong Transformer tiêu chuẩn
- Giới thiệu các biến thể Attention hiện đại được sử dụng trong LLMs

Phần này mang tính **lý thuyết – học thuật**, có hình minh họa và bảng so sánh để làm rõ cơ chế hoạt động.

---

### Phần 2 – OCR với ResNet-50 Encoder và Transformer Decoder

**Mục tiêu:** xây dựng một hệ thống OCR có khả năng nhận dạng văn bản từ ảnh theo cách tiếp cận chuỗi–chuỗi (sequence-to-sequence).

#### Dataset
- Sử dụng bộ dữ liệu từ Hugging Face:
  - `priyank-m/MJSynth_text_recognition`
- Chia thành tập huấn luyện và tập validation với kích thước giới hạn để phù hợp tài nguyên.

#### Tiền xử lý dữ liệu
- Chuẩn hóa văn bản:
  - Unicode normalization
  - Loại bỏ dấu và ký tự đặc biệt
- Xây dựng từ điển ký tự (character-level vocabulary) từ tập train
- Mã hóa chuỗi ký tự thành token IDs với các token đặc biệt:
  - `<PAD>`, `<UNK>`, `<SOS>`, `<EOS>`
- Xử lý ảnh:
  - Resize ảnh về kích thước cố định
  - Data augmentation cho tập train (xoay, affine, perspective, color jitter, blur)

#### Kiến trúc mô hình
- **Encoder:** ResNet-50 (trích xuất đặc trưng ảnh)
  - Feature map được chiếu từ 2048 → 512
- **Decoder:** Transformer Decoder
  - Embedding + Sinusoidal Positional Encoding
  - Causal mask để đảm bảo dự đoán theo thứ tự thời gian
- Áp dụng teacher forcing với xác suất giảm dần theo epoch

#### Huấn luyện
- Optimizer: AdamW
- Learning rate scheduler: OneCycleLR
- Mixed precision training (GradScaler)
- Gradient clipping
- Early stopping (patience = 8)
- Lưu checkpoint tốt nhất (`best.pt`)

#### Suy luận (Inference)
- Greedy decoding ở mức ký tự
- Tích hợp EasyOCR:
  - Phát hiện vùng chữ
  - Cắt/hiệu chỉnh vùng ảnh
  - Đưa vào mô hình nhận dạng

#### Trực quan hóa
- Biểu đồ loss và accuracy cho train/validation
- Minh họa kết quả dự đoán trên tập validation
- Visualization attention heatmap (trung bình/tổng attention weights)

---

## Yêu cầu môi trường

- Python 3.9+
- Khuyến nghị chạy trên GPU (Google Colab hoặc máy có CUDA)

Các thư viện chính:
- `torch`, `torchvision`
- `numpy`
- `opencv-python`
- `matplotlib`
- `datasets`
- `easyocr`
- `Pillow`

---

## Cách chạy

### Cách 1: Google Colab (khuyến nghị)
1. Upload `DL_522h0131_FINAL.ipynb` lên Google Colab
2. Chuyển Runtime sang GPU (nếu có)
3. Chạy các cell theo thứ tự từ trên xuống

### Cách 2: Chạy local
```bash
pip install torch torchvision torchaudio
pip install numpy opencv-python matplotlib pillow datasets easyocr
jupyter notebook DL_522h0131_FINAL.ipynb

Deep Learning Assignments: Attention (LLMs) & OCR with ResNet50 + Transformer

This repository contains the Jupyter Notebook DL_522h0131_FINAL.ipynb, which includes two main parts:

Assignment 1 – In-depth study of Attention mechanisms in Large Language Models (LLMs)

Assignment 2 – OCR system using a CNN Encoder and Transformer Decoder

EasyOCR for text detection

ResNet-50 as the image encoder

Transformer Decoder for sequence-based text recognition

Complete training, evaluation, and inference pipeline

Notebook Overview
Part 1 – Attention in LLMs (Theory)

This section provides a structured and in-depth explanation of modern Attention mechanisms, including:

Self-Attention: intuition, formulation, strengths, and limitations

Computational complexity and scalability analysis

Comparison between simplified attention and standard Transformer attention

Overview of modern Attention variants used in LLMs

This part is theoretical, supported by diagrams and comparisons.

Part 2 – OCR with ResNet-50 Encoder and Transformer Decoder

Objective: build a sequence-to-sequence OCR system for text recognition from images.

Dataset

Hugging Face dataset:

priyank-m/MJSynth_text_recognition

Subsampled for efficient training.

Data Preprocessing

Text normalization:

Unicode normalization

Accent and special character removal

Character-level vocabulary construction

Encoding text sequences with special tokens:

<PAD>, <UNK>, <SOS>, <EOS>

Image preprocessing and augmentation for training

Model Architecture

Encoder: ResNet-50 for visual feature extraction

Feature projection from 2048 → 512

Decoder: Transformer Decoder with positional encoding and causal masking

Teacher forcing with scheduled probability decay

Training

Optimizer: AdamW

Scheduler: OneCycleLR

Mixed precision training

Gradient clipping

Early stopping (patience = 8)

Best model checkpoint saved as best.pt

Inference

Greedy character-level decoding

Integration with EasyOCR for detection + recognition

Visualization

Training and validation curves

OCR prediction samples

Attention heatmap visualization

Requirements

Python 3.9+

GPU recommended

Main libraries:

PyTorch

OpenCV

NumPy

Matplotlib

Hugging Face Datasets

EasyOCR

How to Run
Option 1: Google Colab (recommended)

Upload the notebook

Enable GPU runtime

Run cells sequentially

Option 2: Local
pip install torch torchvision torchaudio
pip install numpy opencv-python matplotlib pillow datasets easyocr
jupyter notebook DL_522h0131_FINAL.ipynb
