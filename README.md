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
