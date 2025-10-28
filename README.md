# Deep_Learning_Attention_Lab_OCR_Prototype
Attention variants (Self/MQA/GQA/Sparse/Linear) study + OCR CNN-to-Transformer prototype.

💡 Tổng quan

Notebook gồm 2 phần:

Attention Lab: so sánh Self-Attention, MQA, GQA, Sparse/Strided, Linear/Performer (độ phức tạp, bộ nhớ, tốc độ).

OCR Prototype: kiến trúc CNN/ResNet Encoder + Transformer Decoder cho bài toán Image→Text (nhận dạng ký tự).

🔍 Insight chính

MQA/GQA: chia sẻ K/V giữa heads ⇒ giảm chi phí suy luận khi số head lớn.

Sparse/Strided: phù hợp chuỗi rất dài.

Linear/Performer: gần tuyến tính theo chiều dài chuỗi ⇒ hợp tài nguyên hạn chế.

🧱 OCR: Kiến trúc & huấn luyện

Encoder: ResNet-style CNN trích xuất feature map.

Decoder: Transformer Decoder sinh chuỗi ký tự; so sánh nhanh hướng CRNN/CTC.

Huấn luyện: Adam/SGD, early-stopping, checkpoint “best”.

Đánh giá: loss, accuracy theo ký tự và chuỗi; phân tích lỗi ký tự dễ nhầm.

📊 Kết quả

Attention: MQA/GQA nhanh hơn Multi-Head chuẩn khi seq_len dài & n_heads lớn.

OCR: độ chính xác chuỗi (CER/WER) cải thiện khi tăng augment + beam search.

📝 Ghi chú kỹ thuật

Đặt seed để tái lập.

Log thời gian/bộ nhớ khi so sánh attention.

Với OCR: thêm augmentation (affine/gaussian), beam search, vocab chuyên biệt.
