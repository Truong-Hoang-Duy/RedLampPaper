# 🚨 RedLamp Vibrant Web Studio — Ứng Dụng Demo & Kiểm Chứng Thực Nghiệm KDD '25

Ứng dụng Web SPA tốc độ cao (FastAPI + 60fps HTML5 Retina Canvas) cho bài báo KDD '25:
> **Robust and Explainable Detector of Time Series Anomaly via Augmenting Multiclass Pseudo-Anomalies (RedLamp)**

Kết nối trực tiếp với kho thực nghiệm **`redlamp_retraining/` (15.4 GB — 4 tập Benchmark, 129 thực thể × 5 seeds = 645 bài test, 295 model checkpoints `bestmodel.pkl`)** và bộ mẫu chuẩn bài báo (`demo/sample_results/`).

---

## 1. Cách khởi chạy Web Studio (Khuyên dùng — Giao diện Sáng, Mượt 60fps)

Từ thư mục gốc của project (`RedLampPaper/`), chạy:

```bash
uvicorn demo.server:app --host 0.0.0.0 --port 8080 --reload
```

Sau đó mở trình duyệt tại: **http://localhost:8080**

*(Nếu vẫn muốn chạy phiên bản Streamlit cũ, có thể dùng `streamlit run demo/app.py`).*

---

## 2. 5 Màn hình Tương tác trong Web Studio

1. **📡 Màn hình 1 — Giám sát & Giải phẫu FAA (`60fps Zero-Reload`)**:
   * Kéo thanh trượt **Ngưỡng FAA $\delta$**, **Trọng số MSE**, **Làm mượt $w$** hoặc **Phóng to vùng thời gian** để xem đồ thị 3 tầng biến đổi tức thì ở tốc độ 60 khung hình/giây (không tải lại trang).
   * Nút **🎯 Zoom tới Đỉnh Bất Thường** tự động nhảy tới đúng thời điểm xảy ra sự cố.
2. **🏆 Màn hình 2 — Bảng vàng Thực nghiệm KDD '25 (`4 Datasets × 5 Seeds`)**:
   * Đối sánh trực diện **Ours (5 Seeds Mean ± Std) vs. Bài báo KDD '25 vs. Bản Legacy (31/10)**.
   * Đường cong hội tụ Loss và Bảng xếp hạng 129 thực thể có ô tìm kiếm + nút **🔍 Mở Giám sát** 1-click.
3. **🧭 Màn hình 3 — Không gian Tiềm ẩn 128D (`2D Latent Embedding & XAI`)**:
   * Bản đồ phân cụm 2D từ `enc.npy` (`128 chiều`) và thanh **Chữ ký Hình thái Bất thường** giải thích nguyên nhân vật lý của sự cố.
4. **⚡ Màn hình 4 — Phòng Giả lập Bơm Lỗi Thời gian thực (`Live PyTorch Inference`)**:
   * Bấm chọn 1 trong 11 loại sự cố để tiêm trực tiếp vào cửa sổ thực tế và gọi REST API chạy mô hình `bestmodel.pkl` (`ConvAEC`) trong ~30ms.
5. **🧬 Màn hình 5 — 12 Lớp Biến Đổi & Nhãn Mềm (`Figure 2a & Eq. 1`)**:
   * Tương tác tức thì với thanh trượt $\alpha, \beta$ của công thức Backward Correction và lưới 12 phép biến đổi chuỗi thời gian kèm vùng mặt nạ `anomaly_mask`.
