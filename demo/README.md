# 🚨 RedLamp Interactive Studio — Demo Đồ Án Cuối Kỳ CS2207

Thư mục này chứa bản Demo tương tác cho bài báo KDD '25:
> **Robust and Explainable Detector of Time Series Anomaly via Augmenting Multiclass Pseudo-Anomalies (RedLamp)**

---

## 1. Cách khởi chạy nhanh (1 câu lệnh)

Từ thư mục gốc của project (`RedLamp/`), chạy:

```bash
streamlit run demo/app.py
```

*(Lần đầu khởi chạy, nếu chưa có dữ liệu trong `./demo/sample_results`, ứng dụng sẽ tự động gọi `demo/generate_sample_data.py` để huấn luyện nhanh 2 bộ mẫu `028_UCR_ABP_Demo` và `040_UCR_EPG_Demo` trong vài giây).*

---

## 2. Kết nối với dữ liệu đã chạy thử của nhóm (từ Google Drive)

Ứng dụng `demo/app.py` được thiết kế để **tự động quét toàn bộ thư mục `./result/`**.
Khi bạn tải thư mục kết quả từ Google Drive về máy và đặt vào:

```text
RedLamp/
└── result/
    └── <run_name>/
        └── <dataset>/
            └── <entity>/
                └── d1_b128_w100_s1/
                    └── 0/
                        ├── bestmodel.pkl
                        └── test_all/
                            ├── input.npy
                            ├── pred.npy
                            ├── pred_label.npy
                            ├── enc.npy
                            └── anomaly_score.npy
```

Chỉ cần tải lại trang Streamlit (`R`), toàn bộ các thực thể bạn đã chạy trên Google Drive sẽ xuất hiện ngay trong menu **📂 Chọn bộ kết quả thực nghiệm** ở thanh bên trái!

---

## 3. Kịch bản Demo 5 phút gợi ý khi thuyết trình (30–35 phút)

1. **Phút 1 – Tab 1 (Multiclass Data Augmentation & Soft Labels):**
   * Trình chiếu lưới 12 phép biến đổi chuỗi thời gian (tái hiện sống động **Figure 2a**).
   * Giải thích vùng tô đỏ (`anomaly_mask`) giúp Autoencoder không học nhầm vùng bất thường, và thanh trượt $\alpha, \beta$ biến nhãn cứng thành nhãn mềm (**Backward Correction – Eq. 1**).
2. **Phút 2–3 – Tab 2 (Giải phẫu Điểm Bất thường & FAA):**
   * Chọn bộ `028_UCR_ABP_Demo` (hoặc `UCR #028` thật từ Drive).
   * Chỉ vào biểu đồ tần suất lớp để thầy và các bạn thấy lớp `Average` xuất hiện vượt ngưỡng $\delta = 0.05$ trên toàn chuỗi test (hiện tượng *Pseudo-Normal Class*).
   * Kéo thanh trượt $\delta$ để chứng minh: khi bật **FAA**, đường điểm bất thường lọc sạch nhiễu nền và chỉ nhô cao đúng tại vị trí Ground Truth!
3. **Phút 4 – Tab 3 (Explainable Latent Space – Figure 6):**
   * Mở biểu đồ **t-SNE 2D** để cho thấy các điểm `Anomaly (Test)` (dấu `X` màu đỏ) tụ lại ngay sát cụm giả bất thường tương ứng (`Noise`/`Spike` với chuỗi ABP, hoặc `Speedup` với chuỗi EPG), giúp giải thích **nguyên nhân/hình thái** của bất thường.
4. **Phút 5 – Tab 4 (Live Anomaly Injection khi hỏi đáp phản biện):**
   * Mời thầy/bạn học chọn ngẫu nhiên 1 loại bất thường (ví dụ `Cutoff`, `Spike`, `Flip`), bấm bơm trực tiếp vào cửa sổ test bình thường và xem `bestmodel.pkl` lập tức phát hiện đúng sai số tái tạo lẫn tên loại bất thường.

