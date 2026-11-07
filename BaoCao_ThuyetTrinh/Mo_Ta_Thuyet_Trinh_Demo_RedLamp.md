# 🎙️ TÀI LIỆU MÔ TẢ & KỊCH BẢN THUYẾT TRÌNH HỆ THỐNG DEMO REDLAMP

**Ứng dụng:** `RedLamp Vibrant Web Studio` (`demo/server.py` + `demo/static/index.html` + `demo/app.py`)  
**Bài báo gốc:** *Robust and Explainable Detector of Time Series Anomaly via Augmenting Multiclass Pseudo-Anomalies (KDD '25)*  

---

## 1. Tổng quan Kiến trúc & Cách Khởi chạy Demo

### 1.1. Mục tiêu của bản Demo trong buổi Bảo vệ / Thuyết trình
Bản Demo được thiết kế để **chứng minh trực quan và tương tác thời gian thực** toàn bộ các luận điểm trong phần **Problem Statement (Duy)**, **Methodology (Bảo)** và **Experiments & Results (QA + Danh)**:
1. Giải quyết **Diversity Gap** bằng lưới 12 phép biến đổi chuỗi thời gian chuyên biệt (*Màn hình 5*).
2. Giải quyết **Label Overconfidence & Anomaly Contamination** bằng cơ chế làm mềm nhãn *Backward Correction* và tái tạo có mặt nạ *Masked Reconstruction Loss* (*Màn hình 4 & 5*).
3. Giải quyết **False Anomalies (Pseudo-Normal Class)** bằng bộ lọc hậu kiểm *Frequent Anomaly Adjustment - FAA* (*Màn hình 1*).
4. Cung cấp **Khả năng giải thích (Explainability - XAI)** thông qua phân cụm không gian tiềm ẩn 128 chiều (*Màn hình 3*) và kiểm chứng thực nghiệm trên **129 thực thể × 5 seeds** (*Màn hình 2*).

### 1.2. Kiến trúc Kỹ thuật của thư mục `demo/`
* **Backend (`demo/server.py`):** Xây dựng bằng **FastAPI**, cung cấp 3 API chính:
  * `GET /api/init`: Nạp danh mục 4 bộ dữ liệu thực nghiệm (`AIOps`, `SMD`, `MSL`, `SMAP`) + bộ mẫu `UCR Demo`, bảng đối sánh tổng hợp (`summary_*.csv`) và bảng xếp hạng 129 thực thể (`metrics_*.csv`).
  * `GET /api/run`: Trả về chuỗi tín hiệu đầu vào, điểm thành phần $s_{\text{MSE}}(t)$, xác suất 12 lớp của Classifier, lịch sử hội tụ Loss và tọa độ phân cụm 2D (từ `enc.npy` 128 chiều) với độ trễ `< 3ms`.
  * `GET /api/inject`: Thực hiện tiêm trực tiếp 1 trong 11 loại bất thường vào cửa sổ dữ liệu thực tế và gọi mô hình PyTorch `ConvAEC` (`bestmodel.pkl`) chạy suy luận trực tiếp (*Live Inference*).
* **Frontend (`demo/static/index.html`):** Giao diện Single-Page Application (SPA) vẽ đồ thị bằng **HTML5 Retina Canvas ở tốc độ 60fps**, cho phép kéo các thanh trượt siêu tham số và thấy ngay sự thay đổi của đường điểm bất thường mà không cần tải lại trang (*Zero-Reload*).
* **Gói dữ liệu di động (`demo/portable_data/`):** Đóng gói sẵn kết quả nén `.npz` và trọng số `bestmodel.pkl` của cả 4 tập dữ liệu giúp chạy mượt trên bất kỳ máy tính thuyết trình nào mà không cần ổ cứng chứa 15.4 GB dữ liệu thô.

### 1.3. Lệnh Khởi chạy Nhanh
Từ thư mục gốc `d:\\Project\\RedLampPaper`, chạy lệnh:
```bash
uvicorn demo.server:app --host 0.0.0.0 --port 8080 --reload
```
Sau đó mở trình duyệt tại: **`http://localhost:8080`**

---

## 2. Bố cục Điều khiển Trung tâm (Top Bar & KPI Strip)

Trên thanh Header cố định ở đầu trang, người thuyết trình có thể thao tác bất kỳ lúc nào:
1. **Bộ chọn 3 cấp (`Miền` $\to$ `Thực thể` $\to$ `Seed S0..S4`):**
   * **AIOps (29 KPIs):** Chuỗi đơn biến ($D=1$) giám sát dịch vụ đám mây.
   * **SMD (28 Servers):** Chuỗi đa biến ($D=38$) giám sát máy chủ công nghiệp.
   * **NASA MSL (23 Kênh) & NASA SMAP (49 Kênh):** Chuỗi đa biến ($D=55$ và $D=25$) từ tàu tự hành Sao Hỏa và vệ tinh NASA.
   * **UCR Demo (#028 ABP & #040 EPG):** Hai chuỗi mẫu kinh điển trong Figure 6 của bài báo KDD '25.
2. **3 Thanh trượt Thời gian thực (60fps Sliders):**
   * **`FAA δ` (`0.01 -> 0.30`, mặc định `0.05`):** Ngưỡng lọc lớp giả bình thường (*Công thức 6*).
   * **`Trọng số MSE` (`0.00 -> 1.00`, mặc định `0.50`):** Tỷ lệ pha trộn giữa điểm tái tạo Autoencoder $s_{\text{MSE}}(t)$ và điểm phân loại Classifier $s_{\text{CE}}(t)$ (*Công thức 8*).
   * **`Làm mượt w` (`1 -> 100`, mặc định `50`):** Cửa sổ trung bình trượt làm mượt chuỗi điểm.
3. **5 Thẻ KPI Tóm tắt:** Hiển thị thông số thực thể `(T, D, W=100)`, `VUS-ROC / R-AUC-ROC`, `R-AUC-PR / RF (Oracle)`, **Hình thái Đỉnh** (lớp bệnh lý chiếm ưu thế) và **Lớp FAA Lọc** (số lớp *false anomaly* bị FAA gạch bỏ).

---

## 3. Chi tiết 5 Màn hình & Kịch bản Thao tác Thuyết trình

### 📡 Màn hình 1: Giám sát & Giải phẫu FAA (`1. Giám sát & FAA`)
* **Ý nghĩa học thuật:** Minh họa trực tiếp Công thức (5)–(8) và cách bộ lọc **FAA** xử lý triệt để hiện tượng **False Anomalies (Pseudo-normal class)**.
* **Bố cục hiển thị:**
  1. **Đồ thị đồng bộ 3 tầng:**
     * *Tầng 1:* Tín hiệu đầu vào $X(t)$ và vùng nền đỏ đánh dấu thời điểm xảy ra lỗi thực tế (Ground-Truth).
     * *Tầng 2:* Đặt cạnh nhau điểm sai số tái tạo $s_{\text{MSE}}(t)$ (xanh lam) và điểm phân loại $s_{\text{CE}}(t)$ (tím).
     * *Tầng 3:* Điểm tổng hợp $s(t)$ đặt chồng giữa **Có FAA (đỏ hồng)** và **Tắt FAA (xám)**.
  2. **Biểu đồ cột ngang (Trái dưới):** Tần suất xuất hiện trung bình của 11 lớp bất thường so với vạch đứt ngưỡng `FAA δ`. Lớp nào vượt ngưỡng sẽ bị tô xám và gạch điểm về `0`.
  3. **Biểu đồ đường (Phải dưới):** Diễn biến xác suất của 11 lớp bất thường theo thời gian.
* **🗣️ Kịch bản thao tác & Lời dẫn thuyết trình:**
  1. Chọn `SMD -> machine-1-1` (hoặc `UCR Demo -> 028_UCR_ABP_Demo`), bấm nút **`[Zoom vùng đỉnh]`**.
  2. Chỉ vào Tầng 3 và kéo thanh trượt **`FAA δ`** từ `0.30` (tắt lọc) xuống `0.05` (mặc định):
     > *"Như nhóm đã trình bày ở phần Problem Statement, một số phép biến đổi ngẫu nhiên có thể tạo ra các mẫu trông vẫn bình thường (False Anomalies), khiến Classifier dự đoán lớp đó xuất hiện liên tục trên tập Test (đường màu xám ở Tầng 3 bị nhiễu cao khắp nơi). Khi ta đưa ngưỡng FAA về 0.05, các lớp vượt vạch đỏ ở biểu đồ góc trái dưới lập tức bị nhận diện là lớp giả bình thường và bị quy điểm về 0. Nhờ đó, đường điểm Có FAA (màu đỏ) triệt tiêu hoàn toàn báo động giả và chỉ dựng đứng tại đúng vùng lỗi thật."*

---

### 🏆 Màn hình 2: Bảng vàng Thực nghiệm KDD '25 (`2. Kết quả KDD '25`)
* **Ý nghĩa học thuật:** Chứng minh tính chính xác và khả năng tái lập thực nghiệm (**Q1: Effectiveness**) trên 4 bộ dữ liệu lớn với 5 lần chạy độc lập (`5 seeds`).
* **Bố cục hiển thị:**
  1. **Biểu đồ cột Đối sánh (Trái trên):** So sánh `Ours (5 Seeds Mean ± Std)` với `Bài báo KDD '25` và `Bản Legacy (31/10)` trên các thước đo `VUS-ROC`, `VUS-PR`, `R-AUC-ROC`, `R-AUC-PR`, `RF`.
  2. **Đồ thị Hội tụ Huấn luyện (Phải trên):** Đường cong giảm hàm mất mát $\mathcal{L}_{\text{MSE}}$ (Train/Valid) và $\mathcal{L}_{\text{CE}}$ qua từng epoch.
  3. **Bảng Xếp Hạng 129 Thực Thể (Phía dưới):** Có ô tìm kiếm trực tiếp và nút **`[Mở →]`** ở từng dòng.
* **🗣️ Kịch bản thao tác & Lời dẫn thuyết trình:**
  1. Bấm chuyển qua lại giữa 4 tab `AIOps`, `SMD`, `MSL`, `SMAP` để hội đồng quan sát kết quả tái lập sát với bài báo KDD '25.
  2. Gõ tìm một thực thể bất kỳ trong bảng xếp hạng và bấm **`[Mở →]`** để nhảy trực tiếp về Màn hình 1 của thực thể đó.

---

### 🧭 Màn hình 3: Không gian Tiềm ẩn 128D & Giải thích XAI (`3. Không gian Tiềm ẩn`)
* **Ý nghĩa học thuật:** Minh họa đóng góp **Explainability (Q5 — Figure 6)**. Giải quyết điểm yếu "hộp đen" của các phương pháp trước đó.
* **Bố cục hiển thị:**
  1. **Bản đồ phân cụm 2D (Trái):** Chiếu vector `enc.npy` (128 chiều) xuống không gian 2 chiều. Các chấm tròn màu là các lớp hình thái; các dấu **$\times$ Đỏ (`Đỉnh Dị Thường`)** là các cửa sổ lỗi thực tế.
  2. **Tỷ lệ Hình thái tại Vùng Đỉnh & Hộp Kết luận XAI (Phải):** Xếp hạng xác suất phân loại tại vùng đỉnh và tự động xuất nhận định giải thích bản chất vật lý của sự cố.
* **🗣️ Kịch bản thao tác & Lời dẫn thuyết trình:**
  1. Chọn `UCR Demo -> 040_UCR_EPG_Demo` hoặc `028_UCR_ABP_Demo`.
  2. Chỉ vào đám mây điểm bên trái và hộp XAI bên phải:
     > *"Vì Encoder của RedLamp được huấn luyện để phân biệt 12 lớp hình thái khác nhau, không gian tiềm ẩn 128 chiều tự động phân tách thành các cụm có ý nghĩa vật lý. Các dấu X màu đỏ (vùng lỗi thực tế trên tập Test) nằm sát cụm lớp nào sẽ giải thích ngay lỗi đó thuộc dạng gì: ví dụ chuỗi #040 EPG tụ về cụm Speedup (biến đổi tần số/chu kỳ), còn chuỗi #028 ABP tụ về cụm Noise (nhiễu cao tần)."*

---

### ⚡ Màn hình 4: Giả lập Bơm Lỗi Thời gian thực (`4. Giả lập Bơm Lỗi`)
* **Ý nghĩa học thuật:** Trình diễn suy luận thời gian thực (**Live PyTorch Inference**) bằng trọng số `bestmodel.pkl` và chứng minh tác dụng của **Masked Reconstruction Loss ($\mathcal{L}_{\text{MSE}}$ có mặt nạ $W$)**.
* **Bố cục hiển thị:**
  1. **11 nút chọn kiểu biến đổi** (`spike`, `flip`, `speedup`, `noise`, `cutoff`, `average`, `scale`, `wander`, `contextual`, `upsidedown`, `mixture`) + thanh trượt chọn **Vị trí cửa sổ** và **Seed**.
  2. **3 Thẻ đo tức thì:** `MSE Gốc` (trước tiêm), `MSE Sau Tiêm` (tăng mạnh sau tiêm) và `Dự đoán Classifier` (tên lớp dự đoán + % độ tự tin).
  3. **Đồ thị sóng 3 đường (Trái) & Xác suất 12 lớp Trước/Sau tiêm (Phải):** Vẽ chồng tín hiệu gốc $Z(t)$, tín hiệu sau tiêm $Y(t)$ (đỏ hồng + vùng mặt nạ hồng) và tín hiệu tái tạo $\hat{X}(t)$ (xanh dương).
* **🗣️ Kịch bản thao tác & Lời dẫn thuyết trình:**
  1. Bấm lần lượt vào các nút **`[spike]`**, **`[cutoff]`**, **`[wander]`**.
  2. Chỉ vào đường **Tái tạo $\hat{X}(t)$ (màu xanh dương)** trên đồ thị bên trái:
     > *"Khi chúng ta tiêm trực tiếp một lỗi Spike hay Cutoff vào cửa sổ dữ liệu (đường màu đỏ), hãy quan sát đường tái tạo màu xanh dương của Decoder: nó vẫn đi theo quỹ đạo bình thường ban đầu chứ không vẽ hùa theo đoạn lỗi. Đó là nhờ mặt nạ bất thường W đã che vùng lỗi trong lúc huấn luyện (Công thức 2). Nhờ vậy, khoảng cách giữa đường đỏ và đường xanh tạo ra sai số MSE tăng vọt, đồng thời Classifier bên phải lập tức nhận diện chính xác loại lỗi vừa bơm vào."*

---

### 🧬 Màn hình 5: 12 Phép Biến Đổi & Nhãn Mềm (`5. 12 Biến Đổi & Nhãn Mềm`)
* **Ý nghĩa học thuật:** Minh họa trực tiếp **Algorithm 1 (Figure 2a)** lấp đầy *Diversity Gap* và **Công thức (1) — Backward Correction** trị *Label Overconfidence & Anomaly Contamination*.
* **Bố cục hiển thị:**
  1. **Thanh trượt hệ số $\alpha$ và $\beta$ + Biểu đồ 12 cột Nhãn mềm $\tilde{y}$:** Cho phép kéo thử $\alpha \in [0, 0.30]$ và $\beta \in [0, 0.04]$ để xem phân bố nhãn mềm thay đổi trực tiếp.
  2. **Lưới 12 Đồ thị Biến đổi (Grid 12 Augmentations):** Vẽ trực tiếp 12 lớp trên cửa sổ hiện tại kèm dải màu hồng `anomaly_mask (W=0)`. Bấm vào ô nào sẽ xem phân bố nhãn mềm của lớp đó.
* **🗣️ Kịch bản thao tác & Lời dẫn thuyết trình:**
  1. Bấm vào các ô trên lưới 12 phép biến đổi để hội đồng thấy sự khác biệt hình thái giữa `Spike`, `Flip`, `Speedup`, `Wander`, `Contextual`...
  2. Kéo thanh trượt **`Hệ số α`** từ `0.00` lên `0.10`:
     > *"Nếu dùng nhãn cứng One-hot (α = 0, β = 0), mô hình sẽ tin tuyệt đối 100% rằng mọi mẫu biến đổi đều là bất thường. Khi áp dụng Backward Correction với α = 0.10 và β = 0.01, nhãn của lớp đang chọn giảm xuống 0.79 và luôn chừa lại 0.11 (11%) xác suất cho lớp Normal. Đây chính là chìa khóa giúp RedLamp không bị quá khớp vào các False Anomalies và chống chịu cực tốt khi dữ liệu huấn luyện bị nhiễm bẩn."*

---

## 4. Gợi ý Thứ tự Trình bày Demo Mượt Nhất (5 Phút)

Để mạch thuyết trình khớp tự nhiên nhất với luồng bài báo (**Problem Statement $\to$ Methodology $\to$ Experiments**), người demo có thể đi theo trình tự:
1. **Phút 1 — Mở Màn hình 5 (`12 Biến Đổi & Nhãn Mềm`):** Nối tiếp ngay sau phần lý thuyết để chỉ cho hội đồng thấy 12 phép biến đổi lấp đầy *Diversity Gap*, mặt nạ hồng `anomaly_mask` và thanh trượt nhãn mềm $\alpha, \beta$.
2. **Phút 2 — Mở Màn hình 4 (`Giả lập Bơm Lỗi`):** Bấm tiêm lỗi trực tiếp để chứng minh mô hình PyTorch `bestmodel.pkl` chạy thời gian thực, đường tái tạo xanh dương giữ nguyên dáng bình thường và Classifier bắt trúng lớp lỗi.
3. **Phút 3 & 4 — Mở Màn hình 1 (`Giám sát & FAA`):** Bấm `Zoom vùng đỉnh` và kéo thanh trượt `FAA δ` để trình diễn sự khác biệt giữa **Tắt FAA** và **Có FAA** (tiêu diệt *False Anomalies*).
4. **Phút 5 — Mở Màn hình 3 (`Không gian Tiềm ẩn`) & Màn hình 2 (`Kết quả KDD '25`):** Chỉ ra khả năng giải thích XAI qua dấu $\times$ đỏ trên bản đồ 2D và chốt lại bằng bảng tổng kết thực nghiệm 645 lượt chạy.
