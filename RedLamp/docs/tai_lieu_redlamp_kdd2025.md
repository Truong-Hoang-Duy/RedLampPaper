# TÀI LIỆU NGHIÊN CỨU TOÀN DIỆN & SỔ TAY HƯỚNG DẪN TÁI LẬP THỰC NGHIỆM
**Bài báo:** *Robust and Explainable Detector of Time Series Anomaly via Augmenting Multiclass Pseudo-Anomalies (REDLAMP)*  
**Tác giả:** Kohei Obata, Yasuko Matsubara, Yasushi Sakurai (SANKEN, University of Osaka, Japan)  
**Hội nghị:** Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD '25)  
**Mã bài báo:** arXiv:2505.20765v1 | **DOI:** 10.1145/3711896.3737110 | **GitHub:** `https://github.com/KoheiObata/RedLamp`

---

## PHẦN 1: TỪ ĐIỂN KHÁI NIỆM NỀN TẢNG CHO NGƯỜI MỚI BẮT ĐẦU
*(Phần này tổng hợp các khái niệm thực tế đã được giải nghĩa chi tiết để làm nền tảng trước khi đọc công thức toán học)*

### 1.1. Cấu trúc Bộ dữ liệu Chuỗi thời gian (Dataset Structure)
* **Số tập dữ liệu con (`# Subdatasets`):** Một bộ dữ liệu lớn (như SMD, UCR) không phải là 1 bảng duy nhất mà là tập hợp của nhiều thực thể độc lập.
  * *Ví dụ:* Bộ **SMD** có `28 tập con` tương ứng với **28 máy chủ (servers)** khác nhau. Bộ **UCR** có `250 tập con` từ 9 lĩnh vực (nhịp tim ECG, huyết áp, nhiệt độ...). Khi huấn luyện trên UCR, AIOps và SMD, ta phải huấn luyện mô hình riêng cho từng tập con rồi lấy trung bình kết quả.
* **Số chiều đặc trưng (`# Features` - $D$):** Số lượng cột dữ liệu (cảm biến/chỉ số) được theo dõi cùng lúc tại một thời điểm (sau khi đã bỏ cột thời gian `timestamp` và cột nhãn `label`).
  * **Đơn biến (Univariate, $D = 1$):** Chỉ theo dõi 1 cột giá trị duy nhất theo thời gian (ví dụ: UCR, AIOps).
  * **Đa biến (Multivariate, $D > 1$):** Theo dõi nhiều cột cảm biến cùng lúc (ví dụ: SMD có $D = 38$ chỉ số CPU, RAM, Disk...; SMAP có $D = 55$; MSL có $D = 25$).
* **Độ dài chuỗi Train trung bình (`Sequence Length` - $T$):** Số lượng điểm thời gian (số dòng dữ liệu) trung bình trong tập huấn luyện của 1 tập con.
  * *Quy đổi ra thời gian thực:* Với SMD ($T = 25,300$ điểm), tần suất thu thập là 1 phút/điểm $\implies 25,300 \text{ phút} \approx 17.5 \text{ ngày}$ theo dõi liên tục. Với nhịp tim ECG (250 điểm/giây), chuỗi $30,522$ điểm tương đương khoảng $2$ phút đo.

### 1.2. Cửa sổ trượt (Sliding Window)
* **Bản chất:** Mạng Học sâu không thể nuốt trọn chuỗi dài hàng chục nghìn điểm cùng lúc. Ta dùng một khung nhìn có độ dài cố định $L = 100$ điểm thời gian (`window_size = 100`) trượt dọc theo chuỗi dài $T$ với bước nhảy $S$ (`stride` / `window_step`).
* **Ví dụ thời gian thực:** Giám sát máy chủ mỗi phút 1 điểm với cửa sổ $L = 5$, bước nhảy $S = 1$:
  * Cửa sổ 1: `[08:00 -> 08:04]`
  * Cửa sổ 2 (trượt 1 phút): `[08:01 -> 08:05]`
* **Ý nghĩa:** Giúp mô hình nhìn thấy **ngữ cảnh quá khứ** để phát hiện bất thường ngữ cảnh (Contextual Anomaly), đồng thời biến ma trận 2D `(T, D)` thành Tensor 3D `(N, D, L)` với số mẫu $N = \lfloor \frac{T - L}{S} \rfloor + 1$.

### 1.3. Quy trình "5 lần chạy độc lập" (`5 runs`)
* Trong Deep Learning, việc khởi tạo trọng số mạng ban đầu, trộn dữ liệu (`shuffle`), lớp `Dropout` và chọn vị trí chèn bất thường giả đều mang tính ngẫu nhiên.
* Để chứng minh kết quả không phải do "ăn may", tác giả chạy toàn bộ quá trình Train & Test lặp lại **5 lần với 5 hạt giống ngẫu nhiên (`random seed`) khác nhau**, sau đó lấy trung bình cộng kết quả của cả 5 lần.

### 1.4. Phân biệt Mô hình (Encoder/Decoder/Classifier) và Không gian ẩn (`Embedding`)
* **Encoder, Decoder, Classifier:** Là các khối **mạng nơ-ron (Mô hình)** chứa các tầng tích chập (1D-CNN) hoặc tầng tuyến tính (MLP) có trọng số học được.
* **Embedding (Vector biểu diễn ẩn):** **Không phải là mô hình** và cũng không phải hàm chuyển chữ thành số. Khi đưa đoạn sóng thô kích thước $D \times 100$ đi qua bộ mã hóa **Encoder**, Encoder sẽ "ép" đoạn sóng đó thành một dãy gồm đúng **128 con số thực** (`shape: (128,)`). Dãy 128 con số tóm tắt toàn bộ tinh túy/triệu chứng của đoạn sóng này chính là **Embedding**.
* Từ cùng một "tờ phiếu tóm tắt" Embedding 128 số này:
  1. Đưa lên **Classifier ("Bác sĩ chẩn đoán"):** Đoán xem đoạn sóng thuộc loại nào trong 12 lớp.
  2. Đưa xuống **Decoder ("Họa sĩ phục chế"):** Vẽ khôi phục lại đoạn sóng bình thường kích thước $D \times 100$.

### 1.5. Tập mặt nạ bất thường (`Anomaly Mask` - $W$)
* **Bản chất:** Là một ma trận nhị phân chỉ gồm số `0` và `1` có cùng kích thước $D \times L$ với cửa sổ dữ liệu, do **thuật toán tăng cường dữ liệu tự tạo ra** khi chèn lỗi giả.
  * `W = 1`: Đánh dấu đúng tọa độ vừa bị chèn bất thường giả vào.
  * `W = 0`: Đánh dấu vùng dữ liệu gốc bình thường không bị chạm tới.
* **Tác dụng cốt lõi:** Dùng trong hàm mất mát tái tạo $\mathcal{L}_{\text{MSE}}$ thông qua phép nhân với $(1 - W)$. Những chỗ có lỗi giả ($W = 1 \implies 1 - W = 0$) sẽ bị nhân với `0` để **che mắt nhánh Decoder**, ngăn không cho Decoder học cách vẽ lại các cục lỗi giả. Nhờ đó, Decoder chỉ học cách vẽ dữ liệu bình thường; khi gặp lỗi thật ở tập Test, nó sẽ vẽ sai lệch lớn và phát hiện được bất thường.

### 1.6. Hiện tượng "Bệnh giả vô hại" (`False Anomaly`) và Lý do cần Nhãn mềm + FAA
* Khi thuật toán cắt ghép ngẫu nhiên (ví dụ phép `Cutoff` hoặc `Average` rơi vào đoạn quá ngắn chỉ 1-2 điểm như `[70, 72, 71, 71, 72, 71]` so với gốc `[70, 72, 71, 70, 72, 71]`, hoặc làm mượt trên chuỗi có chu kỳ quá dài), mẫu sinh ra **mang tiếng là "Bất thường giả" nhưng thực chất nhìn chẳng khác gì "Bình thường"**. Bài báo gọi đây là **False Anomaly**.
* Nếu ép mô hình học nhãn cứng `100% Bất thường` cho các mẫu này, mô hình sẽ bị rối loạn và báo động giả liên tục. REDLAMP giải quyết bằng 2 chốt chặn:
  1. **Lúc Huấn luyện — Nhãn mềm (`Backward Correction`):** Không bao giờ tin 100% vào nhãn. Luôn chừa lại xác suất $\alpha = 0.1$ ($10\%$) rằng mẫu bất thường giả thực chất vẫn là mẫu bình thường, và $\beta = 0.01$ ($1\%$) xác suất nhầm lẫn giữa các lớp.
  2. **Lúc Kiểm thử — Bộ lọc `FAA` (`Frequent Anomaly Adjustment`):** Nếu một lớp bất thường giả nào đó tạo ra toàn mẫu giống bình thường (trở thành *Pseudo-normal class*), thì khi chạy trên tập Test, AI sẽ dự đoán lớp đó xuất hiện suốt ngày (vượt quá ngưỡng $\delta = 5\%$ toàn thời gian). Bộ lọc FAA sẽ tự động coi lớp xuất hiện quá nhiều đó là bình thường và **gạch điểm của lớp đó về 0**.

---

## PHẦN 2: ĐỘNG LỰC NGHIÊN CỨU & 3 GIẢ ĐỊNH CỐT LÕI (INTRODUCTION)

Bài báo đặt nền móng bằng việc so sánh 3 giả định trong Phát hiện bất thường chuỗi thời gian không giám sát (Unsupervised TSAD - minh họa tại **Figure 1**):

1. **Giả định Bình thường (*Normality Assumption* - Autoencoder, VAE, GAN, Anomaly Transformer, Diffusion):**
   * Giả định toàn bộ tập Train đều là bình thường.
   * **Điểm yếu:** Trong thực tế, tập Train luôn bị lẫn các điểm lỗi chưa gán nhãn (**Nhiễm bẩn bất thường - Anomaly Contamination**). Mô hình sẽ học cách tái tạo luôn cả các lỗi nhiễm bẩn này $\implies$ Khái quát hóa quá mức (*Overgeneralization*), bỏ sót lỗi thật (*False Negative cao*).
2. **Giả định Bất thường Nhị phân (*Binary Anomaly Assumption* - NCAD, COUTA, CutAddPaste):**
   * Tạo bất thường giả bằng Data Augmentation rồi dạy mô hình phân loại nhị phân (`0: Normal`, `1: Anomaly`).
   * **3 Điểm yếu chí mạng:**
     1. **Khoảng trống đa dạng (*Diversity Gap*):** Một vài phép biến đổi không thể bao phủ hết các kiểu bất thường thực tế, để lại khoảng trống khiến mô hình vẫn bị tổn thương bởi nhiễm bẩn.
     2. **Bất thường giả sai lệch (*False Anomaly*):** Sinh ra các mẫu biến đổi nhưng trông vẫn bình thường, gây Overfitting và báo động giả (*False Positive cao*).
     3. **Tự tin thái quá vào nhãn (*Overconfidence on Labels*):** Dùng nhãn cứng nhị phân `0/1` tuyệt đối, không tính đến dữ liệu bẩn và False Anomaly.
3. **Giả định Bất thường Đa lớp (*Multiclass Anomaly Assumption* - Đề xuất bởi REDLAMP):**
   * Dùng **$K = 12$ lớp** (1 lớp Normal + 11 lớp bất thường giả chuyên biệt cho chuỗi thời gian) để lấp đầy *Diversity Gap*, kết hợp phân loại đa lớp bằng **Nhãn mềm (Soft Labels)** và **Tái tạo có mặt nạ (Masked Reconstruction)**.

---

## PHẦN 3: PHƯƠNG PHÁP LUẬN & HỆ THỐNG CÔNG THỨC TOÁN HỌC (METHODOLOGY)

### 3.1. Định nghĩa Bài toán (Section 3.1)
* Cho chuỗi thời gian đa biến $X = \{x_1, x_2, \dots, x_T\} \in \mathbb{R}^{D \times T}$ với $D$ đặc trưng và $T$ bước thời gian.
* Dùng cửa sổ trượt kích thước $L = 100$ cắt thành tập huấn luyện $X^{\text{train}} = \{X^{(L)}, X^{(L+1)}, \dots, X^{(T)}\}$, trong đó mỗi cửa sổ thứ $t$ là:
  $$X^{(t)} = \{x_{t-L+1}, \dots, x_t\} \in \mathbb{R}^{D \times L}$$
* Mục tiêu: Dự đoán điểm bất thường $s(t)$ cho mỗi cửa sổ $X^{(t)} \in X^{\text{test}}$ (điểm càng cao càng bất thường).

### 3.2. Thuật toán Tăng cường Dữ liệu 12 Lớp (Section 3.3 & Algorithm 1)
Với mỗi mẫu $X^{(t)} \in \mathbb{R}^{D \times L}$, thuật toán áp dụng lần lượt $K = 12$ phép biến đổi $f^{(k)}$ ($k = 1 \dots 12$) để tạo ra bộ ba:
* Mẫu tăng cường: $X^{(t,k)} \in \mathbb{R}^{D \times L}$ (tỷ lệ số mẫu mỗi lớp là $1:1$ so với tập Train gốc, tức dữ liệu tăng gấp 12 lần).
* Nhãn cứng One-hot: $y^{(t,k)} \in \{0, 1\}^K$ với phần tử thứ $k$ bằng $1$ ($y_k^{(t,k)} = 1$).
* Mặt nạ bất thường: $W^{(t,k)} \in \{0, 1\}^{D \times L}$.

**Quy trình ngẫu nhiên hóa trong Algorithm 1 (cho $k = 2 \dots 12$):**
1. Chọn ngẫu nhiên một tập con các chiều đặc trưng $\text{dims} \subseteq \{1, \dots, D\}$.
2. Với mỗi chiều $\text{dim} \in \text{dims}$, chọn ngẫu nhiên 2 mốc thời gian trong cửa sổ và sắp xếp tăng dần: $\text{st}, \text{ed} = \text{Sort}(\text{Uni}(\{1, \dots, L\}, 2))$, độ dài đoạn biến đổi $\text{length} = \text{ed} - \text{st}$.

**Công thức toán học của 12 phép biến đổi (Section 3.3.2):**
1. **(1) Normal ($k=1$):** Giữ nguyên dữ liệu gốc:
   $$X_{\text{dim}}^{(t,1)} = X_{\text{dim}}^{(t)}, \quad W^{(t,1)} = 0$$
2. **(2) Spike ($k=2$):** Cộng một gai nhọn đột biến tại đúng 1 điểm $\text{st}$:
   $$X_{\text{dim}, \text{st}}^{(t,k)} = X_{\text{dim}, \text{st}}^{(t)} + a, \quad a \sim \mathcal{N}(0, 1), \quad W_{\text{dim}, \text{st}}^{(t,k)} = 1$$
3. **(3) Flip ($k=3$):** Lật ngược đoạn dữ liệu theo chiều ngang (thời gian):
   $$X_{\text{dim}, \text{st}:\text{ed}}^{(t,k)} = X_{\text{dim}, \text{st}:\text{ed}}^{(t)}[::-1], \quad W_{\text{dim}, \text{st}:\text{ed}}^{(t,k)} = 1$$
4. **(4) Speedup ($k=4$):** Thay đổi tần số (tua nhanh gấp đôi $2\times$ hoặc kéo giãn chậm đi $1/2\times$):
   $$X_{\text{dim}, \text{st}:\text{ed}}^{(t,k)} = \begin{cases} X_{\text{dim}, \text{st}:\text{ed}+\text{length}}^{(t)}[::2], & \text{if } 2\times \\ \text{Stretch}\left(X_{\text{dim}, \text{st}:\text{ed}-(\text{length}/2)}^{(t)} : [\text{length}/2] \to [\text{length}]\right), & \text{if } 1/2\times \end{cases}, \quad W_{\text{dim}, \text{st}:\text{ed}}^{(t,k)} = 1$$
5. **(5) Noise ($k=5$):** Cộng nhiễu Gaussian vào đoạn $\text{st}:\text{ed}$:
   $$X_{\text{dim}, \text{st}:\text{ed}}^{(t,k)} = X_{\text{dim}, \text{st}:\text{ed}}^{(t)} + a, \quad a \sim \mathcal{N}(0, 0.1I), \quad W_{\text{dim}, \text{st}:\text{ed}}^{(t,k)} = 1$$
6. **(6) Cutoff ($k=6$):** Thay cả đoạn bằng một hằng số ngẫu nhiên nằm giữa min và max của đoạn đó:
   $$X_{\text{dim}, \text{st}:\text{ed}}^{(t,k)} = \text{Uni}\left(\left\{\min(X_{\text{dim}, \text{st}:\text{ed}}^{(t)}), \dots, \max(X_{\text{dim}, \text{st}:\text{ed}}^{(t)})\right\}, 1\right), \quad W_{\text{dim}, \text{st}:\text{ed}}^{(t,k)} = 1$$
7. **(7) Average ($k=7$):** Làm mượt bằng trung bình trượt với cửa sổ $w = L/5 = 20$:
   $$X_{\text{dim}, \text{st}:\text{ed}}^{(t,k)} = \text{MovingAverage}\left(X_{\text{dim}, \text{st}:\text{ed}}^{(t)}, w = L/5\right), \quad W_{\text{dim}, \text{st}:\text{ed}}^{(t,k)} = 1$$
8. **(8) Scale ($k=8$):** Nhân biên độ đoạn $\text{st}:\text{ed}$ với hệ số ngẫu nhiên:
   $$X_{\text{dim}, \text{st}:\text{ed}}^{(t,k)} = a \cdot X_{\text{dim}, \text{st}:\text{ed}}^{(t)}, \quad a \sim \mathcal{N}(1, 1), \quad W_{\text{dim}, \text{st}:\text{ed}}^{(t,k)} = 1$$
9. **(9) Wander ($k=9$):** Cộng xu hướng tuyến tính vào đoạn $\text{st}:\text{ed}$ và làm lệch toàn bộ đường cơ sở từ $\text{ed}$ đến cuối cửa sổ (mặt nạ $W$ bật bằng $1$ từ $\text{st}$ đến hết cửa sổ $L$):
   $$X_{\text{dim}, \text{st}:\text{ed}}^{(t,k)} = X_{\text{dim}, \text{st}:\text{ed}}^{(t)} + \text{np.linspace}(0, a, \text{length}), \quad X_{\text{dim}, \text{ed}:}^{(t,k)} \mathrel{+}= a, \quad a \sim \mathcal{N}(0, 1), \quad W_{\text{dim}, \text{st}:}^{(t,k)} = 1$$
10. **(10) Contextual ($k=10$):** Vừa nhân biên độ vừa cộng độ lệch nền:
    $$X_{\text{dim}, \text{st}:\text{ed}}^{(t,k)} = a \cdot X_{\text{dim}, \text{st}:\text{ed}}^{(t)} + b, \quad a \sim \mathcal{N}(1, 1), b \sim \mathcal{N}(0, 1), \quad W_{\text{dim}, \text{st}:\text{ed}}^{(t,k)} = 1$$
11. **(11) Upsidedown ($k=11$):** Lật ngược đồ thị theo chiều dọc qua giá trị trung bình của đoạn:
    $$X_{\text{dim}, \text{st}:\text{ed}}^{(t,k)} = 2 \cdot \text{Mean}(X_{\text{dim}, \text{st}:\text{ed}}^{(t)}) - X_{\text{dim}, \text{st}:\text{ed}}^{(t)}, \quad W_{\text{dim}, \text{st}:\text{ed}}^{(t,k)} = 1$$
12. **(12) Mixture ($k=12$):** Tráo đổi đoạn $\text{st}:\text{ed}$ với một đoạn cùng độ dài từ cửa sổ ở thời điểm $t'$ khác:
    $$X_{\text{dim}, \text{st}:\text{ed}}^{(t,k)} = X_{\text{dim}, \text{st}:\text{ed}}^{(t')}, \quad t' \in \{L, \dots, T\}, \quad W_{\text{dim}, \text{st}:\text{ed}}^{(t,k)} = 1$$

### 3.3. Làm mới Nhãn & Hiệu chỉnh Ngược (Label Refurbishment — Công thức 1)
Chuyển đổi nhãn cứng one-hot $y^{(t,k)}$ thành nhãn mềm $\tilde{y}^{(t,k)}$ cho từng phần tử thứ $i \in \{1, \dots, K\}$:
$$\tilde{y}_i^{(t,k)} = \begin{cases} (1 - \alpha - K\beta)y_i^{(t,k)} + \alpha + \beta, & \text{if } i = 1 \text{ (vị trí lớp Normal)}, \\ (1 - \alpha - K\beta)y_i^{(t,k)} + \beta, & \text{otherwise (vị trí các lớp Anomaly } i = 2 \dots K\text{)}, \end{cases} \quad (1)$$
* Với thiết lập mặc định $K = 12, \alpha = 0.1, \beta = 0.01 \implies 1 - \alpha - K\beta = 0.78$:
  * **Nếu mẫu thuộc lớp Normal ($k=1$):** Vector nhãn mềm là $[0.89, 0.01, 0.01, \dots, 0.01]$.
  * **Nếu mẫu thuộc lớp Anomaly (ví dụ $k=2$):** Vector nhãn mềm là $[0.11, 0.79, 0.01, \dots, 0.01]$ (tức luôn có $11\%$ xác suất mẫu biến đổi này thực chất vẫn là Normal).

### 3.4. Hàm Mục tiêu Huấn luyện (Objective Function — Công thức 2, 3, 4)
1. **Hàm mất mát tái tạo có mặt nạ (Masked Reconstruction Loss):**
   $$\mathcal{L}_{\text{MSE}} = \left\| (1 - W^{(t,k)}) \odot (X^{(t,k)} - \hat{X}^{(t,k)}) \right\|_2^2 \quad (2)$$
2. **Hàm mất mát phân loại đa lớp nhãn mềm (Cross-Entropy Loss):**
   $$\mathcal{L}_{\text{CE}} = -\sum_{i=1}^{K} \tilde{y}_i^{(t,k)} \log(\hat{y}_i^{(t,k)}) \quad (3)$$
3. **Hàm mất mát tổng hợp (Total Loss — với $\gamma = 0.1$):**
   $$\mathcal{L} = \gamma \mathcal{L}_{\text{CE}} + (1 - \gamma) \mathcal{L}_{\text{MSE}} \quad (4)$$

### 3.5. Chấm điểm Bất thường lúc Kiểm thử (Anomaly Scoring — Công thức 5, 6, 7, 8)
Khi đưa mẫu kiểm thử $X^{(t)} \in X^{\text{test}}$ vào mô hình:
1. **Điểm sai số tái tạo:**
   $$s_{\text{MSE}}(t) = \left\| X^{(t)} - \hat{X}^{(t)} \right\|_2^2 \quad (5)$$
2. **Hàm điều chỉnh bất thường thường xuyên (Frequent Anomaly Adjustment - FAA):** Loại bỏ các lớp giả bình thường (*pseudo-normal classes*) nếu xác suất dự đoán trung bình của lớp $i$ trên toàn bộ tập Test vượt quá $\delta = 0.05$:
   $$\text{FAA}(\hat{y}_i^{(t)}) = \begin{cases} 0, & \text{if } \frac{1}{|X^{\text{test}}|} \sum_{X^{(t')} \in X^{\text{test}}} \hat{y}_i^{(t')} > \delta, \\ \hat{y}_i^{(t)}, & \text{otherwise}, \end{cases} \quad (6)$$
3. **Điểm lớp bất thường đã điều chỉnh (tổng từ lớp $i = 2$ đến $K$):**
   $$s_{\text{CE}}(t) = \sum_{i=2}^{K} \text{FAA}(\hat{y}_i^{(t)}) \quad (7)$$
4. **Điểm bất thường tổng hợp (chuẩn hóa Min-Max trên toàn chuỗi Test rồi lấy trung bình):**
   $$s(t) = \frac{1}{2} \text{MinMax}_T(s_{\text{MSE}}(t)) + \frac{1}{2} \text{MinMax}_T(s_{\text{CE}}(t)) \quad (8)$$
   *(Hậu xử lý: Áp dụng thêm bộ lọc trung bình trượt Moving Average có độ dài $L/2 = 50$ bước thời gian lên $s(t)$ để làm mượt chuỗi điểm).*

---

## PHẦN 4: THÔNG SỐ BỘ DỮ LIỆU, SIÊU THAM SỐ & KẾT QUẢ THỰC NGHIỆM

### 4.1. Thống kê 5 Bộ dữ liệu Thực nghiệm (Table 2 & Appendix A)

| Thông số | UCR | AIOps | SMD | SMAP | MSL |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Số tập dữ liệu con (`# Subdatasets`)** | 250 | 29 | 28 | 50 | 23 |
| **Số chiều đặc trưng ($D$)** | 1 (Đơn biến) | 1 (Đơn biến) | 38 (Đa biến) | 55 (Đa biến) | 25 (Đa biến) |
| **Lĩnh vực** | Đa dạng (9 nhóm) | Cloud KPIs | Server Machine | Tàu vũ trụ NASA | Tàu Sao Hỏa |
| **Độ dài chuỗi Train TB ($T_{\text{train}}$)** | 30,522 | 103,588 | 25,300 | 2,752 | 2,410 |
| **Tỷ lệ nhiễm bẩn tập Train (%)** | 0.00% | 2.49% | Không rõ | Không rõ | Không rõ |
| **Độ dài chuỗi Test TB ($T_{\text{test}}$)** | 56,173 | 100,649 | 25,300 | 8,203 | 2,923 |
| **Tỷ lệ điểm bất thường tập Test (%)** | 0.85% | 1.81% | 4.21% | 12.61% | 13.01% |
| **Bước nhảy cửa sổ Train (`Stride`)** | $\{1, 10, 100\}$ (để $<10,000$ mẫu) | 10 | 10 | 10 | 10 |
| **Cách chạy tập con** | Tách riêng (5 runs) | Tách riêng (5 runs) | Tách riêng (5 runs) | Gộp chung | Gộp chung |

### 4.2. Bảng Siêu tham số Cố định của REDLAMP (Table 4 & Appendix C)
* **Kiến trúc Encoder:** 4 khối `ConvBlock1D` với số filters `[128, 128, 256, 256]`, `stride = 2`, `BatchNorm1d`, `ReLU`, `Dropout = 0.2` $\to$ `MaxPool1d` $\to$ `Conv1d` ánh xạ về chiều không gian ẩn **`Encoder embedding dim = 128`**.
* **Kiến trúc Decoder:** Đối xứng với Encoder (`Linear` upsampling $\to$ 4 khối `ConvBlock1D` $\to$ `Conv1d` trả về kích thước `(D, 100)`).
* **Kiến trúc Classifier:** MLP 2 tầng (`Linear(128 -> 32)` $\to$ `BatchNorm1d` $\to$ `ReLU` $\to$ `Dropout(0.2)` $\to$ `Linear(32 -> 12)` $\to$ `Softmax`).
* **Huấn luyện:** Optimizer `AdamW` (`lr = 1e-3`), `batch_size = 128`, `window_size L = 100`, tối đa `100 epochs` (Early stopping với 10% Validation).
* **Tham số REDLAMP:** $K = 12, \alpha = 0.1, \beta = 0.01, \gamma = 0.1, \delta = 0.05$.

### 4.3. Tóm tắt Kết quả Chính (Section 4.2 – 4.6)
* **Hiệu năng (Q1):** Đứng số 1 trung bình trên cả 5 bộ dữ liệu với **VUS-ROC = 0.803**, **VUS-PR = 0.430** (vượt phương pháp thứ hai CutAddPaste tới **+17.4%**), **Range F-score = 0.219** (+14.6%) và độ chính xác phát hiện trên UCR đạt **62.0%**.
* **Nghiên cứu cắt bỏ thành phần (Q2 - Ablation Study):** Cắt bỏ bất kỳ thành phần nào (`w/o AM`, `w/o BC`, `w/o L_MSE`, `w/o L_CE`, `w/o FAA` hoặc chuyển về `Binary`) đều làm giảm hiệu năng đáng kể. Đặc biệt nếu gộp 11 loại lỗi về phân loại nhị phân (`Binary`), VUS-PR giảm tới **15.6% trên UCR** và **24.3% trên SMD**.
* **Kháng nhiễm bẩn (Q4 - Robustness):** Khi trộn 5% bất thường vào tập Train của UCR, REDLAMP chỉ giảm nhẹ **1.8%** VUS-PR, trong khi CutAddPaste sụt giảm tới **15.5%** và các mô hình tái tạo thuần túy (AT, BeatGAN) suy giảm nghiêm trọng.
* **Khả năng giải thích (Q5 - Explainability):** Vẽ biểu đồ t-SNE 2D của các vector `Embedding` 128 chiều giúp nhìn thấy rõ các điểm lỗi thực tế trong tập Test nằm sát cụm lớp bệnh giả nào (ví dụ: lỗi chu kỳ của UCR #040 EPG nằm sát cụm `Speedup`; lỗi nhiễu đỉnh sóng UCR #028 ABP nằm sát cụm `Noise`; lỗi máy chủ SMD Machine 1-1 nằm gần `Spike` và `Cutoff`).

---

## PHẦN 5: MÃ NGUỒN PYTORCH TÁI LẬP ĐẦY ĐỦ & KỊCH BẢN MÔ PHỎNG TRỰC QUAN

Dưới đây là mã nguồn Python/PyTorch hoàn chỉnh chứa toàn bộ lõi thuật toán của REDLAMP và hàm chạy thử nghiệm mô phỏng để bạn có thể chạy kiểm tra ngay trên máy tính:

```python
import numpy as np
import torch
import torch.nn as nn

# =====================================================================
# 1. THUẬT TOÁN TĂNG CƯỜNG DỮ LIỆU 12 LỚP & TẠO MẶT NẠ W (ALGORITHM 1)
# =====================================================================
def augment_instance(x_window, all_windows=None):
    """
    Đầu vào: x_window có kích thước (D, L) với L = 100
    Đầu ra:
      - X_aug: (12, D, L) - 12 phiên bản của cửa sổ (1 Normal + 11 Anomaly)
      - y_aug: (12, 12)   - Nhãn cứng One-hot cho 12 lớp
      - W_aug: (12, D, L) - Mặt nạ nhị phân đánh dấu vị trí chèn lỗi (1 = lỗi, 0 = bình thường)
    """
    D, L = x_window.shape
    K = 12
    X_aug = np.zeros((K, D, L), dtype=np.float32)
    y_aug = np.eye(K, dtype=np.float32)
    W_aug = np.zeros((K, D, L), dtype=np.float32)

    for k in range(K):
        x_new = x_window.copy()
        w_mask = np.zeros((D, L), dtype=np.float32)
        
        if k == 0:  # Lớp 1: Normal (Giữ nguyên)
            X_aug[k] = x_new
            continue
            
        num_dims = np.random.randint(1, D + 1)
        dims = np.random.choice(D, size=num_dims, replace=False)
        
        for dim in dims:
            pts = np.sort(np.random.choice(np.arange(0, L), size=2, replace=False))
            st, ed = int(pts[0]), int(pts[1])
            if ed - st < 2:
                ed = min(L, st + 2)
            length = ed - st
            seg = x_new[dim, st:ed].copy()

            if k == 1:    # (2) Spike
                x_new[dim, st] += np.random.normal(0, 1.0)
                w_mask[dim, st] = 1.0
            elif k == 2:  # (3) Flip
                x_new[dim, st:ed] = seg[::-1]
                w_mask[dim, st:ed] = 1.0
            elif k == 3:  # (4) Speedup (2x hoặc 0.5x)
                if np.random.rand() < 0.5 and st + 2 * length <= L:
                    x_new[dim, st:ed] = x_window[dim, st:st + 2 * length:2][:length]
                else:
                    half_len = max(1, length // 2)
                    x_new[dim, st:ed] = np.interp(
                        np.linspace(0, half_len - 1, length),
                        np.arange(half_len), seg[:half_len]
                    )
                w_mask[dim, st:ed] = 1.0
            elif k == 4:  # (5) Noise
                x_new[dim, st:ed] += np.random.normal(0, np.sqrt(0.1), size=length)
                w_mask[dim, st:ed] = 1.0
            elif k == 5:  # (6) Cutoff
                val = np.random.uniform(np.min(seg), np.max(seg) + 1e-6)
                x_new[dim, st:ed] = val
                w_mask[dim, st:ed] = 1.0
            elif k == 6:  # (7) Average (Moving average w = L/5)
                w_size = max(1, L // 5)
                kernel = np.ones(w_size) / w_size
                x_new[dim, st:ed] = np.convolve(seg, kernel, mode='same')
                w_mask[dim, st:ed] = 1.0
            elif k == 7:  # (8) Scale
                a = np.random.normal(1.0, 1.0)
                x_new[dim, st:ed] = a * seg
                w_mask[dim, st:ed] = 1.0
            elif k == 8:  # (9) Wander
                a = np.random.normal(0.0, 1.0)
                x_new[dim, st:ed] += np.linspace(0, a, length)
                x_new[dim, ed:] += a
                w_mask[dim, st:] = 1.0
            elif k == 9:  # (10) Contextual
                a, b = np.random.normal(1.0, 1.0), np.random.normal(0.0, 1.0)
                x_new[dim, st:ed] = a * seg + b
                w_mask[dim, st:ed] = 1.0
            elif k == 10: # (11) Upsidedown
                x_new[dim, st:ed] = 2.0 * np.mean(seg) - seg
                w_mask[dim, st:ed] = 1.0
            elif k == 11: # (12) Mixture
                if all_windows is not None and len(all_windows) > 1:
                    rand_idx = np.random.randint(0, len(all_windows))
                    x_new[dim, st:ed] = all_windows[rand_idx][dim, st:ed]
                w_mask[dim, st:ed] = 1.0

        X_aug[k] = x_new
        W_aug[k] = w_mask
    return X_aug, y_aug, W_aug

# =====================================================================
# 2. HIỆU CHỈNH NHÃN MỀM (BACKWARD CORRECTION - CÔNG THỨC 1)
# =====================================================================
def backward_correction(y_onehot, alpha=0.1, beta=0.01, K=12):
    y_tilde = (1.0 - alpha - K * beta) * y_onehot + beta
    y_tilde[:, 0] += alpha  # Cộng thêm alpha vào cột đầu tiên (Lớp Normal k=1)
    return y_tilde

# =====================================================================
# 3. KIẾN TRÚC MẠNG REDLAMP (ENCODER -> EMBEDDING -> DECODER & CLASSIFIER)
# =====================================================================
class ConvBlock1D(nn.Module):
    def __init__(self, in_c, out_c, stride=2, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_c),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.block(x)

class REDLAMP(nn.Module):
    def __init__(self, in_features, window_size=100, K=12, emb_dim=128, cls_dim=32):
        super().__init__()
        self.in_features = in_features
        self.window_size = window_size
        
        # Khối Encoder nén dữ liệu về Embedding 128 chiều
        self.enc_blocks = nn.Sequential(
            ConvBlock1D(in_features, 128, stride=2, dropout=0.2),
            ConvBlock1D(128, 128, stride=2, dropout=0.2),
            ConvBlock1D(128, 256, stride=2, dropout=0.2),
            ConvBlock1D(256, 256, stride=2, dropout=0.2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(256, emb_dim, kernel_size=1)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Nhánh 1: Classifier ("Bác sĩ chẩn đoán" 12 lớp)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, cls_dim),
            nn.BatchNorm1d(cls_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(cls_dim, K),
            nn.Softmax(dim=-1)
        )
        
        # Nhánh 2: Decoder ("Họa sĩ phục chế" chuỗi bình thường)
        self.dec_linear = nn.Linear(emb_dim, 256 * window_size)
        self.dec_blocks = nn.Sequential(
            ConvBlock1D(256, 256, stride=1, dropout=0.2),
            ConvBlock1D(256, 128, stride=1, dropout=0.2),
            ConvBlock1D(128, 128, stride=1, dropout=0.2),
            ConvBlock1D(128, 128, stride=1, dropout=0.2),
            nn.Conv1d(128, in_features, kernel_size=1)
        )

    def forward(self, x):
        feat = self.enc_blocks(x)
        emb = self.adaptive_pool(feat).squeeze(-1)  # Tờ phiếu tóm tắt Embedding: (Batch, 128)
        y_hat = self.classifier(emb)                # Dự đoán xác suất 12 lớp: (Batch, 12)
        dec_in = self.dec_linear(emb).view(-1, 256, self.window_size)
        x_hat = self.dec_blocks(dec_in)             # Chuỗi tái tạo: (Batch, D, L)
        return x_hat, y_hat, emb

# =====================================================================
# 4. HÀM MẤT MÁT & HÀM CHẤM ĐIỂM BẤT THƯỜNG KÈM BỘ LỌC FAA
# =====================================================================
def redlamp_loss(x_aug, x_hat, w_mask, y_tilde, y_hat, gamma=0.1):
    # Nhân với (1 - w_mask) để che đi vùng lỗi giả, không bắt Decoder vẽ lại lỗi
    masked_diff = (1.0 - w_mask) * (x_aug - x_hat)
    l_mse = torch.mean(torch.sum(masked_diff ** 2, dim=(1, 2)))
    l_ce = -torch.mean(torch.sum(y_tilde * torch.log(y_hat + 1e-8), dim=-1))
    return gamma * l_ce + (1.0 - gamma) * l_mse
```

---

## PHẦN 6: LỘ TRÌNH TIẾP TỤC NGHIÊN CỨU CHO BUỔI TIẾP THEO
*(Khi bắt đầu cuộc hội thoại mới vào ngày mai, bạn chỉ cần gửi file này lên và chọn 1 trong các mục dưới đây để chúng ta đi tiếp)*

1. **Chạy thử nghiệm Code trực tiếp trên máy tính:** Chạy kịch bản sinh sóng giả lập và huấn luyện thử mô hình REDLAMP trên 1 tập dữ liệu con của UCR hoặc SMD để quan sát hàm Loss giảm dần và xem biểu đồ bắt lỗi.
2. **Đọc hiểu cấu trúc Source Code GitHub chính thức của tác giả:** Phân tích từng file trong repo `github.com/KoheiObata/RedLamp` để biết cách nạp dữ liệu của riêng bạn vào mô hình.
3. **Đi sâu vào Thước đo đánh giá mới (`VUS-PR`, `VUS-ROC`, `Range F-score`):** Tìm hiểu tại sao giới nghiên cứu hiện nay loại bỏ cách tính `Point-Adjusted F1` cũ và cách tính `VUS-PR` trong thực nghiệm.
4. **Hướng phát triển / Cải tiến bài báo (Future Work cho Đồ án / Luận văn):** Tác giả nêu rõ ở phần Conclusion rằng các phép biến đổi hiện tại đang áp dụng độc lập trên từng chiều đặc trưng và chưa khai thác sâu mối tương quan chéo giữa các cảm biến trong chuỗi đa biến (Multivariate correlation anomalies) — đây là hướng mở rộng cực tốt nếu bạn muốn phát triển nghiên cứu riêng.