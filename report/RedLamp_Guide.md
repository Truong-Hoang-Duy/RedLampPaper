# 📘 CẨM NANG TOÀN DIỆN REDLAMP (KDD '25): TỪ ĐIỂN THUẬT NGỮ, KIẾN TRÚC `ConvAEC`, GIẢI PHẪU CÔNG THỨC (1)–(8) & TỔNG QUAN HỆ THỐNG

**Tên bài báo gốc:** *Robust and Explainable Detector of Time Series Anomaly via Augmenting Multiclass Pseudo-Anomalies (REDLAMP)*  
**Tác giả bài báo gốc:** Kohei Obata, Yasuko Matsubara, Yasushi Sakurai (Đại học Osaka, Nhật Bản) — **Hội nghị KDD '25** ([`2505.20765v1.pdf`](./2505.20765v1.pdf))  
**Báo cáo tái lập thực nghiệm (CS2207):** [`Khai_thác_dữ_liệu_và_ứng_dụng___CS2207_26_1_CH_02.pdf`](./Khai_thác_dữ_liệu_và_ứng_dụng___CS2207_26_1_CH_02.pdf)  
**Kịch bản thuyết trình Live Demo chi tiết (từng thao tác & số liệu):** Xem tại [`RedLamp_Demo_Presentation.md`](./RedLamp_Demo_Presentation.md)

> **🎯 Mục tiêu của tài liệu tổng hợp này:**  
> Tài liệu này hợp nhất toàn bộ nội dung hướng dẫn lý thuyết nền tảng dành cho người mới bắt đầu Machine Learning, giải nghĩa mọi **từ viết tắt**, **ký hiệu toán học**, **kiến trúc `1D-CNN` / `ConvAEC`**, **12 lớp bất thường giả** và **công thức toán học từ (1) đến (8)** bằng các **ví dụ con số nhẩm tính được ngay trong 10 giây**, đồng thời tổng quan cấu trúc 3 Tab của ứng dụng Web Demo.

---

# PHẦN 1: TỪ ĐIỂN GIẢI NGHĨA TẤT CẢ CÁC TỪ VIẾT TẮT & THUẬT NGỮ (TRA CỨU NHANH)

## 1.1. Các thuật ngữ của riêng phương pháp REDLAMP
| Viết tắt | Tiếng Anh đầy đủ | Nghĩa Tiếng Việt | Giải thích bình dân trong 1 câu |
| :--- | :--- | :--- | :--- |
| **TSAD** | *Time Series Anomaly Detection* | Phát hiện bất thường chuỗi thời gian | Bài toán tìm xem ở giây/phút thứ mấy thì cảm biến (nhiệt độ, nhịp tim, CPU...) bị nhảy số hoặc biến dạng bất thường. |
| **REDLAMP** | *Robust and Explainable Detector of Time Series Anomaly via Augmenting Multiclass Pseudo-Anomalies* | Bộ phát hiện bất thường bền vững & có tính giải thích bằng cách tạo bất thường giả đa lớp | Tên mô hình của bài báo: tự tạo ra **11 kiểu lỗi giả** để dạy AI vừa biết bắt lỗi, vừa chỉ ra được lỗi đó thuộc hình dạng nào. |
| **AM** | *Anomaly Mask* ($W$) | Mặt nạ đánh dấu vùng lỗi | Một tấm bìa chỉ gồm số `0` và `1`, đánh dấu chính xác khúc nào vừa bị chèn lỗi giả để "che mắt" không cho bộ vẽ lại (Decoder) học vẹt lỗi đó. |
| **BC** | *Backward Correction* | Hiệu chỉnh ngược (Nhãn mềm) | Thay vì bắt AI tin tuyệt đối 100% rằng mẫu tạo ra là lỗi (`[0, 1]`), ta giảm xuống còn 79% và luôn chừa 11% khả năng nó vẫn là bình thường (`[0.11, 0.79, ...]`). |
| **FAA** | *Frequent Anomaly Adjustment* | Bộ lọc bất thường xuất hiện quá thường xuyên | Quy tắc: *"Bất thường thật thì phải hiếm"*. Nếu lúc chạy thực tế mà AI bảo một loại lỗi xuất hiện quá 5% tổng thời gian ($\delta = 0.05$), bộ lọc FAA coi đó là báo động giả và gạch điểm của loại đó về `0`. |

## 1.2. Các thuật ngữ Mô hình Học sâu (Deep Learning) & Hàm mất mát (Loss)
| Viết tắt | Tiếng Anh đầy đủ | Nghĩa Tiếng Việt | Giải thích bình dân trong 1 câu |
| :--- | :--- | :--- | :--- |
| **ConvAEC** | *Convolutional Autoencoder + NonLinear Classifier* | Mạng Tự mã hóa Tích chập kết hợp Bộ phân loại Phi tuyến | Tên kiến trúc mạng trong code (`RedLamp/models/meta.py`), kế thừa từ SREA, gồm 3 khối: `ConvEncoder`, `ConvDecoder` và `NonLinear Classifier`. |
| **AE** | *Autoencoder* | Mạng tự mã hóa (Nén & Khôi phục) | Mạng gồm 2 nửa: **Encoder** (nén đoạn sóng dài thành vài con số tóm tắt) và **Decoder** (từ vài con số đó vẽ khôi phục lại đoạn sóng ban đầu). |
| **1D-CNN** | *1D Convolutional Neural Network* | Mạng tích chập 1 chiều | Mạng dùng các "kính lúp" (`kernel`) trượt dọc theo 1 hướng duy nhất là trục thời gian để nhận diện hình dáng sóng (gai nhọn, dốc lên, sóng sin...). |
| **MLP** | *Multi-Layer Perceptron* (`NonLinear Classifier`) | Mạng nơ-ron đa tầng (Bộ phân loại) | Khối mạng đóng vai **"Bác sĩ chẩn đoán"**: nhận bản tóm tắt từ Encoder rồi phán đoán đoạn sóng thuộc lớp nào trong 12 lớp. |
| **Dims ($D$)** | *Dimensions / Features* (`n_features`) | Số chiều đặc trưng đầu vào | Số lượng cảm biến / chỉ số đo tại **cùng một mốc thời gian $t$** (`1` với đơn biến; `25`, `38`, `55` với đa biến). |
| **Embedding** | *Latent Embedding Vector* (`emb_dim = 128`) | Vector biểu diễn tiềm ẩn (128 số) | **"Tờ phiếu khám bệnh 128 con số"** do Encoder tóm tắt từ cửa sổ 100 điểm đầu vào (luôn cố định 128 số bất kể `dims` bằng bao nhiêu). |
| **MSE** | *Mean Squared Error* ($\mathcal{L}_{\text{MSE}}$) | Sai số bình phương trung bình | Đo xem **hình vẽ lại của Decoder khác hình gốc bao nhiêu**: lấy `(Gốc - Vẽ lại)` bình phương lên. Lệch càng nhiều $\implies$ MSE càng cao $\implies$ bất thường! |
| **CE** | *Cross-Entropy Loss* ($\mathcal{L}_{\text{CE}}$) | Hàm mất mát Entropy chéo | Đo xem **Classifier đoán 12 lớp sai lệch bao nhiêu** so với nhãn mềm mục tiêu. Đoán càng đúng lớp thì CE càng nhỏ (tiến về 0). |
| **Softmax** | *Softmax Activation* | Hàm chuẩn hóa xác suất | Biến 12 điểm số thô ở đầu ra của Classifier thành **12 con số xác suất cộng lại đúng bằng 100% (`1.0`)**. |

## 1.3. Các thuật ngữ Đánh giá Kết quả (Metrics) & Phân tích
| Viết tắt | Tiếng Anh đầy đủ | Nghĩa Tiếng Việt | Giải thích bình dân trong 1 câu |
| :--- | :--- | :--- | :--- |
| **FPR** | *False Positive Rate* | Tỷ lệ Dương tính giả (**Báo động giả**) | Máy đang chạy bình thường nhưng AI lại hú còi báo lỗi. |
| **FNR** | *False Negative Rate* | Tỷ lệ Âm tính giả (**Bỏ sót lỗi**) | Máy đang bị hỏng thật nhưng AI lại im lặng bảo bình thường. |
| **ROC / PR** | *Receiver Operating Characteristic / Precision-Recall* | Đường cong ROC và Đường cong Độ chính xác - Độ phủ | Hai biểu đồ chuẩn đo chất lượng bộ phát hiện lỗi trên mọi ngưỡng cắt. Khi lỗi rất hiếm (như trong chuỗi thời gian), **PR** khắt khe và chuẩn xác hơn **ROC**. |
| **R-AUC-ROC / R-AUC-PR** | *Range Area Under the Curve* | Diện tích dưới đường cong tính theo khoảng (Range) | Phiên bản nâng cấp của AUC dành riêng cho chuỗi thời gian: tính điểm theo **cả đoạn thời gian bị lỗi** (có vùng đệm $L/2 = 50$ xung quanh) thay vì chấm từng giây rời rạc. |
| **VUS-ROC / VUS-PR** | *Volume Under the Surface* | Thể tích dưới bề mặt (Thước đo chuẩn nhất KDD '25) | Quét qua mọi độ dài vùng đệm khác nhau để tạo thành mặt cong 3D rồi tính thể tích. **Số càng gần `1.0` là mô hình càng xuất sắc.** |
| **RF** | *Range F-score* | Điểm cân bằng F1 theo khoảng thời gian | Trung bình điều hòa giữa Precision và Recall tính trên từng phân đoạn lỗi, có phạt nếu phát hiện bị băm vụn thành nhiều mảnh. |
| **t-SNE / PCA** | *t-Distributed Stochastic Neighbor Embedding / Principal Component Analysis* | Thuật toán giảm chiều dữ liệu (từ 128D xuống 2D) | "Ép phẳng" không gian 128 chiều của `Embedding` xuống mặt phẳng 2D ($x, y$) để vẽ biểu đồ phân cụm trực quan. |
| **XAI** | *Explainable Artificial Intelligence* | AI có khả năng giải thích | Không chỉ phán "Có lỗi!", AI còn chỉ ra được **"Lỗi này có hình thái giống lỗi nhiễu sóng (Noise), đảo cực (Flip) hay kẹt cảm biến (Cutoff)"**. |

## 1.4. Bảng Phân Biệt Nhanh 4 Thuật Ngữ Dễ Nhầm Lẫn Nhất Khi Hội Đồng Hỏi

| Thuật ngữ | Bản chất | Xuất hiện lúc nào? | Giải thích trong 1 câu | Hậu quả nếu không xử lý | Vũ khí của REDLAMP |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`False Anomaly`** *(Lớp giả bình thường)* | **Mẫu dữ liệu** nhân tạo | Lúc **Huấn luyện** | Biến đổi tín hiệu tạo lỗi giả (như `average`), nhưng kết quả trông **vẫn giống hệt bình thường**. | Tăng **FPR** *(Báo động giả)* | **Nhãn mềm ($\alpha=0.1$)** lúc học & **Bộ lọc FAA ($> 0.05$)** lúc kiểm thử. |
| **`Anomaly Contamination`** *(Nhiễm bẩn bất thường)* | **Mẫu dữ liệu** thực tế | Lúc **Huấn luyện** | Tập Train chưa gán nhãn bị **lẫn sẵn lỗi thật** nhưng máy tưởng là bình thường. | Tăng **FNR** *(Bỏ sót lỗi thật)* | **11 lớp biến đổi đa dạng** + **Nhãn mềm ($\beta=0.01$)** + **Mặt nạ tái tạo ($W$)**. |
| **`Điểm kết hợp thô (raw_combined)`** | **Điểm số suy luận** | Lúc **Dự đoán & Giải thích** | Trung bình cộng $(\text{MSE} + \text{CE}) / 2$ chưa qua chuẩn hóa Min-Max. | Không bị kéo đỉnh giả lên `1.0` khi chuỗi sạch | Đối chiếu trực tiếp với ngưỡng phân vị **`P99.5`** ở Tab 2 & Bước 5 Tab 3 của Demo. |
| **`Điểm bất thường tương đối (anomaly_score)`** | **Điểm chuẩn hóa `[0, 1]`** | Lúc **Đánh giá toàn chuỗi** | Điểm kết hợp sau khi làm mượt 50 bước và chuẩn hóa Min-Max trên toàn chuỗi Test. | Luôn có đỉnh cực đại bằng `1.0` trên mỗi chuỗi | Vẽ đường đỏ tổng quan ở Tab 1 và dùng để chấm **`VUS-ROC`**, **`VUS-PR`**, **`RF`**. |

## 1.5. Năm Bộ Dữ Liệu Chuẩn Trong Bài Báo & Kết Quả Tái Lập Của Nhóm
* **`UCR` (*UCR Anomaly Archive — `1 dim`*):** 250 chuỗi đơn biến ($D=1$) từ y tế (ECG, ABP), nông nghiệp (côn trùng EPG), công nghiệp... Tập Train sạch 100% (0% lỗi); 77.7% lỗi nằm hoàn toàn trong biên độ bình thường (*lỗi hình thái*). Điểm bài báo: `VUS-ROC = 0.897`, `VUS-PR = 0.492`, `RF = 0.234`.
* **`AIOps / IOPS` (*Cloud KPIs — `1 dim`*):** 29 chuỗi đơn biến ($D=1$) theo dõi KPI máy chủ Web đám mây. Tập Train bị nhiễm bẩn trung bình `2.49%` lỗi thật. Nhóm tái lập 5 seeds đạt `VUS-ROC = 0.926 ± 0.004`, `VUS-PR = 0.458 ± 0.009`, `RF = 0.180 ± 0.007` (**vượt nhẹ bài báo** `0.911 / 0.448 / 0.179`).
* **`SMD` (*Server Machine Dataset — `38 dims`*):** 28 máy chủ, mỗi máy đo đồng thời **$D=38$ chỉ số** (CPU, RAM, Disk, Network...) suốt 5 tuần; mỗi đoạn lỗi thường liên quan trung vị 4 chiều. Nhóm tái lập 5 seeds đạt `VUS-ROC = 0.823 ± 0.005`, `VUS-PR = 0.374 ± 0.013`, `RF = 0.239 ± 0.009` (**vượt nhẹ bài báo** `0.813 / 0.366 / 0.235`).
* **`SMAP` (*`25 dims`*) & `MSL` (*`55 dims`*) (*NASA Spacecraft Datasets*):** Vệ tinh đo độ ẩm đất (`SMAP`) và Tàu tự hành Sao Hỏa (`MSL`) của NASA, gộp chung các kênh khi train 1 mô hình chia sẻ vì mỗi kênh ngắn (~2,410 – 2,752 điểm). Nhóm tái lập đạt `MSL VUS-PR = 0.469 ± 0.022` (bài báo `0.482`) và `SMAP VUS-PR = 0.329 ± 0.015` (bài báo `0.360`).

---

# PHẦN 2: CÁCH ĐỌC CÁC KÝ HIỆU TOÁN HỌC TRONG BÀI BÁO (KHÔNG SỢ CÔNG THỨC!)

Chỉ cần nắm **6 quy tắc ký hiệu** đơn giản sau là bạn đọc hiểu trọn vẹn mọi công thức trong bài báo:

1. **Chữ in hoa ($X, W$) vs. Chữ thường ($x, y, s$):**
   * Chữ in hoa như $X^{(t)}$ là một **bảng số (ma trận)** chứa cả một đoạn cửa sổ thời gian.
   * Chữ thường như $s(t)$ là **một con số điểm** hoặc một danh sách ngắn (vector).
2. **Dấu mũ nón ($\hat{X}, \hat{y}$) vs. Dấu ngã ($\tilde{y}$) vs. Không dấu ($X, y$):**
   * **Không dấu ($X, y$):** Dữ liệu gốc ban đầu / Nhãn cứng ban đầu.
   * **Có dấu mũ nón ($\hat{X}, \hat{y}$ — đọc là "X-hat", "y-hat"):** Kết quả **do mạng AI dự đoán ra**. (Cứ thấy dấu `^` trên đầu là sản phẩm của AI!).
   * **Có dấu ngã ($\tilde{y}$ — đọc là "y-tilde"):** Nhãn đã được **làm mềm** (chỉnh sửa nhẹ từ nhãn gốc $y$).
3. **Ký hiệu $\in \mathbb{R}^{D \times L}$ (thuộc không gian số thực kích thước $D \times L$):**
   * Nghĩa là **một bảng Excel có $D$ hàng (số cảm biến `dims`) và $L$ cột (độ dài cửa sổ thời gian)**. Ví dụ $D=38, L=100$ là bảng 38 hàng × 100 cột.
4. **Ký hiệu $\odot$ hoặc $\circ$ (Phép nhân từng vị trí — Hadamard / Element-wise):**
   * *Ví dụ:* $[2, 5, 8] \odot [1, 0, 1] = [2\times 1, \; 5\times 0, \; 8\times 1] = [2, 0, 8]$.
5. **Ký hiệu $\| A \|_2^2$ (Chuẩn bình phương L2):**
   * Lấy từng con số trong bảng $A$ bình phương lên rồi cộng tất cả lại: $\| [3, -2, 1] \|_2^2 = 3^2 + (-2)^2 + 1^2 = 14$.
6. **Ký hiệu $a \sim \mathcal{N}(\mu, \sigma^2)$ (Bốc số ngẫu nhiên theo Phân phối chuẩn hình quả chuông):**
   * Máy tính bốc ngẫu nhiên một con số $a$ xoay quanh tâm $\mu$ với độ lệch chuẩn $\sigma$ theo **Quy tắc 68–95–99.7** ($\sigma = 1$ không phải chặn cứng ở $[-1, +1]$ mà ~68.2% rơi vào $[-1, +1]$, ~95.4% rơi vào $[-2, +2]$, và ~99.7% rơi vào $[-3, +3]$):

```text
                  Đỉnh chuông (Tâm μ = 0)
                           _.-""""-._
                         .'    |     '.
                        /      |       \
                       / 34.1% | 34.1%  \        <-- Khúc bụng giữa cao nhất:
                      /        |         \           34.1% + 34.1% = 68.2%
                    .'         |          '.
                  .'   13.6%   |   13.6%    '.   <-- Cộng thêm 2 bên sườn:
               _.'      |      |      |       '._    13.6% + 68.2% + 13.6% = 95.4%
         _..--'  2.1%   |      |      |   2.1%   '--.._
   ========================================================= (Trục giá trị a)
        -3.0          -2.0   -1.0    0   +1.0   +2.0       +3.0
                        |<--- 68.2% --->|
                 |<--------- 95.4% --------->|
         |<----------------- 99.7% ----------------->|
```

---

# PHẦN 3: CÂU CHUYỆN CỐT LÕI — TẠI SAO CÁC PHƯƠNG PHÁP CŨ THẤT BẠI & TẠI SAO CẦN REDLAMP?

Trong bài toán Không giám sát (*Unsupervised TSAD*), dữ liệu lịch sử đưa cho AI học **không hề được đánh dấu sẵn chỗ nào là bình thường, chỗ nào là lỗi**. Hai hướng tiếp cận trước đây đều gặp "tử huyệt" (**Figure 1** của bài báo):

### Hướng 1: Giả định Bình thường (*Normality Assumption* — Autoencoder, GAN, Transformer, Diffusion thuần túy)
* **Cách làm:** Giả sử 100% tập Train đều sạch, dạy AI học vẽ lại toàn bộ dữ liệu đó.
* **Tử huyệt — Nhiễm bẩn bất thường (*Anomaly Contamination*):** Dữ liệu thu thập hàng tháng trời luôn bị lẫn sẵn một số đoạn lỗi thật không ai biết. Mạng học sâu quá thông minh sẽ học thuộc lòng luôn cách vẽ lại các lỗi nhiễm bẩn này $\implies$ Đến lúc chạy thật gặp lại lỗi đó, nó vẽ lại đẹp như thường $\implies$ **Bỏ sót lỗi thật (FNR cao)**.

### Hướng 2: Giả định Bất thường Nhị phân (*Binary Anomaly Assumption* — NCAD, COUTA, CutAddPaste)
* **Cách làm:** Cắt ghép dữ liệu gốc tạo "Bất thường giả" rồi dạy AI phân loại nhị phân 2 lớp: `0` (Gốc = Bình thường) và `1` (Cắt ghép = Lỗi).
* **3 Tử huyệt chí mạng:**
  1. **Khoảng trống đa dạng (*Diversity Gap*):** Chỉ dùng 1–2 kiểu biến đổi không thể phủ kín muôn hình vạn trạng lỗi thực tế, tạo lỗ hổng kiến thức khiến AI vẫn bị lừa bởi lỗi nhiễm bẩn và lỗi hình thái lạ.
  2. **Bất thường giả vô hại (*False Anomaly / Pseudo-normal*):** Cắt ghép hoặc làm trơn ngẫu nhiên nhiều khi tạo ra đoạn sóng **nhìn chẳng khác gì bình thường**. Ép AI phân biệt mẫu này với mẫu bình thường khiến AI học vẹt (*Overfitting*) và **báo động giả liên tục (FPR cao)**.
  3. **Quá tự tin vào nhãn cứng (*Overconfidence on Labels*):** Gán nhãn cứng tuyệt đối `[1, 0]` và `[0, 1]` bỏ qua thực tế là tập gốc có lỗi nhiễm bẩn và tập nhân tạo có *False Anomaly*.

### 💡 Giải pháp của REDLAMP: Giả định Bất thường Đa lớp (*Multiclass Anomaly Assumption*)
REDLAMP hóa giải trọn vẹn cả 4 tử huyệt bằng **4 vũ khí**:
1. **Tạo 12 Lớp Đa dạng (1 lớp Bình thường + 11 lớp Lỗi giả chuyên biệt):** Lấp kín *Diversity Gap* và tạo không gian tiềm ẩn có khả năng **giải thích hình thái lỗi (XAI)**.
2. **Mặt nạ che lỗi (*Anomaly Mask* $W$):** Che vùng lỗi giả khi dạy Decoder, ép Decoder chỉ học vẽ phần bình thường.
3. **Nhãn mềm (*Backward Correction* $\tilde{y}$):** Không tin 100% vào nhãn cứng — luôn chừa `11%` khả năng mẫu lỗi giả thực ra là bình thường và `11%` khả năng mẫu gốc bị nhiễm bẩn.
4. **Bộ lọc tần suất lúc kiểm thử (*FAA*):** Lớp lỗi giả nào xuất hiện $> 5\%$ toàn thời gian Test thì đích thị là lớp *False Anomaly* vô hại $\implies$ Tự động gạch điểm lớp đó về `0`!

---

# PHẦN 4: GIẢI PHẪU 12 LỚP BẤT THƯỜNG GIẢ, MẶT NẠ $W$ & LIÊN HỆ SỰ CỐ ĐỜI THỰC

## 4.1. Bảng Tra Cứu Nhanh 12 Phép Biến Đổi & Quy Tắc Mặt Nạ $W$
Giả sử đoạn sóng gốc gồm 6 điểm: `[10, 12, 14, 16, 18, 20]`, máy chọn ngẫu nhiên khúc giữa `st = 2` đến `ed = 5` (tức `[14, 16, 18]`) để tiêm lỗi giả. Khi đó **mặt nạ $W = [0, 0, \mathbf{1, 1, 1}, 0]$** (trong bài báo $W=1$ đánh dấu vùng lỗi; lưu ý trong code `loader_aug.py`, biến `anomaly_mask` lưu trực tiếp $1 - W$, tức bằng `0` ở vùng lỗi và `1` ở vùng bình thường):

| Thứ tự (Code / Paper) | Tên lớp | Công thức toán học | Ví dụ biến đổi từ `[10, 12, 14, 16, 18, 20]` & Mặt nạ $W$ |
| :---: | :--- | :--- | :--- |
| **`0` / $k=1$** | **`normal`** | $X_{\text{dim}}^{(t,1)} = X_{\text{dim}}^{(t)}$ | Giữ nguyên 100%: `[10, 12, 14, 16, 18, 20]`. Toàn bộ $W = [0, 0, 0, 0, 0, 0]$. |
| **`1` / $k=2$** | **`spike`** | $X_{\text{dim},\text{st}}^{(t)} + a, \; a \sim \mathcal{N}(0, 1)$ | Cộng gai nhọn tại **đúng 1 điểm `st`**: `[10, 12, 64, 16, 18, 20]`. $W$ chỉ bật `1` tại đúng điểm `st`. |
| **`2` / $k=3$** | **`flip`** | $X_{\text{dim},\text{st}:\text{ed}}^{(t)}[::-1]$ | Lật ngược thứ tự thời gian đầu–đuôi: `[10, 12, 18, 16, 14, 20]`. |
| **`3` / $k=4$** | **`speedup`** | Lấy mẫu `[::2]` ($2\times$) hoặc giãn nội suy ($0.5\times$) | Tua nhanh gấp đôi hoặc làm chậm một nửa tần số nhịp điệu sóng trong đoạn `[st:ed]`. |
| **`4` / $k=5$** | **`noise`** | $X_{\text{dim},\text{st}:\text{ed}}^{(t)} + a, \; a \sim \mathcal{N}(0, 0.1I)$ | Cộng nhiễu trắng Gaussian làm đường sóng răng cưa lăn tăn: `[10, 12, 14.3, 15.6, 18.4, 20]`. |
| **`5` / $k=6$** | **`cutoff`** | $\text{Uni}(\min, \max)$ hằng số | Cắt cụt thành vạch ngang phẳng lì ở một mức cố định: `[10, 12, 15, 15, 15, 20]`. |
| **`6` / $k=7$** | **`average`** | $\text{MovingAverage}(w = L/5 = 20)$ | Làm trơn/san phẳng các dao động chi tiết bằng trung bình trượt 20 điểm. |
| **`7` / $k=8$** | **`scale`** | $a \times X_{\text{dim},\text{st}:\text{ed}}^{(t)}, \; a \sim \mathcal{N}(1, 1)$ | Phóng đại hoặc thu nhỏ biên độ (ví dụ $a=2 \to$ `[10, 12, 28, 32, 36, 20]`). |
| **`8` / $k=9$** | **`wander`** | Cộng `linspace(0, a)` rồi giữ độ lệch $+a$ đến cuối | Trôi đường cơ sở (*Baseline Drift*): đoạn sau cũng bị kênh lên $+a$, nên **$W = 1$ từ `st` đến tận cuối cửa sổ!** |
| **`9` / $k=10$** | **`contextual`** | $a \times X_{\text{dim},\text{st}:\text{ed}}^{(t)} + b$ | Biến đổi ngữ cảnh: vừa nhân biên độ $a$ vừa cộng độ lệch nền $b$. |
| **`10` / $k=11$** | **`upsidedown`** | $2 \times \text{Mean} - X_{\text{dim},\text{st}:\text{ed}}^{(t)}$ | Lật úp đỉnh sóng thành đáy sóng qua trục trung bình cộng của đoạn. |
| **`11` / $k=12$** | **`mixture`** | $X_{\text{dim},\text{st}:\text{ed}}^{(t')}$ ($t' \neq t$) | Tráo đoạn: cắt một đoạn từ cửa sổ ở thời điểm $t'$ khác ghép đè vào cửa sổ hiện tại $t$. |

---

## 4.2. Liên Hệ Thực Tế Ngoài Đời Của 11 Lớp Bất Thường Giả (Ví Dụ Nhiệt Độ CPU $\mathbf{Z} = [10, \mathbf{20, 30, 40}, 10]$)

* **Nhóm A — Biến đổi Điểm & Nhiễu:**
  1. **`1. spike` (`[10, 20, 95, 40, 10]`):** Xung điện áp tăng vọt tức thời, tia bức xạ vũ trụ bắn trúng cảm biến vệ tinh NASA, hoặc dội ping mạng (*ping spike*) trong 1 giây.
  2. **`4. noise` (`[10, 23, 27, 44, 10]`):** Dây cáp cảm biến bị lỏng mối nối hoặc bị nhiễu điện từ trường công nghiệp.
* **Nhóm B — Biến đổi Hình học & Chiều hướng:**
  3. **`5. cutoff` (`[10, 15, 15, 15, 10]`):** Cảm biến bị "đơ/treo" phần cứng ở mức bão hòa, hoặc dịch vụ web bị sập nên lưu lượng rơi về hằng số.
  4. **`2. flip` (`[10, 40, 30, 20, 10]`):** Quy trình đóng/mở van công nghiệp hoặc chu trình nạp/xả bị đảo ngược thứ tự vận hành.
  5. **`10. upsidedown` (đỉnh `[20, 50, 20]` lật thành hố `[40, 10, 40]`):** Đấu ngược cực âm/dương cảm biến, hoặc lưu lượng tụt đáy đúng vào khung giờ cao điểm lẽ ra phải đạt đỉnh.
* **Nhóm C — Biến đổi Tần số & Độ mượt:**
  6. **`3. speedup` (`[10, 40, 10, 40, 10]`):** Động cơ quay quá tốc độ định mức, nhịp tim đập nhanh gấp đôi, hoặc tác vụ *cronjob* bị lỗi lặp tần suất $2\times$.
  7. **`6. average` (nhấp nhô `[15, 45, 15]` bị san phẳng `[23, 25, 23]`):** Cảm biến giảm độ nhạy không bắt kịp dao động nhanh. *(Chú ý: Nếu chuỗi gốc vốn đã phẳng sẵn như `[20, 20, 20]`, phép `average` sinh ra mẫu giống hệt bình thường — đây chính là nguồn gốc điển hình của **`False Anomaly`**!).*
* **Nhóm D — Biến đổi Biên độ, Xu hướng & Ngữ cảnh:**
  8. **`7. scale` (`[10, 40, 60, 80, 10]`):** Sai lệch hệ số khuếch đại (*gain*) của thiết bị đo hoặc tải máy chủ tăng gấp đôi đồng loạt.
  9. **`8. wander` (`[10, 30, 50, 70, 40]`):** Hiện tượng **rò rỉ bộ nhớ (*Memory Leak*)** làm mức RAM dâng cao dần lên theo thời gian mà không hạ xuống.
  10. **`9. contextual` (`[10, 35, 40, 45, 10]`):** CPU chạy `45%` là hợp lệ lúc 12 giờ trưa, nhưng xuất hiện lúc 3 giờ sáng (lẽ ra nghỉ ở `10%`) thì là bất thường ngữ cảnh.
  11. **`11. mixture` (`[10, 8, 9, 8, 10]`):** Lỗi lệch đồng hồ đồng bộ (*clock desynchronization*) hoặc mất gói tin làm hai đoạn dữ liệu lệch pha ghép nối tiếp vào nhau.

---

# PHẦN 5: KIẾN TRÚC `1D-CNN` / `ConvAEC` & GIẢI PHẪU CÔNG THỨC TOÁN HỌC TỪ (1) ĐẾN (8)

## 5.1. Hiểu Nhanh `1D-CNN` Là Gì & Tại Sao Tác Giả Chọn `ConvAEC`?

### 1) Cách `1D-CNN` hoạt động trong 10 giây
`1D-CNN` dùng các bộ lọc nhỏ ("kính lúp" `kernel`) trượt từ trái sang phải dọc theo trục thời gian. Ví dụ chuỗi 6 giây có gai nhọn `[10, 10, 50, 10, 10, 10]` đi qua kính lúp săn gai nhọn `[-1, +2, -1]`:
* Ở đoạn phẳng `[10, 10, 10]`: $10\times(-1) + 10\times 2 + 10\times(-1) = \mathbf{0}$ (im lặng).
* Khi trượt trúng đỉnh `[10, 50, 10]`: $10\times(-1) + 50\times 2 + 10\times(-1) = \mathbf{+80}$ (kích hoạt vọt cao báo hiệu bắt trúng hình dáng gai nhọn!).
* **Lưu ý quan trọng:** Dù dữ liệu đa biến có tới `38 dims` (như `SMD`), mỗi kính lúp `kernel = 4` có kích thước `4 × 38` ôm trọn cả 38 cảm biến cùng lúc và **chỉ trượt theo 1 hướng duy nhất là trục thời gian**, nên vẫn gọi là **`1D-CNN`**.

### 2) Cấu trúc 3 khối của mạng `ConvAEC` (`RedLamp/models/meta.py` & `cnn.py`)
1. **`ConvEncoder`:** Đưa cửa sổ `[1, 100, D]` qua 4 khối `ConvBlock` (`filters = [128, 128, 256, 256]`, `kernel = 4`, `stride = 2`, `padding = 2`, `BatchNorm1d`, `ReLU`, `Dropout = 0.2`) nén dần độ dài thời gian `100 -> 51 -> 26 -> 14 -> 8`, sau đó qua `Global MaxPool1D` (về `[1, 256, 1]`) và `Conv1x1` tạo ra **véc-tơ `Embedding` 128 chiều (`[1, 128, 1]`)**.
2. **`ConvDecoder`:** Nén ngược từ `Embedding 128 chiều` qua tầng `Linear` upsampling, 4 khối `ConvTranspose1d` đối xứng `[256, 256, 128, 128]` và `Conv1x1` để khôi phục cửa sổ $\hat{X} \in \mathbb{R}^{1 \times 100 \times D}$.
3. **`NonLinear Classifier`:** Từ cùng `Embedding 128 chiều`, đi qua mạng MLP 2 tầng (`Linear(128 -> 32)` $\to$ `BatchNorm` $\to$ `ReLU` $\to$ `Dropout` $\to$ `Linear(32 -> 12)` $\to$ `Softmax`) dự đoán phân phối xác suất 12 lớp $\hat{y} \in \mathbb{R}^{1 \times 12}$.

### 3) Năm lý do tác giả chọn `ConvAEC` thay vì `Transformer`, `LSTM/RNN` hay `Diffusion`
1. **Cần sự cộng hưởng của cả 2 nhánh (`AE + Classifier`):** Nếu bỏ Classifier (`w/o L_CE`), mô hình mất khả năng bắt lỗi hình thái tinh vi (giảm `12.4% – 14.8%` VUS-PR); nếu bỏ Decoder (`w/o L_MSE`), hiệu năng sụt tới `23.7%` trên SMD.
2. **Bắt hình thái cục bộ & Bất biến vị trí lỗi:** `1D-CNN + Global MaxPool1D` giúp phát hiện biến dạng dù nó rơi vào đầu, giữa hay cuối cửa sổ 100 điểm.
3. **Song song hóa cực nhanh cho dữ liệu nhân bản gấp $12\times$:** Sinh 12 lớp làm tập Train phình gấp 12 lần — `1D-CNN` xử lý song song toàn cửa sổ nhanh vượt trội so với tính toán tuần tự của `LSTM/RNN` hay khử nhiễu lặp của `Diffusion`.
4. **Không bị học vẹt dữ liệu bẩn hay "đói dữ liệu" như `Transformer`:** Hội tụ ổn định trên cả chuỗi ngắn (~2,400 điểm ở SMAP/MSL) chỉ với **1 bộ siêu tham số duy nhất** cho cả 5 tập dữ liệu.
5. **Kế thừa kiến trúc chống nhiễu nhãn `SREA` (ECML 2021):** Phối hợp tự nhiên với cơ chế Nhãn mềm (*Backward Correction*).

---

## 5.2. Công Thức (1) — Nhãn Mềm Hiệu Chỉnh Ngược (*Backward Correction*)
$$\tilde{y}_i^{(t,k)} = \begin{cases} (1 - \alpha - K\beta)y_i^{(t,k)} + \alpha + \beta, & \text{nếu } i = 1 \text{ (vị trí lớp Normal)}, \\ (1 - \alpha - K\beta)y_i^{(t,k)} + \beta, & \text{nếu } i \in \{2, \dots, 12\} \text{ (vị trí 11 lớp Lỗi)}, \end{cases} \quad (1)$$

* Với $K = 12$, $\alpha = 0.10$ (xác suất mẫu lỗi giả thực chất là *False Anomaly* bình thường), $\beta = 0.01$ (xác suất nhầm lẫn giữa mọi lớp), ta có $(1 - \alpha - K\beta) = 1 - 0.10 - 0.12 = \mathbf{0.78}$:
  * **Mẫu thuộc lớp lỗi (ví dụ `Spike`, $y_2 = 1$):** Ô `Normal` nhận $\tilde{y}_1 = 0.10 + 0.01 = \mathbf{0.11}$ (`11%`); ô `Spike` nhận $\tilde{y}_2 = 0.78 + 0.01 = \mathbf{0.79}$ (`79%`); 10 ô lỗi còn lại mỗi ô nhận $\mathbf{0.01}$ (`1%`). Tổng $= 1.00$.
  * **Mẫu thuộc lớp gốc (`Normal`, $y_1 = 1$):** Ô `Normal` nhận $\tilde{y}_1 = 0.78 + 0.10 + 0.01 = \mathbf{0.89}$ (`89%`); 11 ô lỗi mỗi ô nhận $\mathbf{0.01}$ (`1%` $\times 11 = 11\%$ dự phòng khả năng mẫu gốc bị **nhiễm bẩn bất thường**!).

---

## 5.3. Công Thức (2), (3), (4) — Hàm Mất Mát Khi Huấn Luyện (*Objective Function*)
1. **Hàm mất mát tái tạo có mặt nạ — Công thức (2):**
   $$\mathcal{L}_{\text{MSE}} = \left\| \left(1 - W^{(t,k)}\right) \odot \left(X^{(t,k)} - \hat{X}^{(t,k)}\right) \right\|_2^2 \quad (2)$$
   * *Ví dụ 4 con số:* Gốc `[10, 10, 10, 10]` chèn `Spike` thành $X = [10, 10, \mathbf{50}, 10]$ với $(1 - W) = [1, 1, \mathbf{0}, 1]$. Nếu Decoder vẽ đường chuẩn $\hat{X} = [10, 9, \mathbf{10}, 10]$, độ lệch $(X - \hat{X}) = [0, 1, \mathbf{40}, 0]$ sau khi nhân với $(1 - W)$ thành $[0, 1, \mathbf{0}, 0] \implies \mathcal{L}_{\text{MSE}} = 1$. **Decoder không bị phạt khi từ chối vẽ gai nhọn `50`**, nên chỉ học vẽ tín hiệu bình thường!
2. **Hàm mất mát phân loại nhãn mềm — Công thức (3):**
   $$\mathcal{L}_{\text{CE}} = -\sum_{i=1}^{K} \tilde{y}_i^{(t,k)} \log\left(\hat{y}_i^{(t,k)}\right) \quad (3)$$
3. **Hàm mất mát tổng hợp — Công thức (4):**
   $$\mathcal{L} = \gamma \mathcal{L}_{\text{CE}} + (1 - \gamma) \mathcal{L}_{\text{MSE}}, \quad \text{với } \gamma = 0.10 \quad (4)$$

---

## 5.4. Công Thức (5), (6), (7), (8) — Chấm Điểm Bất Thường Lúc Kiểm Thử & Bộ Lọc FAA
1. **Điểm sai số tái tạo — Công thức (5):**
   $$s_{\text{MSE}}(t) = \left\| X^{(t)} - \hat{X}^{(t)} \right\|_2^2 \quad (5)$$
2. **Bộ lọc FAA & Điểm phân loại điều chỉnh — Công thức (6) & (7):**
   $$\text{FAA}\left(\hat{y}_i^{(t)}\right) = \begin{cases} 0, & \text{nếu } \frac{1}{T - L + 1}\sum_{j=L}^{T} \hat{y}_i^{(j)} > \delta, \\ \hat{y}_i^{(t)}, & \text{ngược lại}, \end{cases} \qquad s_{\text{CE}}(t) = \sum_{i=2}^{K} \text{FAA}\left(\hat{y}_i^{(t)}\right) \quad (6, 7)$$
   * Với $\delta = 0.05$ (`5%`), lớp `normal` ($i=1$) luôn bị bỏ qua, và bất kỳ lớp lỗi giả nào có xác suất trung bình toàn chuỗi Test $> 5\%$ sẽ bị coi là *Pseudo-normal class* và gạch điểm về `0`.
3. **Điểm bất thường tổng hợp cuối cùng — Công thức (8):**
   $$s(t) = \frac{1}{2}\text{MinMax}_T\left(s_{\text{MSE}}(t)\right) + \frac{1}{2}\text{MinMax}_T\left(s_{\text{CE}}(t)\right) \quad (8)$$
   *(Trong code chính thức và bài tái lập của nhóm, điểm số được làm mượt thêm bằng bộ lọc trung bình trượt cửa sổ $w = L/2 = 50$ bước).*

---

# PHẦN 6: TẠI SAO DÙNG `VUS-PR` / `VUS-ROC`, TÓM TẮT THỰC NGHIỆM Q1–Q5 & SIÊU THAM SỐ

## 6.1. Tại sao bài báo dùng `VUS-PR` thay vì `Accuracy` hay `F1-PA`?
1. **Nghịch lý `Accuracy = 99.15%`:** Trong tập UCR, lỗi chỉ chiếm `0.85%` thời gian (còn `99.15%` là bình thường). Một mô hình "lười biếng" lúc nào cũng phán *Bình thường* sẽ đạt ngay `Accuracy = 99.15%` dù bỏ sót 100% sự cố!
2. **Ảo tưởng của `F1-PA` (*Point Adjustment*):** Chỉ cần chạm trúng 1 điểm trong đoạn lỗi dài là tính đúng cả đoạn, làm phồng kết quả ngay cả với điểm số ngẫu nhiên.
3. **Ưu thế của `VUS-PR` (*Volume Under the Surface — Precision-Recall*):** Tính điểm theo phân đoạn có vùng đệm (buffer) quét liên tục tạo thành mặt cong 3D trên mọi ngưỡng cắt, vừa dung sai độ lệch biên của cửa sổ trượt, vừa cực kỳ nhạy với báo động giả khi tỷ lệ lỗi hiếm (`0.85% – 13%`).

## 6.2. Bốn Kết Quả Thực Nghiệm Nổi Bật Nhất Trong Bài Báo (Mục 4.2 – 4.6)
* **Q1 (Effectiveness):** Dẫn đầu trung bình trên 5 bộ dữ liệu với **`VUS-ROC = 0.803`**, **`VUS-PR = 0.430`**, **`RF = 0.219`** (vượt phương pháp đứng nhì *CutAddPaste* tới **+17.4%** về `VUS-PR`).
* **Q2 (Ablation Study):** Gộp 11 lớp lỗi về phân loại nhị phân (`Binary`) làm `VUS-PR` sụt giảm **15.3%** (trên UCR) và **24.5%** (trên SMD); tắt `AM`, `BC`, `FAA`, $\mathcal{L}_{\text{MSE}}$ hay $\mathcal{L}_{\text{CE}}$ đều làm giảm điểm rõ rệt.
* **Q4 (Robustness):** Khi trộn **5% lỗi nhiễm bẩn vào tập Train** của UCR, REDLAMP chỉ giảm nhẹ **1.8%** `VUS-PR`, trong khi *CutAddPaste* sụt giảm tới **15.5%**.
* **Q5 (Explainability):** Biểu đồ t-SNE 2D của `Embedding 128 chiều` cho thấy lỗi thật gom cụm ngay sát lớp lỗi giả tương đồng về hình thái vật lý (EPG gần `Speedup`, ABP gần `Noise`, SMD gần `Spike` và `Cutoff`).

## 6.3. Bảng Tổng Hợp Siêu Tham Số Cố Định Duy Nhất Cho Mọi Bộ Dữ Liệu (Table 4)
| Tên tham số | Ký hiệu | Giá trị mặc định | Ý nghĩa dễ hiểu |
| :--- | :---: | :---: | :--- |
| Kích thước cửa sổ trượt | $L$ ($W$) | `100` | Mỗi lần AI nhìn một đoạn gồm 100 bước thời gian liên tiếp. |
| Số chiều không gian ẩn (Embedding) | `emb_dim` | `128` | Cửa sổ 100 điểm được Encoder tóm tắt thành đúng 128 con số. |
| Số bộ lọc 4 tầng Conv1D | `filters` | `[128, 128, 256, 256]` | Số lượng "kính lúp" qua 4 tầng tích chập (`kernel = 4`, `stride = 2`). |
| Tổng số lớp phân loại | $K$ | `12` | Gồm 1 lớp `Normal` + 11 lớp bất thường giả. |
| Xác suất *False Anomaly* | $\alpha$ | `0.10` ($10\%$) | Cộng bù xác suất về lớp `Normal` trong Công thức (1). |
| Xác suất nhầm lẫn giữa mọi lớp | $\beta$ | `0.01` ($1\%$) | Chia đều xác suất dự phòng cho cả 12 lớp trong Công thức (1). |
| Trọng số hàm mất mát phân loại | $\gamma$ | `0.10` | Dùng trong Công thức (4): $\mathcal{L} = 0.1\mathcal{L}_{\text{CE}} + 0.9\mathcal{L}_{\text{MSE}}$. |
| Ngưỡng lọc lớp giả bình thường FAA | $\delta$ | `0.05` ($5\%$) | Lớp lỗi giả xuất hiện $> 5\%$ thời gian Test sẽ bị gạch điểm về `0`. |

---

# PHẦN 7: TỔNG QUAN 3 TAB CỦA ỨNG DỤNG WEB DEMO (`REDLAMP DEMO`)

Khi khởi chạy [`../run_demo.bat`](../run_demo.bat) (mở `http://127.0.0.1:8000`), giao diện Web Demo gồm **3 Tab trên thanh Header** tương ứng với 3 chức năng cốt lõi:

1. **Tab 1 — `1. Chọn Pretrain & Mô hình`:**
   * Chọn 1 trong 5 bộ dữ liệu (`SMD`, `AIOps / IOPS`, `MSL`, `SMAP`, `UCR`), chọn `Entity` và `Seed (0 - 4)`.
   * Xem bảng **Thông số cấu hình mô hình** (`ConvAEC`, `dims`, `W = 100`, `Embedding = 128`), bảng **3 chỉ số đánh giá thực tế (`VUS-ROC`, `VUS-PR`, `RF` kèm `mean ± std` của 5 seeds)** và **Biểu đồ tổng quan toàn chuỗi thời gian**.
2. **Tab 2 — `2. Nhập điểm & Dự đoán` (Live Inference):**
   * Ghép 1 điểm dữ liệu mới nhập tay (`1` hoặc `D` chiều) vào sau 99 điểm ngữ cảnh trước đó, điều chỉnh ngưỡng quyết định (mặc định phân vị `P99.5`) và chạy mô hình `ConvAEC` thời gian thực.
   * Xuất ngay kết luận **`TRẠNG THÁI BÌNH THƯỜNG`** / **`PHÁT HIỆN BẤT THƯỜNG`**, nhóm bất thường giả tương đồng nhất, 4 thẻ điểm thô (`MSE`, `CE`, `Điểm kết hợp thô`, `Ngưỡng P99.5`), bảng các lớp bị loại bởi FAA, đồ thị tái tạo cửa sổ 100 điểm và nút **`🔍 Xem giải thích cho điểm này (Đồng bộ)`**.
3. **Tab 3 — `3. Trình bày Giải thích` (Walkthrough 5 Bước XAI):**
   * Trình chiếu trực quan từng bước bên trong mạng (hỗ trợ phím mũi tên `←` / `→`): **Bước 1 (Đầu vào 100 điểm)** $\to$ **Bước 2 (Encoder 4 khối Conv1D & Embedding 128D)** $\to$ **Bước 3 (Decoder & đối chứng MSE)** $\to$ **Bước 4 (Classifier 12 lớp & lọc FAA)** $\to$ **Bước 5 (Điểm kết hợp theo thời gian & Công thức số thực tế)**.

👉 **Để xem kịch bản bấm máy từng bước, lời dẫn và từng con số thực tế dùng khi đứng thuyết trình trước hội đồng**, hãy mở tệp: **[`RedLamp_Demo_Presentation.md`](./RedLamp_Demo_Presentation.md)**.
