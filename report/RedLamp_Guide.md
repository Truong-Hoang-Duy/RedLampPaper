# 📘 CẨM NANG TOÀN DIỆN REDLAMP (KDD '25): LÝ THUYẾT, CÔNG THỨC TOÁN HỌC, VÍ DỤ CON SỐ & HƯỚNG DẪN 3 TAB WEB DEMO

**Tên bài báo:** *Robust and Explainable Detector of Time Series Anomaly via Augmenting Multiclass Pseudo-Anomalies (REDLAMP)*  
**Tác giả:** Kohei Obata, Yasuko Matsubara, Yasushi Sakurai (Đại học Osaka, Nhật Bản) — **Hội nghị KDD '25**  

> **🎯 Mục tiêu của tài liệu này:**  
> Tài liệu này hợp nhất toàn bộ phần **giải nghĩa lý thuyết – công thức toán học (1) đến (8) dành cho người mới bắt đầu** và **hướng dẫn chi tiết 3 Tab của ứng dụng Web Demo (`REDLAMP DEMO`)**. Mọi từ viết tắt, ký hiệu toán học, 12 lớp bất thường giả và cơ chế bên trong mạng `ConvAEC` đều được minh họa bằng **ví dụ con số cụ thể (nhẩm tính được ngay)** để phục vụ báo cáo và thuyết trình trực tiếp trên máy chiếu.

---

# PHẦN 1: TỪ ĐIỂN GIẢI NGHĨA TẤT CẢ CÁC TỪ VIẾT TẮT & THUẬT NGỮ (TRA CỨU NHANH)

Khi đọc bài báo hoặc trình chiếu giao diện Web Demo, bạn sẽ liên tục gặp các chữ viết tắt và thuật ngữ dưới đây:

## 1.1. Các thuật ngữ của riêng phương pháp REDLAMP
| Viết tắt | Tiếng Anh đầy đủ | Nghĩa Tiếng Việt | Giải thích bình dân trong 1 câu |
| :--- | :--- | :--- | :--- |
| **TSAD** | *Time Series Anomaly Detection* | Phát hiện bất thường chuỗi thời gian | Bài toán tìm xem ở giây/phút thứ mấy thì cảm biến (nhiệt độ, nhịp tim, CPU...) bị nhảy số bất thường. |
| **REDLAMP** | *Robust and Explainable Detector of Time Series Anomaly via Augmenting Multiclass Pseudo-Anomalies* | Bộ phát hiện bất thường bền vững & có tính giải thích bằng cách tạo bất thường giả đa lớp | Tên mô hình của bài báo: tự tạo ra **11 kiểu lỗi giả** để dạy AI vừa biết bắt lỗi, vừa chỉ ra được lỗi đó thuộc hình dạng nào. |
| **AM** | *Anomaly Mask* ($W$) | Mặt nạ đánh dấu vùng lỗi | Một tấm bìa chỉ gồm số `0` và `1`, đánh dấu chính xác khúc nào vừa bị chèn lỗi giả để "che mắt" không cho bộ vẽ lại (Decoder) học vẹt lỗi đó. |
| **BC** | *Backward Correction* | Hiệu chỉnh ngược (Nhãn mềm) | Thay vì bắt AI tin tuyệt đối 100% rằng mẫu tạo ra là lỗi (`[0, 1]`), ta giảm xuống còn 79% và luôn chừa 11% khả năng nó vẫn là bình thường (`[0.11, 0.79, ...]`). |
| **FAA** | *Frequent Anomaly Adjustment* | Bộ lọc bất thường xuất hiện quá thường xuyên | Quy tắc: *"Bất thường thật thì phải hiếm"*. Nếu lúc chạy thực tế mà AI bảo một loại lỗi xuất hiện quá 5% tổng thời gian ($\delta = 0.05$), bộ lọc FAA sẽ coi đó là báo động giả và gạch điểm của loại đó về `0`. |

## 1.2. Các thuật ngữ Mô hình Học sâu (Deep Learning) & Hàm mất mát (Loss)
| Viết tắt | Tiếng Anh đầy đủ | Nghĩa Tiếng Việt | Giải thích bình dân trong 1 câu |
| :--- | :--- | :--- | :--- |
| **ConvAEC** | *Convolutional Autoencoder + NonLinear Classifier* | Mạng Tự mã hóa Tích chập kết hợp Bộ phân loại Phi tuyến | Tên kiến trúc mạng trong code (`RedLamp/models/meta.py`), gồm 3 khối: `ConvEncoder`, `ConvDecoder` và `NonLinear Classifier`. |
| **AE** | *Autoencoder* | Mạng tự mã hóa (Nén & Khôi phục) | Mạng gồm 2 nửa: **Encoder** (nén đoạn sóng dài thành vài con số tóm tắt) và **Decoder** (từ vài con số đó vẽ khôi phục lại đoạn sóng ban đầu). |
| **1D-CNN** | *1D Convolutional Neural Network* | Mạng tích chập 1 chiều | Mạng dùng một "kính lúp" trượt dọc theo trục thời gian để nhận diện hình dáng sóng (gai nhọn, dốc lên, sóng sin...). |
| **MLP** | *Multi-Layer Perceptron* (`NonLinear Classifier`) | Mạng nơ-ron đa tầng (Bộ phân loại) | Khối mạng đóng vai **"Bác sĩ chẩn đoán"**: nhận bản tóm tắt từ Encoder rồi phán đoán đoạn sóng thuộc lớp nào trong 12 lớp. |
| **Embedding** | *Latent Embedding Vector* | Vector biểu diễn tiềm ẩn (128 số) | **"Tờ phiếu khám bệnh 128 con số"** do Encoder tóm tắt từ đoạn sóng 100 điểm đầu vào. Từ tờ phiếu này, Classifier sẽ đoán bệnh và Decoder sẽ vẽ lại sóng. |
| **MSE** | *Mean Squared Error* ($\mathcal{L}_{\text{MSE}}$) | Sai số bình phương trung bình | Đo xem **hình vẽ lại của Decoder khác hình gốc bao nhiêu**: lấy `(Gốc - Vẽ lại)` bình phương lên. Nếu lệch nhiều $\implies$ MSE cao $\implies$ có bất thường! |
| **CE** | *Cross-Entropy Loss* ($\mathcal{L}_{\text{CE}}$) | Hàm mất mát Entropy chéo | Đo xem **Classifier đoán 12 lớp sai lệch bao nhiêu** so với nhãn mục tiêu. Đoán càng đúng lớp thì CE càng nhỏ (tiến về 0). |
| **Softmax** | *Softmax Activation* | Hàm chuẩn hóa xác suất | Biến 12 điểm số thô ở đầu ra của Classifier thành **12 con số xác suất cộng lại đúng bằng 100% (`1.0`)**. |

## 1.3. Các thuật ngữ Đánh giá Kết quả (Metrics) & Phân tích
| Viết tắt | Tiếng Anh đầy đủ | Nghĩa Tiếng Việt | Giải thích bình dân trong 1 câu |
| :--- | :--- | :--- | :--- |
| **FPR** | *False Positive Rate* | Tỷ lệ Dương tính giả (**Báo động giả**) | Máy đang chạy bình thường nhưng AI lại hú còi báo lỗi. |
| **FNR** | *False Negative Rate* | Tỷ lệ Âm tính giả (**Bỏ sót lỗi**) | Máy đang bị hỏng thật nhưng AI lại im lặng bảo bình thường. |
| **ROC / PR** | *Receiver Operating Characteristic / Precision-Recall* | Đường cong ROC và Đường cong Độ chính xác - Độ phủ | Hai biểu đồ chuẩn để đo chất lượng bộ phát hiện lỗi mà không phụ thuộc vào việc chọn ngưỡng cắt cố định. Khi lỗi rất hiếm (như trong chuỗi thời gian), **PR** đánh giá khắt khe và chuẩn xác hơn **ROC**. |
| **R-AUC-ROC / R-AUC-PR** | *Range Area Under the Curve* | Diện tích dưới đường cong tính theo khoảng (Range) | Phiên bản nâng cấp của AUC dành riêng cho chuỗi thời gian: tính điểm theo **cả một đoạn thời gian bị lỗi** (có vùng đệm xung quanh) thay vì chỉ chấm từng giây rời rạc. |
| **VUS-ROC / VUS-PR** | *Volume Under the Surface* | Thể tích dưới bề mặt (Thước đo chuẩn nhất KDD '25) | Thay vì chọn cố định 1 độ dài vùng đệm, VUS quét qua nhiều vùng đệm khác nhau để tạo thành mặt 3D rồi tính thể tích. **Số càng gần `1.0` là mô hình càng xuất sắc.** |
| **RF** | *Range F-score* | Điểm cân bằng F1 theo khoảng thời gian | Điểm trung bình điều hòa giữa Precision (báo động có chuẩn không) và Recall (có bắt hết các đoạn lỗi không). |
| **t-SNE / PCA** | *t-Distributed Stochastic Neighbor Embedding / Principal Component Analysis* | Thuật toán giảm chiều dữ liệu (từ 128D xuống 2D) | Con người không thể nhìn được không gian 128 chiều (128 con số của Embedding), nên ta dùng PCA hoặc t-SNE để "ép phẳng" 128 chiều xuống mặt phẳng 2D ($x, y$) để vẽ biểu đồ phân cụm. |
| **XAI** | *Explainable Artificial Intelligence* | AI có khả năng giải thích | Không chỉ phán "Có lỗi!", AI còn giải thích được **"Lỗi này có hình dạng giống lỗi nhiễu sóng (Noise) hay lỗi kẹt cảm biến (Cutoff)"**. |

## 1.4. Bảng Phân Biệt Nhanh 4 Thuật Ngữ Dễ Nhầm Lẫn Nhất Khi Hội Đồng Hỏi

| Thuật ngữ | Nó là cái gì? | Xuất hiện lúc nào? | Bản chất sự việc (Hiểu trong 1 câu) | Hậu quả gây ra | Cách RedLamp xử lý trên Demo |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`False Anomaly`** *(Bất thường giả sai lệch / Lớp giả bình thường)* | **Mẫu dữ liệu** nhân tạo | Lúc **Huấn luyện** | Biến đổi tín hiệu để tạo lỗi giả (như `average`), nhưng kết quả trông **vẫn giống hệt bình thường**. | Làm tăng **FPR** *(Báo động giả)* | Dùng **Nhãn mềm ($\alpha=0.1$)** lúc học & **Loại bỏ lớp có tần suất trung bình $> 0.05$ (FAA)** ở Bước 4 Tab 2 và Tab 3. |
| **`Anomaly Contamination`** *(Nhiễm bẩn bất thường)* | **Mẫu dữ liệu** thực tế | Lúc **Huấn luyện** | Dữ liệu gốc chưa gán nhãn bị **lẫn sẵn lỗi thật** nhưng máy lại tưởng là bình thường. | Làm tăng **FNR** *(Bỏ sót lỗi thật)* | Dùng **11 lớp biến đổi đa dạng** + **Nhãn mềm ($\beta=0.01$)** + **Mặt nạ tái tạo ($W$)**. |
| **`Điểm kết hợp thô (raw_combined)`** | **Điểm số suy luận** | Lúc **Kiểm thử & Dự đoán** | Trung bình cộng $(\text{MSE} + \text{CE}) / 2$ chưa qua chuẩn hóa Min-Max. | Dùng để so sánh trực tiếp với ngưỡng `P99.5` | Hiển thị ở Bước 5 Tab 2 và 4 ô điểm số ở Tab 3 để kết luận `BẤT THƯỜNG` / `BÌNH THƯỜNG`. |
| **`Điểm bất thường tương đối (anomaly_score)`** | **Điểm chuẩn hóa `[0, 1]`** | Lúc **Đánh giá toàn chuỗi** | Điểm kết hợp thô sau khi chuẩn hóa Min-Max trên toàn chuỗi kiểm thử. | Luôn có đỉnh cực đại bằng `1.0` trên mỗi chuỗi | Vẽ đường màu đỏ trên biểu đồ tổng quan ở Tab 1 và dùng để tính `VUS-ROC`, `VUS-PR`, `RF`. |

## 1.5. Tên 5 Bộ dữ liệu Chuẩn trong Bài báo & trên Web Demo
* **`SMD` (*Server Machine Dataset — 38 dims*):** Gồm 28 máy chủ công nghiệp, mỗi máy chủ theo dõi cùng lúc **$D=38$ chỉ số** (CPU, RAM, Disk, Network...) suốt 5 tuần.
* **`AIOps / IOPS` (*Cloud KPIs — 1 dim*):** Gồm 29 chuỗi đơn biến ($D=1$) theo dõi chỉ số vận hành máy chủ Web đám mây. Tập Train bị lẫn khoảng 2.49% lỗi.
* **`MSL` (*Mars Science Laboratory — 55 dims*):** Gồm 23 thực thể từ Tàu tự hành Sao Hỏa Curiosity của NASA, theo dõi **$D=55$ kênh** cảm biến và bộ truyền động.
* **`SMAP` (*Soil Moisture Active Passive — 25 dims*):** Gồm 54 thực thể từ Vệ tinh đo độ ẩm đất của NASA, theo dõi **$D=25$ kênh**.
* **`Anomaly Archive` (*UCR Anomaly Archive — 1 dim*):** Gồm 250 chuỗi đơn biến ($D=1$) từ y tế (điện tâm đồ ECG, huyết áp ABP), nông nghiệp (côn trùng EPG), công nghiệp... Tập Train hoàn toàn sạch (0% lỗi).

---

# PHẦN 2: CÁCH ĐỌC CÁC KÝ HIỆU TOÁN HỌC TRONG BÀI BÁO (KHÔNG SỢ CÔNG THỨC!)

Trước khi đọc công thức (1) đến (8), bạn chỉ cần nắm **6 quy tắc ký hiệu** cực kỳ đơn giản sau:

1. **Chữ in hoa ($X, W$) vs. Chữ thường ($x, y, s$):**
   * Chữ in hoa như $X^{(t)}$ là một **bảng số (ma trận)** chứa cả một đoạn cửa sổ thời gian.
   * Chữ thường như $s(t)$ là **một con số điểm** hoặc một danh sách ngắn (vector).
2. **Dấu mũ nón ($\hat{X}, \hat{y}$) vs. Dấu ngã ($\tilde{y}$) vs. Không dấu ($X, y$):**
   * **Không dấu ($X, y$):** Dữ liệu gốc ban đầu / Nhãn gốc ban đầu.
   * **Có dấu mũ nón ($\hat{X}, \hat{y}$ — đọc là "X-hat", "y-hat"):** Kết quả **do mạng AI dự đoán ra**. (Cứ thấy dấu `^` trên đầu nghĩa là sản phẩm của AI!).
   * **Có dấu ngã ($\tilde{y}$ — đọc là "y-tilde"):** Nhãn đã được **làm mềm** (chỉnh sửa nhẹ từ nhãn gốc $y$).
3. **Ký hiệu $\in \mathbb{R}^{D \times L}$ (đọc là "thuộc không gian số thực kích thước D nhân L"):**
   * Đừng bận tâm chữ $\mathbb{R}$, nó chỉ có nghĩa là **một bảng Excel có $D$ hàng (số cảm biến) và $L$ cột (số mốc thời gian)**. Ví dụ $D=1, L=100$ nghĩa là một dãy gồm đúng 100 con số liên tiếp.
4. **Ký hiệu $\odot$ hoặc $\circ$ (Phép nhân Hadamard / Element-wise):**
   * Nghĩa là **nhân từng vị trí tương ứng với nhau**.
   * *Ví dụ:* $[2, 5, 8] \odot [1, 0, 1] = [2\times 1, \; 5\times 0, \; 8\times 1] = [2, 0, 8]$.
5. **Ký hiệu $\| A \|_2^2$ (Chuẩn bình phương L2):**
   * Nghĩa là **lấy từng con số trong bảng $A$ bình phương lên (nhân chính nó) rồi cộng tất cả lại**.
   * *Ví dụ:* $\| [3, -2, 1] \|_2^2 = 3^2 + (-2)^2 + 1^2 = 9 + 4 + 1 = 14$. (Bình phương giúp mọi số âm đều thành số dương và phạt nặng những chỗ lệch lớn).
6. **Ký hiệu $a \sim \mathcal{N}(\mu, \sigma^2)$ (Bốc số ngẫu nhiên theo Phân phối chuẩn hình chuông):**
   * Nghĩa là máy tính bốc ngẫu nhiên một con số $a$ xoay quanh tâm trung bình $\mu$ với phương sai $\sigma^2$ (độ lệch chuẩn $\sigma$).
   * **Lưu ý quan trọng:** $\sigma = 1$ **không phải là giới hạn chặn cứng** ở $[-1, +1]$, mà là độ lệch trung bình theo quy tắc hình quả chuông (**Quy tắc 68–95–99.7**). Hãy hình dung tổng diện tích bên dưới quả chuông là **100 viên bi (100%)** được rải xuống:
     * Vì **bụng chuông ở giữa (`-1` đến `+1`) phình cao nhất**, nó chứa được nhiều bi nhất: mỗi nửa bên trái/phải chứa `34.1%` $\implies$ Cộng lại `34.1% + 34.1% = 68.2%` (~68 viên bi rơi vào `[-1, +1]`).
     * Hai **sườn chuông (`-2` đến `-1` và `+1` đến `+2`)** thấp dần xuống, mỗi bên chứa thêm `13.6%` $\implies$ Cộng dồn từ `-2` đến `+2` là: `13.6% + 68.2% + 13.6% = 95.4%` (~95 viên bi rơi vào `[-2, +2]`).
     * Hai **đuôi chuông ngoài cùng (`-3` đến `-2` và `+2` đến `+3`)** dẹt sát mặt đất, mỗi bên chỉ chứa `2.1%` $\implies$ Cộng dồn từ `-3` đến `+3` là `99.7%`.

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

Hãy tưởng tượng bạn cần dạy một AI làm **bảo vệ giám sát nhịp tim hoặc nhiệt độ máy chủ**, nhưng trong dữ liệu lịch sử đưa cho AI học **không hề có ai đánh dấu sẵn chỗ nào là bình thường, chỗ nào là lỗi** (bài toán Không giám sát — *Unsupervised*).

Các nhà khoa học trước nay đi theo 2 hướng chính, nhưng cả 2 đều có "tử huyệt" (minh họa ở **Figure 1** của bài báo):

### Hướng 1: Giả định Bình thường (*Normality Assumption* — dùng Autoencoder, GAN, Transformer thuần túy)
* **Cách làm:** Giả sử 100% dữ liệu trong tập huấn luyện đều là bình thường. Dạy AI học thuộc lòng cách vẽ lại dữ liệu đó. Hy vọng rằng lúc chạy thật, gặp ca bệnh lạ thì AI sẽ vẽ sai và phát hiện ra.
* **Tử huyệt — Nhiễm bẩn bất thường (*Anomaly Contamination*):** Trong thực tế, dữ liệu thu thập hàng tháng trời từ máy chủ **luôn bị lẫn một vài lúc máy bị lỗi mà không ai biết** (gọi là dữ liệu bị nhiễm bẩn). Vì mạng học sâu (Deep Learning) rất thông minh, nó học thuộc lòng luôn cả cách vẽ các cục lỗi nhiễm bẩn này! Đến lúc đi thi gặp đúng lỗi đó, nó vẽ lại đẹp như thường $\implies$ **Bỏ sót lỗi thật (False Negative cao)**.

### Hướng 2: Giả định Bất thường Nhị phân (*Binary Anomaly Assumption* — như NCAD, CutAddPaste)
* **Cách làm:** Tự lấy dữ liệu gốc ra cắt ghép, làm méo đi để tạo thành **"Bất thường giả" (Pseudo-anomalies)**. Sau đó dạy AI phân loại thành **2 nhóm (Nhị phân)**: Nhóm `0` (Dữ liệu gốc = Bình thường) và Nhóm `1` (Dữ liệu cắt ghép = Bất thường).
* **3 Tử huyệt chí mạng:**
  1. **Khoảng trống đa dạng (*Diversity Gap*):** Chỉ dùng 1–2 kiểu cắt ghép thì không thể bắt chước hết muôn hình vạn trạng lỗi ngoài đời thực. Những kiểu lỗi chưa được dạy tạo thành một "lỗ hổng kiến thức" (*Diversity Gap*). Nếu tập Train có lỗi nhiễm bẩn rơi vào lỗ hổng này, AI vẫn bị lừa!
  2. **Bất thường giả vô hại (*False Anomaly / Pseudo-normal*):** Vì máy tính cắt ghép hoàn toàn ngẫu nhiên (chọn bừa đoạn thời gian để làm mượt hoặc cắt dán), có những lúc đoạn bị biến đổi xong **nhìn chẳng khác gì dữ liệu bình thường**! Bài báo gọi đây là **False Anomaly** (mang tiếng là mẫu lỗi giả nhưng thực chất vẫn bình thường). Nếu bắt AI phải tìm cách phân biệt mẫu này với mẫu bình thường, AI sẽ bị *Overfitting* và **báo động giả liên tục (False Positive cao)**.
  3. **Quá tự tin vào nhãn cứng (*Overconfidence on Labels*):** Gán nhãn cứng tuyệt đối `0` (100% bình thường) cho dữ liệu gốc và `1` (100% lỗi) cho dữ liệu cắt ghép là sai lầm, vì dữ liệu gốc có thể chứa lỗi nhiễm bẩn, còn dữ liệu cắt ghép có thể chứa *False Anomaly*.

### 💡 Giải pháp của REDLAMP: Giả định Bất thường Đa lớp (*Multiclass Anomaly Assumption*)
REDLAMP giải quyết trọn vẹn cả 4 tử huyệt trên bằng 4 vũ khí:
1. **Tạo 12 Lớp Đa dạng (1 lớp Bình thường + 11 lớp Lỗi giả chuyên biệt):** Lấp kín *Diversity Gap* và giúp không gian học được có khả năng **giải thích tên loại lỗi (XAI)**.
2. **Mặt nạ che lỗi (*Anomaly Mask* $W$):** Che phần lỗi giả lại khi dạy Decoder vẽ, để Decoder tuyệt đối chỉ học vẽ phần bình thường.
3. **Nhãn mềm (*Backward Correction* $\tilde{y}$):** Không bao giờ tin 100% vào nhãn. Luôn dặn AI: *"Mẫu lỗi giả này vẫn có 11% khả năng là bình thường đấy nhé!"* $\implies$ Trị dứt điểm *Overconfidence*.
4. **Bộ lọc tần suất lúc kiểm thử (*FAA*):** Lúc chạy thực tế, lớp lỗi giả nào bị AI gọi tên quá nhiều ($> 5\%$ tổng thời gian) thì đích thị là lớp *False Anomaly* vô hại $\implies$ Tự động gạch điểm lớp đó về `0`!

---

# PHẦN 4: HIỂU TRỌN VẸN LUỒNG THUẬT TOÁN QUA VÍ DỤ GIẢ LẬP 5 CON SỐ & 12 LỚP BẤT THƯỜNG GIẢ

Hãy tưởng tượng chúng ta theo dõi **nhiệt độ CPU của 1 chiếc máy chủ** theo từng giây. Bình thường máy chạy êm ở khoảng **`10°C – 40°C`**.

## 4.1. Thu thập dữ liệu & Cắt thành "Cửa sổ trượt" (`Sliding Window`)

Giả sử máy chủ ghi lại nhiệt độ liên tiếp theo từng giây: `10, 20, 30, 40, 10, 20, 25, ...`
* Trong bài báo và trên Web Demo, AI nhìn **1 cụm $L = W = 100$ điểm liên tiếp** (bước trượt $\text{Step} = 10$ hoặc $1$) để hiểu ngữ cảnh xung quanh.
* Để dễ nhẩm bằng mắt, ta giả sử cửa sổ chỉ dài **$W = 5$ con số**. Một cửa sổ gốc bình thường $\mathbf{Z}$ sẽ có dạng:
  $$\mathbf{Z} = [10, \ 20, \ 30, \ 40, \ 10]$$

---

## 4.2. Giải thích chi tiết 12 Lớp Phân Loại (1 Lớp Bình Thường + 11 Lớp Bất Thường Giả)

Vì dữ liệu huấn luyện không có sẵn nhãn "Lỗi", RedLamp lấy cửa sổ bình thường $\mathbf{Z} = [10, \ \mathbf{20, \ 30, \ 40}, \ 10]$ (ứng với lớp `0: normal` — Bình thường) ra và **tự tay làm biến dạng đoạn ở giữa** ($\mathbf{[20, 30, 40]}$, từ vị trí `st` đến `ed`) bằng **11 công thức toán học khác nhau** (`RedLamp/loaders/loader_aug.py`) để tạo ra tổng cộng **12 lớp phân loại ($K = 12$)**:

* **Lớp `0. normal` ($k=1$ trong bài báo) — Bình thường:**
  * **Công thức:** $X_{\text{dim}}^{(t,1)} = X_{\text{dim}}^{(t)}$
  * **Ví dụ 5 con số:** Giữ nguyên 100% không sửa gì cả: $\mathbf{Z} = [10, 20, 30, 40, 10]$.

### Nhóm A: Biến đổi Điểm & Nhiễu (Tác động đột ngột hoặc làm bẩn tín hiệu)
1. **Lớp `1. spike` ($k=2$) — Đỉnh nhọn (*Spike / Point Outlier*):**
   * **Công thức:** $X_{\text{dim},\text{st}}^{(t,k)} = X_{\text{dim},\text{st}}^{(t)} + a$ (biến động tăng/giảm biên độ đột biến cục bộ tại 1–2 điểm thời gian).
   * **Ví dụ 5 con số:** Chỉ sửa đúng con số ở chính giữa (`30` $\rightarrow$ `95`):
     $$\mathbf{Y}_{\text{spike}} = [10, \ 20, \ \mathbf{95}, \ 40, \ 10]$$
   * **Sự cố ngoài đời thực:** Xung điện áp tăng vọt tức thời, tia bức xạ bắn trúng cảm biến vệ tinh NASA, hoặc mạng bị dội ping (*ping spike*) trong 1 giây rồi hết ngay.

2. **Lớp `4. noise` ($k=5$) — Nhiễu ngẫu nhiên (*Noise*):**
   * **Công thức:** $X_{\text{dim},\text{st}:\text{ed}}^{(t,k)} + a, \; a \sim \mathcal{N}(0, 0.1I)$ (biên độ bị xáo trộn bởi nhiễu trắng Gaussian làm đường sóng bị "răng cưa" lăn tăn).
   * **Ví dụ 5 con số:** Cộng/trừ ngẫu nhiên vào đoạn `[20, 30, 40]`:
     $$\mathbf{Y}_{\text{noise}} = [10, \ \mathbf{23, \ 27, \ 44}, \ 10]$$
   * **Sự cố ngoài đời thực:** Dây cáp cảm biến bị lỏng mối nối hoặc bị nhiễu điện từ trường.

### Nhóm B: Biến đổi Hình học & Chiều hướng (Cắt cụt hoặc lật ngược sóng)
3. **Lớp `5. cutoff` ($k=6$) — Cắt cụt biên độ (*Cutoff / Flatline*):**
   * **Công thức:** Thay cả đoạn bằng 1 hằng số cố định $c \in [\min, \max]$ (đồ thị biến thành vạch ngang phẳng lì).
   * **Ví dụ 5 con số:** Ép cả 3 số giữa đứng im ở mức `15`:
     $$\mathbf{Y}_{\text{cutoff}} = [10, \ \mathbf{15, \ 15, \ 15}, \ 10]$$
   * **Sự cố ngoài đời thực:** Cảm biến bị "đơ" phần cứng (treo giá trị / bão hòa), hoặc dịch vụ web bị sập nên lưu lượng rơi thẳng về một mức cố định.

4. **Lớp `2. flip` ($k=3$) — Đảo cực / Đảo chiều (*Flip*):**
   * **Công thức:** $X_{\text{dim},\text{st}:\text{ed}}^{(t,k)} = X_{\text{dim},\text{st}:\text{ed}}^{(t)}[::-1]$ (lật ngược thứ tự đoạn tín hiệu theo chiều thời gian; đang từ xu hướng tăng sẽ biến thành xu hướng giảm).
   * **Ví dụ 5 con số:** Lật ngược `[20, 30, 40]` thành `[40, 30, 20]`:
     $$\mathbf{Y}_{\text{flip}} = [10, \ \mathbf{40, \ 30, \ 20}, \ 10]$$
   * **Sự cố ngoài đời thực:** Quy trình đóng/mở van công nghiệp hoặc chu trình nạp/xả bị đảo ngược thứ tự vận hành.

5. **Lớp `10. upsidedown` ($k=11$) — Lộn ngược chuỗi (*Upside-down*):**
   * **Công thức:** $2 \times \text{Mean} - X_{\text{dim},\text{st}:\text{ed}}^{(t)}$ (lấy giá trị trung bình $\mu$ của đoạn tín hiệu làm trục đối xứng, **lật úp đỉnh sóng xuống thành đáy sóng** theo chiều dọc).
   * **Ví dụ 5 con số:** Giả sử đoạn giữa là một đỉnh nhô lên `[20, 50, 20]` (trung bình $\mu = 30$). Khi lật úp qua mức $30$, đỉnh `50` biến thành đáy `10`:
     $$\text{Gốc: } [10, \ \mathbf{20, \ 50, \ 20}, \ 10] \quad \longrightarrow \quad \mathbf{Y}_{\text{upsidedown}} = [10, \ \mathbf{40, \ 10, \ 40}, \ 10]$$
   * **Sự cố ngoài đời thực:** Đấu ngược cực âm/dương của cảm biến, hoặc lưu lượng hệ thống đột ngột tụt đáy đúng vào khung giờ cao điểm lẽ ra phải đạt đỉnh.

### Nhóm C: Biến đổi Tần số & Độ mượt (Co giãn nhịp độ hoặc làm phẳng sóng)
6. **Lớp `3. speedup` ($k=4$) — Tăng tốc tần số (*Speedup: $2\times$ hoặc $0.5\times$*):**
   * **Công thức:** Lấy mẫu cách quãng `[::2]` ($2\times$) hoặc nội suy giãn chậm lại ($0.5\times$).
   * **Ví dụ 5 con số:** Nhồi nhét 2 nhịp lên-xuống vào trong cùng 3 giây:
     $$\mathbf{Y}_{\text{speedup}} = [10, \ \mathbf{40, \ 10, \ 40}, \ 10]$$
   * **Sự cố ngoài đời thực:** Động cơ quay quá tốc độ định mức, hoặc tác vụ ngầm (*cronjob*) bị lỗi chạy lặp lại với tần suất nhanh gấp đôi bình thường.

7. **Lớp `6. average` ($k=7$) — Trung bình hóa (*Average / Moving Average Smoothing*):**
   * **Công thức:** $\text{MovingAverage}(X_{\text{dim},\text{st}:\text{ed}}^{(t)}, w=L/5=20)$ (tín hiệu bị làm phẳng bằng trung bình trượt, mất đi dao động chi tiết).
   * **Ví dụ 5 con số:** Đoạn sóng đang nhấp nhô mạnh `[15, 45, 15]` bị san phẳng về quanh mức trung bình `25`:
     $$\text{Gốc: } [10, \ \mathbf{15, \ 45, \ 15}, \ 10] \quad \longrightarrow \quad \mathbf{Y}_{\text{average}} = [10, \ \mathbf{23, \ 25, \ 23}, \ 10]$$
   * **Sự cố ngoài đời thực & Nguồn gốc sinh ra `False Anomaly`:** Cảm biến bị lì độ nhạy không bắt kịp các dao động nhanh. **Đặc biệt lưu ý:** Nếu chuỗi gốc vốn đã phẳng sẵn (ví dụ `[20, 21, 20, 20, 21]`), thì sau khi lấy trung bình trượt `average`, kết quả tạo ra là `[20, 20.3, 20.3, 20.3, 21]` — **nhìn chẳng khác gì tín hiệu bình thường cả!**

### Nhóm D: Biến đổi Biên độ, Nền & Ngữ cảnh (Kéo giãn, trôi dạt hoặc hỗn hợp)
8. **Lớp `7. scale` ($k=8$) — Co giãn biên độ (*Scale*):**
   * **Công thức:** $a \times X_{\text{dim},\text{st}:\text{ed}}^{(t)}, \; a \sim \mathcal{N}(1, 1)$ (nhân biên độ đoạn tín hiệu với hệ số phóng đại hoặc thu nhỏ $a$).
   * **Ví dụ 5 con số:** Nhân gấp đôi ($a = 2$) đoạn `[20, 30, 40]`:
     $$\mathbf{Y}_{\text{scale}} = [10, \ \mathbf{40, \ 60, \ 80}, \ 10]$$
   * **Sự cố ngoài đời thực:** Hệ số khuếch đại (*gain*) của thiết bị đo bị sai lệch, hoặc tải máy chủ tăng gấp đôi đồng loạt.

9. **Lớp `8. wander` ($k=9$) — Trôi dạt xu hướng (*Wander / Baseline Drift*):**
   * **Công thức:** Cộng dốc tuyến tính `linspace(0, a)` vào đoạn giữa rồi giữ nguyên độ lệch $a$ đó đến tận cuối cửa sổ (mặt nạ đánh dấu lỗi kéo dài từ `st` đến hết cửa sổ).
   * **Ví dụ 5 con số:** Cộng thêm đường dốc `[+10, +20, +30]` và giữ độ lệch `+30` ở điểm cuối:
     $$\text{Gốc: } [10, \ 20, \ 30, \ 40, \ 10] \quad \longrightarrow \quad \mathbf{Y}_{\text{wander}} = [10, \ \mathbf{30, \ 50, \ 70, \ 40}]$$
   * **Sự cố ngoài đời thực:** Hiện tượng **rò rỉ bộ nhớ (*Memory Leak*)** làm mức sử dụng RAM cứ dâng cao dần lên theo thời gian mà không hạ xuống.

10. **Lớp `9. contextual` ($k=10$) — Bất thường ngữ cảnh (*Contextual: $a \cdot y + b$*):**
    * **Công thức:** $a \times X_{\text{dim},\text{st}:\text{ed}}^{(t)} + b$ (kết hợp đồng thời cả nhân biên độ $a$ lẫn cộng độ lệch nền $b$; giá trị nằm trong khoảng hợp lệ nhưng không khớp với nhịp điệu xung quanh).
    * **Ví dụ 5 con số:** Nhân $a = 0.5$ rồi cộng $b = 25$ vào `[20, 30, 40]`:
      $$\mathbf{Y}_{\text{contextual}} = [10, \ \mathbf{35, \ 40, \ 45}, \ 10]$$
    * **Sự cố ngoài đời thực:** CPU chạy ở mức `45%` là bình thường vào 12 giờ trưa, nhưng nếu xuất hiện vào lúc 3 giờ sáng (khi hệ thống đáng lẽ phải nghỉ ở mức `10%`) thì đó là **bất thường ngữ cảnh**.

11. **Lớp `11. mixture` ($k=12$) — Bất thường hỗn hợp / Tráo đoạn (*Mixture / Segment Swap*):**
    * **Công thức:** $X_{\text{dim},\text{st}:\text{ed}}^{(t,k)} = X_{\text{dim},\text{st}:\text{ed}}^{(t')}$ (cắt một đoạn tín hiệu ở thời điểm $t'$ hoàn toàn khác ghép đè vào vị trí hiện tại $t$).
    * **Ví dụ 5 con số:** Đang ở chu kỳ tăng `[10, 20, 30, 40, 10]`, bỗng nhiên bị ghép một đoạn `[8, 9, 8]` từ lúc máy đang nghỉ vào giữa:
      $$\mathbf{Y}_{\text{mixture}} = [10, \ \mathbf{8, \ 9, \ 8}, \ 10]$$
    * **Sự cố ngoài đời thực:** Lỗi đồng bộ thời gian (*clock desynchronization*) hoặc mất gói tin làm hai đoạn dữ liệu lệch pha bị ghép nối tiếp vào nhau.

---

# PHẦN 5: GIẢI PHẪU TỪNG CÔNG THỨC TOÁN HỌC TỪ (1) ĐẾN (8) & KIẾN TRÚC MẠNG `ConvAEC`

## 5.1. Cửa sổ trượt (Sliding Window — Mục 3.1)
* **Công thức trong bài báo:**
  $$X^{(t)} = \{x_{t-L+1}, \dots, x_t\} \in \mathbb{R}^{D \times L}$$
* **Hiểu đơn giản:** Chuỗi thời gian ban đầu rất dài (ví dụ $T = 25,000$ điểm). Mạng AI dùng một khung cửa sổ dài $L = 100$ điểm liên tiếp. Với mỗi cửa sổ $X^{(t)}$, AI chấm ra một con số điểm bất thường $s(t) \ge 0$. Con số $s(t)$ càng lớn thì khả năng tại thời điểm $t$ đang có sự cố càng cao.

---

## 5.2. Công thức (1) — Nhãn Mềm Hiệu Chỉnh Ngược (Backward Correction)

* **Công thức trong bài báo:**
  $$\tilde{y}_i^{(t,k)} = \begin{cases} (1 - \alpha - K\beta)y_i^{(t,k)} + \alpha + \beta, & \text{nếu } i = 1 \text{ (vị trí lớp Normal)}, \\ (1 - \alpha - K\beta)y_i^{(t,k)} + \beta, & \text{nếu } i \in \{2, \dots, 12\} \text{ (vị trí 11 lớp Lỗi)}, \end{cases} \quad (1)$$

* **Giải nghĩa từng chữ cái:**
  * $K = 12$: Tổng số lớp (Lớp thứ 1 là `Normal`, Lớp thứ 2 đến 12 là 11 loại lỗi giả).
  * $y^{(t,k)}$: **Nhãn cứng (One-hot)** ban đầu — chỉ có đúng vị trí lớp $k$ bằng `1`, còn lại 11 vị trí kia bằng `0`.
  * $\tilde{y}^{(t,k)}$: **Nhãn mềm (Soft label)** sau khi đã được làm dịu đi.
  * $\alpha = 0.1$ ($10\%$): Xác suất dự phòng rằng **"Một mẫu mang tiếng là lỗi giả thực chất nhìn vẫn giống Bình thường (False Anomaly)"**. Vì thế ta luôn cộng bù $\alpha = 0.1$ vào vị trí lớp `Normal`!
  * $\beta = 0.01$ ($1\%$): Xác suất dự phòng rằng **"Một lớp bất kỳ có thể bị nhầm lẫn sang một trong 12 lớp"**.
  * Cụm $(1 - \alpha - K\beta)$: Phần độ tin cậy còn lại dành cho lớp chính: $1 - 0.1 - 12 \times 0.01 = \mathbf{0.78}$.

* **🔢 Ví dụ tính tay cụ thể:**
  * **Trường hợp A — Mẫu đang xét là Lớp `Spike` ($y_{\text{spike}} = 1$, các vị trí khác $y_i = 0$):**
    * Tại ô `Normal` ($y_1 = 0$): $\tilde{y}_1 = 0.78 \times 0 + 0.10 + 0.01 = \mathbf{0.11}$ ($11\%$).
    * Tại ô `Spike` ($y_2 = 1$): $\tilde{y}_2 = 0.78 \times 1 + 0.01 = \mathbf{0.79}$ ($79\%$).
    * Tại 10 ô lỗi còn lại ($y_i = 0$): $\tilde{y}_i = 0.78 \times 0 + 0.01 = \mathbf{0.01}$ ($1\%$).
    * $\implies$ **Nhãn mềm thu được:** $\tilde{y} = [\mathbf{0.11}, \; \mathbf{0.79}, \; 0.01, \; 0.01, \dots, 0.01]$ *(Tổng 12 ô $= 0.11 + 0.79 + 10 \times 0.01 = 1.00$)*.
  * **Trường hợp B — Mẫu đang xét là Lớp `Normal` ($y_1 = 1$):**
    * Tại ô `Normal` ($y_1 = 1$): $\tilde{y}_1 = 0.78 \times 1 + 0.10 + 0.01 = \mathbf{0.89}$ ($89\%$).
    * Tại 11 ô lỗi còn lại ($y_i = 0$): $\tilde{y}_i = \mathbf{0.01}$ ($1\%$ mỗi lớp) — chừa lại $11\%$ khả năng mẫu gốc bị lẫn **lỗi nhiễm bẩn (`Anomaly Contamination`)**!

---

## 5.3. Kiến Trúc Mạng Học Sâu `ConvAEC` (`Encoder` $\rightarrow$ `Decoder` + `NonLinear Classifier`)

Kiến trúc mô hình trong `RedLamp/models/meta.py` mang tên **`ConvAEC (Convolutional Autoencoder + NonLinear Classifier)`**, xử lý cửa sổ đầu vào qua **3 khối** (tương ứng trực tiếp với **Bước 2, Bước 3, Bước 4 ở Tab 2 của Web Demo**):

1. **Khối 1 — `ConvEncoder` (Trích xuất đặc trưng nén thành véc-tơ `Embedding 128 chiều`):**
   * Cửa sổ đầu vào kích thước `[1, 100, n_features]` đi qua **4 tầng tích chập 1D (`ConvBlock 0..3`, số bộ lọc `[128, 128, 256, 256]`, `stride = 2`)** giảm dần chiều thời gian (`100` $\rightarrow$ `51` $\rightarrow$ `26` $\rightarrow$ `14` $\rightarrow$ `8`), sau đó qua `MaxPool1D` (về độ dài `1`, kích thước `[1, 256, 1]`) và `Conv1x1` để tạo ra **véc-tơ `Embedding` 128 chiều (`[1, 128, 1]`)**.
2. **Khối 2 — `ConvDecoder` (Giải mã tái tạo chuỗi & Tính điểm `MSE`):**
   * Từ véc-tơ `Embedding 128 chiều`, mạng giải mã tích chập chuyển vị (`ConvTranspose1d`) cố gắng vẽ lại cửa sổ bình thường $\hat{X}$.
3. **Khối 3 — `NonLinear Classifier` (Phân loại xác suất 12 lớp & Tính điểm `CE`):**
   * Từ véc-tơ `Embedding 128 chiều`, mạng MLP phi tuyến 2 tầng (`Linear(128 -> 32)` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Linear(32 -> 12)` $\rightarrow$ `Softmax`) dự đoán xác suất rơi vào **12 lớp** ($\hat{y}$).

---

## 5.4. Công thức (2), (3), (4) — Hàm Mất Mát Khi Huấn Luyện (Objective Function)

### 1) Công thức (2) — Hàm mất mát tái tạo có mặt nạ ($\mathcal{L}_{\text{MSE}}$):
$$\mathcal{L}_{\text{MSE}} = \left\| \left(1 - W^{(t,k)}\right) \odot \left(X^{(t,k)} - \hat{X}^{(t,k)}\right) \right\|_2^2 \quad (2)$$
* Trong bài báo, mặt nạ $W$ bằng `1` ở vùng vừa bị chèn lỗi giả và bằng `0` ở vùng bình thường $\implies (1 - W)$ sẽ bằng **`0` ở vùng lỗi giả** và bằng **`1` ở vùng bình thường** *(Lưu ý: Trong code `loader_aug.py`, biến `anomaly_mask` lưu trực tiếp giá trị `1 - W`, tức bằng `0` ở vùng lỗi và `1` ở vùng bình thường)*.
* **🔢 Ví dụ tính tay 4 con số:**
  * Sóng bình thường gốc: $[10, \; 10, \; 10, \; 10]$.
  * Chèn lỗi `Spike` vào vị trí thứ 3 $\implies X^{(t,k)} = [10, \; 10, \; \mathbf{50}, \; 10]$, và $(1 - W) = [1, \; 1, \; \mathbf{0}, \; 1]$.
  * Giả sử Decoder vẽ ra đường sóng bình thường: $\hat{X}^{(t,k)} = [10, \; 9, \; \mathbf{10}, \; 10]$.
  * Hiệu số $(X - \hat{X}) = [0, \; 1, \; \mathbf{40}, \; 0]$.
  * Nhân với $(1 - W)$:
    $$(1 - W) \odot (X - \hat{X}) = [1\times 0, \; 1\times 1, \; \mathbf{0\times 40}, \; 1\times 0] = [0, \; 1, \; \mathbf{0}, \; 0]$$
  * Bình phương và cộng lại: $\mathcal{L}_{\text{MSE}} = 0^2 + 1^2 + 0^2 + 0^2 = 1$.
  * **Ý nghĩa:** Nhờ nhân với `0` ở vị trí có lỗi giả, **Decoder KHÔNG bị phạt khi nó từ chối vẽ cái gai nhọn 50**! Nhờ đó Decoder chỉ học vẽ phần bình thường.

### 2) Công thức (3) — Hàm mất mát phân loại nhãn mềm ($\mathcal{L}_{\text{CE}}$):
$$\mathcal{L}_{\text{CE}} = -\sum_{i=1}^{K} \tilde{y}_i^{(t,k)} \log\left(\hat{y}_i^{(t,k)}\right) \quad (3)$$

### 3) Công thức (4) — Tổng hợp hàm mất mát ($\mathcal{L}$):
$$\mathcal{L} = \gamma \mathcal{L}_{\text{CE}} + (1 - \gamma) \mathcal{L}_{\text{MSE}} \quad (4)$$
* Với $\gamma = 0.1$, tổng mức phạt khi huấn luyện bằng **$10\%$ điểm phạt phân loại ($\mathcal{L}_{\text{CE}}$) + $90\%$ điểm phạt vẽ lại sóng ($\mathcal{L}_{\text{MSE}}$)**.

---

## 5.5. Công thức (5), (6), (7), (8) — Chấm Điểm Bất Thường Lúc Kiểm Thử & Bộ Lọc FAA

Khi chạy mô hình trên tập kiểm thử $X^{\text{test}}$:

1. **Bước 1 — Điểm sai số tái tạo thô (`raw_mse`) — Công thức (5):**
   $$s_{\text{MSE}}(t) = \left\| X^{(t)} - \hat{X}^{(t)} \right\|_2^2 \quad (5)$$
2. **Bước 2 — Lọc bỏ "Lớp giả bình thường" bằng bộ lọc FAA & Tính điểm phân loại thô (`raw_ce`) — Công thức (6) & (7):**
   $$\text{FAA}\left(\hat{y}_i^{(t)}\right) = \begin{cases} 0, & \text{nếu } \frac{1}{T - L + 1}\sum_{j=L}^{T} \hat{y}_i^{(j)} > \delta, \\ \hat{y}_i^{(t)}, & \text{ngược lại}, \end{cases} \quad (6)$$
   $$s_{\text{CE}}(t) = \sum_{i=2}^{K} \text{FAA}\left(\hat{y}_i^{(t)}\right) \quad (7)$$
   * **Ý nghĩa:** Lớp `normal` ($i=1$) luôn bị loại khỏi tổng điểm bất thường. Ngoài ra, nếu bất kỳ lớp lỗi giả nào có **xác suất trung bình trên toàn chuỗi kiểm thử $> \delta = 0.05$ ($5\%$)** (xuất hiện tràn lan cả lúc máy chạy bình thường — *False Anomaly*), bộ lọc **FAA** tự động đưa lớp đó vào danh sách **lớp bị loại trừ (`excluded_classes`)** và gạch điểm của lớp đó về `0`!
3. **Bước 3 — Điểm kết hợp cuối cùng — Công thức (8) & Ngưỡng quyết định `P99.5` trên Web Demo:**
   * **Điểm chuẩn hóa toàn chuỗi (Công thức 8 trong bài báo):**
     $$s(t) = \frac{1}{2}\text{MinMax}_T\left(s_{\text{MSE}}(t)\right) + \frac{1}{2}\text{MinMax}_T\left(s_{\text{CE}}(t)\right) \quad (8)$$
     *(Dùng để vẽ đường màu đỏ tổng quan ở Tab 1 và tính `VUS-ROC`, `VUS-PR`, `RF`).*
   * **Điểm kết hợp thô (`raw_combined`) & Ngưỡng `P99.5` (ở Tab 2 và Tab 3 của Web Demo):**
     $$\text{Điểm kết hợp thô} = \frac{\text{Điểm tái tạo thô (MSE)} + \text{Điểm phân loại thô (CE)}}{2}$$
     Vì chuẩn hóa Min-Max khiến mọi chuỗi đều có đỉnh bằng `1.0`, nên để ra quyết định **`BẤT THƯỜNG`** hay **`BÌNH THƯỜNG`** cho một cửa sổ/điểm cụ thể trên thực tế, Web Demo đối chiếu trực tiếp **Điểm kết hợp thô (`raw_combined`)** với **Ngưỡng phân vị thứ `99,5%` (`P99.5`)** của phân phối điểm trên chuỗi!

---

## 5.6. Kết Quả Thực Nghiệm Nổi Bật (Q1–Q5) & Bảng Siêu Tham Số Cố Định (Table 4)

### Tóm tắt kết quả thực nghiệm trong bài báo (Mục 4.2 – 4.6):
1. **Đứng đầu trên 5 bộ dữ liệu (Q1):** Đạt trung bình **`VUS-ROC = 0.803`**, **`VUS-PR = 0.430`** (vượt phương pháp đứng thứ hai là *CutAddPaste* tới **+17.4%**).
2. **Chứng minh vai trò của từng thành phần (Q2 — Ablation Study):** Nếu tắt bất kỳ thành phần nào (`w/o AM`, `w/o BC`, `w/o FAA`, hoặc gộp 11 lớp lỗi về phân loại nhị phân `Binary`), điểm `VUS-PR` lập tức tụt giảm mạnh (giảm tới **24.5%** trên SMD khi quay về phân loại nhị phân).
3. **Sức đề kháng trước dữ liệu bẩn (Q4 — Robustness):** Khi trộn thêm **5% lỗi vào tập Train** của UCR, REDLAMP chỉ giảm nhẹ **1.8%** điểm `VUS-PR`, trong khi *CutAddPaste* sụt giảm tới **15.5%**.
4. **Khả năng giải thích nguyên nhân lỗi (Q5 — Explainability):** Các điểm lỗi thật trên tập Test nằm sát cụm lớp lỗi giả nào sẽ chỉ ra ngay hình thái vật lý của sự cố đó (ví dụ: lỗi nhịp sinh học côn trùng EPG nằm sát cụm `Speedup`; lỗi nhiễu sóng huyết áp ABP nằm sát cụm `Noise`; lỗi máy chủ SMD nằm gần `Spike` và `Cutoff`).

### Bảng tổng hợp siêu tham số cố định của mô hình (Table 4):
| Tên tham số | Ký hiệu | Giá trị mặc định | Ý nghĩa dễ hiểu |
| :--- | :---: | :---: | :--- |
| Kích thước cửa sổ trượt | $L$ ($W$) | `100` | Mỗi lần AI nhìn một đoạn gồm 100 điểm thời gian liên tiếp. |
| Số chiều không gian ẩn (Embedding) | `emb_dim` | `128` | Đoạn sóng 100 điểm được Encoder tóm tắt thành đúng 128 con số. |
| Tổng số lớp phân loại | $K$ | `12` | Gồm 1 lớp `Normal` + 11 lớp bất thường giả. |
| Xác suất mẫu lỗi giả thực ra là bình thường | $\alpha$ | `0.10` ($10\%$) | Dùng trong Công thức (1) để làm mềm nhãn về phía lớp Normal. |
| Xác suất nhầm lẫn giữa các lớp | $\beta$ | `0.01` ($1\%$) | Dùng trong Công thức (1) để chia đều xác suất dự phòng cho mọi lớp. |
| Trọng số hàm mất mát phân loại | $\gamma$ | `0.10` | Dùng trong Công thức (4): $\mathcal{L} = 0.1\mathcal{L}_{\text{CE}} + 0.9\mathcal{L}_{\text{MSE}}$. |
| Ngưỡng lọc lớp giả bình thường FAA | $\delta$ | `0.05` ($5\%$) | Dùng trong Công thức (6): Lớp lỗi nào xuất hiện $> 5\%$ toàn thời gian Test sẽ bị gạch điểm về 0. |

---

# PHẦN 6: HƯỚNG DẪN CHI TIẾT 3 TAB TRONG GIAO DIỆN WEB DEMO (`REDLAMP DEMO`) & KỊCH BẢN THUYẾT TRÌNH 5 PHÚT

Khi khởi chạy `python demo_redlamp/server.py` và mở địa chỉ `http://127.0.0.1:8000`, toàn bộ giao diện được thiết kế **cố định vừa khít khung nhìn màn hình máy tính (`100vh` Zero-Scroll)** với cỡ chữ lớn phục vụ trình chiếu máy chiếu, chia thành **3 Tab theo đúng trình tự báo cáo**:

---

## 6.1. Tab 1: `1. Chọn Pretrain & Mô hình` (Chọn dữ liệu, xem cấu hình, chỉ số đánh giá & tổng quan chuỗi)

Màn hình Tab 1 gồm **3 khối** hiển thị đồng thời trên một khung nhìn duy nhất:

1. **Khối trên — `Bộ chọn Dataset & Entity`:**
   * **`1. Tập dữ liệu (Dataset)`:** Cho phép chọn 1 trong **5 bộ dữ liệu chuẩn của bài báo KDD '25**: `SMD (38 dims)`, `AIOps / IOPS (1 dim)`, `MSL (55 dims)`, `SMAP (25 dims)`, `Anomaly Archive (UCR 250 entities - 1 dim)`.
   * **`Tìm kiếm Entity`:** Ô nhập từ khóa lọc nhanh thực thể (ví dụ gõ `001`, `machine-1-1`, `C-1`, `A-1`...).
   * **`2. Thực thể (Entity)`:** Hộp chọn thực thể cụ thể trong tập dữ liệu đang xét.
   * **`3. Seed huấn luyện (0 - 4)`:** Chuyển đổi giữa 5 hạt giống huấn luyện độc lập (`Seed 0 (Mặc định)` $\rightarrow$ `Seed 4`) đối với 4 tập đa seed (`SMD`, `AIOps`, `MSL`, `SMAP`). Khi đổi bất kỳ thông số nào, hệ thống hiển thị **hiệu ứng chờ (Loading Spinner)** và tự động nạp dữ liệu từ ổ cứng (hoặc tải On-Demand từ Kaggle nếu chưa có).
2. **Khối giữa (Chia 2 cột):**
   * **Cột trái — `Thông số cấu hình mô hình`:** Hiển thị đầy đủ 8 thông số kỹ thuật thật của checkpoint đang nạp:
     * `Mô hình`: `ConvAEC (Convolutional Autoencoder + NonLinear Classifier)`
     * `Số lớp phân loại`: `12 (1 normal + 11 synthetic anomalies)`
     * `Số chiều (n_features)`: `38` / `1` / `55` / `25`
     * `Kích thước cửa sổ (W)`: `100`
     * `Bước trượt (Step)`: `10` (hoặc `1`)
     * `Chiều Embedding`: `128`
     * `Đường dẫn model`: Đường dẫn tới tệp `bestmodel.pkl`
     * `Trạng thái nạp`: Huy hiệu xanh `Model đã nạp trong bộ nhớ`
   * **Cột phải — `Kết quả đánh giá chất lượng (Metrics)`:** Hiển thị 3 ô chỉ số lớn **`VUS-ROC`**, **`VUS-PR`**, **`RF (Range F-score)`** của đúng Seed đang chọn, kèm dòng chữ bên dưới ghi rõ **`Trung bình 5 seed: mean ± std`** để hội đồng đối chiếu độ ổn định qua 5 lần chạy.
3. **Khối dưới — `Tổng quan toàn bộ chuỗi thời gian của Entity`:**
   * Vẽ toàn bộ chuỗi thời gian kiểm thử của thực thể: **Đường màu xám đậm** là giá trị chuỗi đo lường (Chiều 1); **Đường màu đỏ** là điểm bất thường tương đối (`anomaly_score`), kèm nút **`Tải PNG`**.

---

## 6.2. Tab 2: `2. Trình bày Giải thích` (Walkthrough 5 bước giải thích mô hình trong ~5 phút)

Ngay sau khi chọn mô hình ở Tab 1, người thuyết trình bấm sang **Tab 2** để dẫn dắt hội đồng đi qua **5 bước xử lý bên trong mạng `ConvAEC`** (có thể bấm nút **`◀ Quay lại` / `Tiếp theo ▶`**, bấm trực tiếp vào 5 viên thuốc bước `1..5`, hoặc bấm phím mũi tên **`←` / `→`** trên bàn phím):

* **Thanh điều khiển trên cùng:**
  * **5 nút bước kèm thời lượng gợi ý:** `1. Đầu vào (~0,5p)` $\rightarrow$ `2. Encoder (~1p)` $\rightarrow$ `3. Decoder (~1p)` $\rightarrow$ `4. Classifier (~1p)` $\rightarrow$ `5. Điểm kết hợp (~1,5p)`.
  * **Khung `CHẾ ĐỘ GIẢI THÍCH`:** Cho biết đang giải thích trên `Chuỗi dữ liệu mẫu của Entity` (`Mẫu kiểm thử`) hay đang giải thích cho `Điểm nạp thủ công (t=99)` được đồng bộ từ Tab 3 sang.
  * **Thanh trượt `Vị trí cửa sổ (t)` & 2 nút nhảy nhanh:** Nút **`Xem cửa sổ chuẩn`** (nhảy tới cửa sổ bình thường đối chứng) và **`Xem cửa sổ đỉnh lỗi`** (nhảy ngay tới vị trí có điểm bất thường cao nhất chuỗi).

* **Nội dung chi tiết của 5 Bước (5 Slides):**
  1. **`Bước 1: Chuỗi thời gian & Tô sáng cửa sổ 100 điểm đang xét` (~0,5 phút):**
     * Vẽ lát cắt chuỗi dữ liệu xung quanh vị trí $t$ và **tô vùng khung màu đỏ hồng nét đứt** đánh dấu chính xác cửa sổ 100 điểm $[t-99, t]$ đang được cắt ra để đưa vào mạng `ConvAEC`.
  2. **`Bước 2: Mạng trích xuất đặc trưng (Encoder -> Embedding)` (~1 phút):**
     * **Cột trái:** Sơ đồ cấu trúc mạng tích chập & kích thước tensor thật qua từng tầng (`Cửa sổ đầu vào [1, 100, D]` $\rightarrow$ `ConvBlock 0..3` $\rightarrow$ `MaxPool1D [1, 256, 1]` $\rightarrow$ `Conv1x1 -> Embedding [1, 128, 1]`), kèm mục mở rộng xem *Năng lượng kích hoạt trung bình (`Mean Activation Energy`)* trích xuất bằng Forward Hook thật từ PyTorch.
     * **Cột phải:** Biểu đồ cột trực quan hóa **véc-tơ đặc trưng nén `Embedding` (128 chiều)** (các giá trị dương màu xanh dương, giá trị âm màu đỏ).
  3. **`Bước 3: Mạng giải mã tái tạo (Decoder) & Sai số bình phương` (~1 phút):**
     * Đặt **2 biểu đồ song song cùng thang đo**:
       * **Bên trái (`Cửa sổ đang xét`):** Đường tái tạo màu xanh dương không bám theo kịp đoạn tín hiệu bất thường màu xám đậm $\rightarrow$ Huy hiệu báo **`MSE` cao**.
       * **Bên phải (`Cửa sổ bình thường đối chứng`):** Đường tái tạo màu xanh dương khớp gần như hoàn hảo với đường gốc $\rightarrow$ Huy hiệu báo **`MSE` rất nhỏ**.
  4. **`Bước 4: Bộ phân loại xác suất 12 loại bất thường giả định` (~1 phút):**
     * Đặt **2 biểu đồ cột 12 lớp song song**:
       * **Bên trái (`Xác suất 12 lớp tại cửa sổ khảo sát`):** Lớp bất thường hợp lệ có xác suất cao nhất được tô **màu đỏ nổi bật** kèm nhãn `%` trên đỉnh cột.
       * **Bên phải (`Xác suất 12 lớp tại cửa sổ bình thường`):** Hầu hết xác suất dồn vào các lớp bị loại trừ (màu xám nhạt).
     * Chú giải rõ: **Màu đỏ** là lớp có khả năng bất thường hợp lệ; **Màu xám nhạt** là lớp bị loại khỏi `ce_score` (Lớp `normal` hoặc lớp có tần suất trung bình toàn chuỗi $> 0,05$ theo bộ lọc FAA).
  5. **`Bước 5: Điểm kết hợp & Công thức tính điểm cuối cùng` (~1,5 phút):**
     * **Biểu đồ phía trên:** Vẽ đồng thời 3 đường theo thời gian: `Điểm tái tạo (MSE - xám đậm)`, `Điểm phân loại (CE - xanh dương)`, `Điểm thô kết hợp (đỏ)` và đường kẻ ngang nét đứt đỏ **`Ngưỡng P99.5`**.
     * **Hộp `CÔNG THỨC KẾT HỢP VỚI SỐ LIỆU THỰC TẾ` phía dưới:** Thay trực tiếp con số thật của cửa sổ đang xét vào công thức:
       $$\text{Điểm cuối} = (\text{Điểm tái tạo} + \text{Điểm phân loại}) / 2 = (\text{raw\_mse} + \text{raw\_ce}) / 2$$
       và đối chiếu với `Ngưỡng P99.5` để xuất ra kết luận **`BẤT THƯỜNG`** hoặc **`BÌNH THƯỜNG`**.
  * *Lưu ý:* Ở đáy mỗi bước đều có **hộp câu tóm tắt ngắn gọn ($\le 20$ từ)** giúp người thuyết trình đọc nhanh ý chính ngay trên slide.

---

## 6.3. Tab 3: `3. Nhập điểm & Dự đoán` (Thử nghiệm bơm điểm mới & Đồng bộ ngược lại Tab 2)

Sau khi hội đồng đã hiểu rõ 5 bước ở Tab 2, người thuyết trình chuyển sang **Tab 3** để thực hiện **Demo tương tác trực tiếp (Live Inference)**:

1. **Cột trái — `Nhập điểm dữ liệu mới & Cửa sổ ngữ cảnh`:**
   * **Chọn nguồn 99 điểm ngữ cảnh trước đó:**
     * `(a) 99 điểm cuối của chuỗi mẫu` *(Mặc định)*
     * `(b) Lặp lại giá trị trung vị của chuỗi`
     * `(c) Tự dán / tải lên lịch sử` *(Cho phép dán 99 dòng dữ liệu tùy ý)*
   * **Nhập `Giá trị điểm mới (n_features giá trị)`:**
     * Tự động hiển thị 1 ô nhập (với tập 1 chiều như `AIOps`, `Anomaly Archive`) hoặc lưới `D1..D38` / `D1..D55` (với `SMD`, `SMAP`, `MSL`). Có nút **`Điền mẫu gợi ý`** để điền nhanh giá trị trung vị bình thường, sau đó người dùng có thể sửa 1 hoặc vài chiều lên cao (ví dụ gõ `3.0` hoặc `5.0`) để giả lập đột biến.
   * **Thanh trượt `Ngưỡng quyết định điểm thô`:** Mặc định tại phân vị `99,5% (P99.5)`, có thể kéo điều chỉnh linh hoạt hoặc bấm `Mặc định (P99.5)`.
   * **Nút `⚡ Thực hiện dự đoán bất thường`:** Gọi mô hình PyTorch `ConvAEC` suy luận trực tiếp cho cửa sổ 100 điểm vừa ghép.
2. **Cột phải (Phần trên) — `Kết quả phân tích điểm`:**
   * **Hộp `KẾT LUẬN`:** Báo rõ `TRẠNG THÁI BÌNH THƯỜNG` (nền xanh) hoặc `PHÁT HIỆN BẤT THƯỜNG` (nền đỏ).
   * **Thẻ `PHÂN LOẠI 12 NHÓM BẤT THƯỜNG GIẢ (PSEUDO ANOMALY)`:** Chỉ rõ điểm vừa nhập giống với nhóm biến dạng giả lập nào nhất (ví dụ: *Đỉnh nhọn (Spike)*, *Trôi dạt xu hướng (Wander)*...), kèm xác suất `%`, mô tả tiếng Việt và lý do phân loại.
   * **Nút `🔍 Xem giải thích 5 phút cho điểm này (Đồng bộ)`:** Tính năng ăn điểm nhất của Tab 3! Ngay khi bấm nút này, giao diện **tự động chuyển về Tab 2 (`2. Trình bày Giải thích`)** và cập nhật toàn bộ 5 bước giải thích theo đúng điểm vừa nhập (vẽ dấu chấm tròn đỏ `Điểm nạp mới (t=99)` trên đồ thị Bước 1, Bước 3 và Bước 5).
   * **4 Thẻ điểm số thô (`MSE`, `CE`, `Điểm kết hợp thô`, `Ngưỡng P99.5`)**, danh sách **Top-3 nhãn bất thường giả định** và menu mở rộng **Xem các lớp bị loại khỏi điểm phân loại (Lớp bình thường & Tần suất > 0,05)**.
3. **Cột phải (Phần dưới) — `Biểu đồ cửa sổ 100 điểm & Chuỗi tái tạo của mô hình`:**
   * Hiển thị ngay lập tức đường cửa sổ đầu vào 100 điểm (màu xám đậm, với điểm mới ở tận cùng bên phải $t=100$) chồng với đường tái tạo của Autoencoder (màu xanh dương) và chỉ số `MSE`.

---

## 6.4. Gợi Ý Luồng Thuyết Trình Demo Trong 5 Phút (Theo Đúng Thứ Tự Tab 1 $\rightarrow$ Tab 2 $\rightarrow$ Tab 3)

1. **Phút 1 — Mở màn tại `Tab 1 (1. Chọn Pretrain & Mô hình)`:**
   * Giới thiệu 5 bộ dữ liệu chuẩn của bài báo (`SMD`, `AIOps`, `MSL`, `SMAP`, `Anomaly Archive`) và chọn thực thể mẫu `SMD / machine-1-1 / Seed 0`.
   * Chỉ vào bảng **Thông số cấu hình mô hình** (`ConvAEC (Convolutional Autoencoder + NonLinear Classifier)`, cửa sổ $W=100$, Embedding $128$, $12$ lớp phân loại) và bảng **3 chỉ số (`VUS-ROC`, `VUS-PR`, `RF`)** kèm trung bình $\pm$ độ lệch chuẩn của 5 Seed.
   * Chỉ xuống **Biểu đồ tổng quan chuỗi thời gian** ở đáy Tab 1 để thấy các vùng điểm bất thường màu đỏ vọt lên rõ rệt.
2. **Phút 2 đến Phút 3,5 — Giải thích cơ chế hoạt động tại `Tab 2 (2. Trình bày Giải thích)`:**
   * Bấm sang **Tab 2**, lần lượt bấm nút **`Tiếp theo ▶`** (hoặc phím mũi tên `→`) đi qua 5 bước:
     * **Bước 1:** Chỉ vào vùng khung đỏ nét đứt cắt cửa sổ 100 điểm tại đỉnh lỗi.
     * **Bước 2:** Chỉ vào sơ đồ 4 tầng `ConvBlock` nén cửa sổ thành véc-tơ `Embedding 128 chiều`.
     * **Bước 3:** Chỉ vào sự tương phản giữa `MSE` rất cao ở cửa sổ khảo sát (bên trái) và `MSE` cực nhỏ ở cửa sổ bình thường đối chứng (bên phải).
     * **Bước 4:** Chỉ vào cột màu đỏ cao nhất trong 12 lớp ở cửa sổ khảo sát và giải thích các cột màu xám nhạt bị loại khỏi `ce_score` (lớp `normal` và các lớp có tần suất $> 0,05$).
     * **Bước 5:** Đọc hộp **Công thức kết hợp với số liệu thực tế** $(\text{MSE} + \text{CE}) / 2$ vượt ngưỡng `P99.5` $\Rightarrow$ Kết luận **BẤT THƯỜNG**.
3. **Phút 3,5 đến Phút 5 — Tương tác trực tiếp tại `Tab 3 (3. Nhập điểm & Dự đoán)` & Đồng bộ về `Tab 2`:**
   * Bấm sang **Tab 3**: Đầu tiên giữ nguyên giá trị trung vị mặc định và bấm **`⚡ Thực hiện dự đoán bất thường`** $\rightarrow$ Hệ thống báo **`TRẠNG THÁI BÌNH THƯỜNG`**.
   * Tiếp theo, nhập một giá trị đột biến (ví dụ sửa `D1` lên `3.0` hoặc `5.0`) rồi bấm lại **`⚡ Thực hiện dự đoán bất thường`** $\rightarrow$ Hệ thống lập tức đổi sang **`PHÁT HIỆN BẤT THƯỜNG`**, chỉ rõ nhóm bất thường giả (ví dụ *Đỉnh nhọn - Spike*) và vẽ đường gốc vọt cao tách rời khỏi đường tái tạo ở biểu đồ góc dưới bên phải.
   * Cuối cùng, bấm nút **`🔍 Xem giải thích 5 phút cho điểm này (Đồng bộ)`** để tự động nhảy về **Tab 2** và chiếu cho hội đồng thấy điểm vừa nhập (dấu chấm đỏ tại $t=99$) biến đổi qua cả 5 bước của mô hình như thế nào!
