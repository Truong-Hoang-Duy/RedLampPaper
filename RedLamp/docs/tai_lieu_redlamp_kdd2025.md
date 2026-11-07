# 📘 CẨM NANG GIẢI MÃ BÀI BÁO REDLAMP (KDD '25) — DÀNH CHO NGƯỜI MỚI BẮT ĐẦU MACHINE LEARNING

**Tên bài báo:** *Robust and Explainable Detector of Time Series Anomaly via Augmenting Multiclass Pseudo-Anomalies (REDLAMP)*  
**Tác giả:** Kohei Obata, Yasuko Matsubara, Yasushi Sakurai (Đại học Osaka, Nhật Bản) — **Hội nghị KDD '25**  

> **🎯 Mục tiêu của tài liệu này:**  
> Tài liệu này được biên soạn lại theo phong cách **"cầm tay chỉ việc"**, dành cho người chỉ mới biết sơ qua về Machine Learning. Mọi **từ viết tắt**, **ký hiệu toán học lạ mắt** và **công thức từ (1) đến (8)** đều được dịch sang ngôn ngữ đời thường và minh họa bằng **ví dụ con số cụ thể (nhẩm tính được ngay)**.

---

## PHẦN 1: TỪ ĐIỂN GIẢI NGHĨA TẤT CẢ CÁC TỪ VIẾT TẮT (TRA CỨU NHANH)

Khi đọc bài báo hoặc xem giao diện Demo, bạn sẽ liên tục gặp các chữ viết tắt dưới đây. Hãy dùng bảng này để tra cứu:

### 1.1. Các thuật ngữ của riêng phương pháp REDLAMP
| Viết tắt | Tiếng Anh đầy đủ | Nghĩa Tiếng Việt | Giải thích bình dân trong 1 câu |
| :--- | :--- | :--- | :--- |
| **TSAD** | *Time Series Anomaly Detection* | Phát hiện bất thường chuỗi thời gian | Bài toán tìm xem ở giây/phút thứ mấy thì cảm biến (nhiệt độ, nhịp tim, CPU...) bị nhảy số bất thường. |
| **REDLAMP** | *Robust and Explainable Detector of Time Series Anomaly via Augmenting Multiclass Pseudo-Anomalies* | Bộ phát hiện bất thường bền vững & có tính giải thích bằng cách tạo bất thường giả đa lớp | Tên mô hình của bài báo: tự tạo ra **11 kiểu lỗi giả** để dạy AI vừa biết bắt lỗi, vừa chỉ ra được lỗi đó thuộc hình dạng nào. |
| **AM** | *Anomaly Mask* ($W$) | Mặt nạ đánh dấu vùng lỗi | Một tấm bìa chỉ gồm số `0` và `1`, đánh dấu chính xác khúc nào vừa bị chèn lỗi giả để "che mắt" không cho bộ vẽ lại (Decoder) học vẹt lỗi đó. |
| **BC** | *Backward Correction* | Hiệu chỉnh ngược (Nhãn mềm) | Thay vì bắt AI tin tuyệt đối 100% rằng mẫu tạo ra là lỗi (`[0, 1]`), ta giảm xuống còn 79% và luôn chừa 11% khả năng nó vẫn là bình thường (`[0.11, 0.79...]`). |
| **FAA** | *Frequent Anomaly Adjustment* | Bộ lọc bất thường xuất hiện quá thường xuyên | Quy tắc: *"Bất thường thật thì phải hiếm"*. Nếu lúc chạy thực tế mà AI bảo một loại lỗi xuất hiện quá 5% tổng thời gian ($\delta = 0.05$), bộ lọc FAA sẽ coi đó là báo động giả và gạch điểm của loại đó về `0`. |

### 1.2. Các thuật ngữ Mô hình Học sâu (Deep Learning) & Hàm mất mát (Loss)
| Viết tắt | Tiếng Anh đầy đủ | Nghĩa Tiếng Việt | Giải thích bình dân trong 1 câu |
| :--- | :--- | :--- | :--- |
| **AE** | *Autoencoder* | Mạng tự mã hóa (Nén & Khôi phục) | Mạng gồm 2 nửa: **Encoder** (nén đoạn sóng dài thành vài con số tóm tắt) và **Decoder** (từ vài con số đó vẽ khôi phục lại đoạn sóng ban đầu). |
| **1D-CNN** | *1D Convolutional Neural Network* | Mạng tích chập 1 chiều | Mạng dùng một "kính lúp" trượt dọc theo trục thời gian để nhận diện hình dáng sóng (gai nhọn, dốc lên, sóng sin...). |
| **MLP** | *Multi-Layer Perceptron* | Mạng nơ-ron đa tầng (Classifier) | Khối mạng đóng vai **"Bác sĩ chẩn đoán"**: nhận bản tóm tắt từ Encoder rồi phán đoán đoạn sóng thuộc lớp nào trong 12 lớp. |
| **Embedding** | *Latent Embedding Vector* | Vector biểu diễn tiềm ẩn (128 số) | **"Tờ phiếu khám bệnh 128 con số"** do Encoder tóm tắt từ đoạn sóng đầu vào. Từ tờ phiếu này, Classifier sẽ đoán bệnh và Decoder sẽ vẽ lại sóng. |
| **MSE** | *Mean Squared Error* ($\mathcal{L}_{\text{MSE}}$) | Sai số bình phương trung bình | Đo xem **hình vẽ lại của Decoder khác hình gốc bao nhiêu**: lấy `(Gốc - Vẽ lại)` bình phương lên. Nếu lệch nhiều $\implies$ MSE cao $\implies$ có bất thường! |
| **CE** | *Cross-Entropy Loss* ($\mathcal{L}_{\text{CE}}$) | Hàm mất mát Entropy chéo | Đo xem **Classifier đoán 12 lớp sai lệch bao nhiêu** so với nhãn mục tiêu. Đoán càng đúng lớp thì CE càng nhỏ (tiến về 0). |
| **Softmax** | *Softmax Activation* | Hàm chuẩn hóa xác suất | Biến 12 điểm số thô ở đầu ra của Classifier thành **12 con số xác suất cộng lại đúng bằng 100% (1.0)**. |

### 1.3. Các thuật ngữ Đánh giá Kết quả (Metrics) & Phân tích
| Viết tắt | Tiếng Anh đầy đủ | Nghĩa Tiếng Việt | Giải thích bình dân trong 1 câu |
| :--- | :--- | :--- | :--- |
| **FPR** | *False Positive Rate* | Tỷ lệ Dương tính giả (**Báo động giả**) | Máy đang chạy bình thường nhưng AI lại hú còi báo lỗi. |
| **FNR** | *False Negative Rate* | Tỷ lệ Âm tính giả (**Bỏ sót lỗi**) | Máy đang bị hỏng thật nhưng AI lại im lặng bảo bình thường. |
| **ROC / PR** | *Receiver Operating Characteristic / Precision-Recall* | Đường cong ROC và Đường cong Độ chính xác - Độ phủ | Hai biểu đồ chuẩn để đo chất lượng bộ phát hiện lỗi mà không phụ thuộc vào việc chọn ngưỡng cắt cố định. Khi lỗi rất hiếm (như trong chuỗi thời gian), **PR** đánh giá khắt khe và chuẩn xác hơn **ROC**. |
| **R-AUC-ROC / R-AUC-PR** | *Range Area Under the Curve* | Diện tích dưới đường cong tính theo khoảng (Range) | Phiên bản nâng cấp của AUC dành riêng cho chuỗi thời gian: tính điểm theo **cả một đoạn thời gian bị lỗi** (có vùng đệm xung quanh) thay vì chỉ chấm từng giây rời rạc. |
| **VUS-ROC / VUS-PR** | *Volume Under the Surface* | Thể tích dưới bề mặt (Thước đo chuẩn nhất KDD '25) | Thay vì chọn cố định 1 độ dài vùng đệm, VUS quét qua nhiều vùng đệm khác nhau để tạo thành mặt 3D rồi tính thể tích. **Số càng gần `1.0` là mô hình càng xuất sắc.** |
| **RF** | *Range F-score* | Điểm cân bằng F1 theo khoảng thời gian | Điểm trung bình điều hòa giữa Precision (báo động có chuẩn không) và Recall (có bắt hết các đoạn lỗi không). |
| **t-SNE / PCA** | *t-Distributed Stochastic Neighbor Embedding / Principal Component Analysis* | Thuật toán giảm chiều dữ liệu (từ 128D xuống 2D) | Con người không thể nhìn được không gian 128 chiều (128 con số của Embedding), nên ta dùng PCA hoặc t-SNE để "ép phẳng" 128 chiều xuống mặt phẳng 2D ($x, y$) để vẽ biểu đồ chấm điểm. |
| **XAI** | *Explainable Artificial Intelligence* | AI có khả năng giải thích | Không chỉ phán "Có lỗi!", AI còn giải thích được **"Lỗi này có hình dạng giống lỗi nhiễu sóng (Noise) hay lỗi kẹt cảm biến (Cutoff)"**. |

### 1.4. Tên 5 Bộ dữ liệu Chuẩn trong Bài báo
* **UCR (*UCR Anomaly Archive*):** Gồm 250 chuỗi đơn biến ($D=1$ cảm biến) từ y tế (điện tâm đồ ECG, huyết áp ABP), nông nghiệp (côn trùng EPG), công nghiệp... Tập Train hoàn toàn sạch (0% lỗi).
* **AIOps:** Gồm 29 chuỗi đơn biến ($D=1$) theo dõi chỉ số vận hành máy chủ Web đám mây (Cloud KPIs). Tập Train bị lẫn khoảng 2.49% lỗi.
* **SMD (*Server Machine Dataset*):** Gồm 28 máy chủ, mỗi máy chủ theo dõi cùng lúc **$D=38$ chỉ số** (CPU, RAM, Disk, Network...) suốt 5 tuần.
* **SMAP & MSL (*NASA Spacecraft Datasets*):** Dữ liệu thực tế từ Vệ tinh đo độ ẩm đất (**SMAP**) và Tàu tự hành Sao Hỏa (**MSL**) của NASA, theo dõi từ 25 đến 55 kênh cảm biến và bộ truyền động.

---

## PHẦN 2: CÁCH ĐỌC CÁC KÝ HIỆU TOÁN HỌC TRONG BÀI BÁO (KHÔNG SỢ CÔNG THỨC!)

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

## PHẦN 3: CÂU CHUYỆN CỐT LÕI — TẠI SAO CÁC PHƯƠNG PHÁP CŨ THẤT BẠI & TẠI SAO CẦN REDLAMP?

Hãy tưởng tượng bạn cần dạy một AI làm **bảo vệ giám sát nhịp tim hoặc nhiệt độ máy chủ**, nhưng trong dữ liệu lịch sử đưa cho AI học **không hề có ai đánh dấu sẵn chỗ nào là bình thường, chỗ nào là lỗi** (bài toán Không giám sát - *Unsupervised*).

Các nhà khoa học trước nay đi theo 2 hướng chính, nhưng cả 2 đều có "tử huyệt" (minh họa ở **Figure 1** của bài báo):

### Hướng 1: Giả định Bình thường (*Normality Assumption* — dùng Autoencoder, GAN, Transformer thuần túy)
* **Cách làm:** Giả sử 100% dữ liệu trong tập huấn luyện đều là bình thường. Dạy AI học thuộc lòng cách vẽ lại dữ liệu đó. Hy vọng rằng lúc chạy thật, gặp ca bệnh lạ thì AI sẽ vẽ sai và phát hiện ra.
* **Tử huyệt — Nhiễm bẩn bất thường (*Anomaly Contamination*):** Trong thực tế, dữ liệu thu thập hàng tháng trời từ máy chủ **luôn bị lẫn một vài lúc máy bị lỗi mà không ai biết** (gọi là dữ liệu bị nhiễm bẩn). Vì mạng học sâu (Deep Learning) rất thông minh, nó học thuộc lòng luôn cả cách vẽ các cục lỗi nhiễm bẩn này! Đến lúc đi thi gặp đúng lỗi đó, nó vẽ lại đẹp như thường $\implies$ **Bỏ sót lỗi thật (False Negative cao)**.

### Hướng 2: Giả định Bất thường Nhị phân (*Binary Anomaly Assumption* — như NCAD, CutAddPaste)
* **Cách làm:** Tự lấy dữ liệu gốc ra cắt ghép, làm méo đi để tạo thành **"Bất thường giả" (Pseudo-anomalies)**. Sau đó dạy AI phân loại thành **2 nhóm (Nhị phân)**: Nhóm `0` (Dữ liệu gốc = Bình thường) và Nhóm `1` (Dữ liệu cắt ghép = Bất thường).
* **3 Tử huyệt chí mạng:**
  1. **Khoảng trống đa dạng (*Diversity Gap*):** Chỉ dùng 1–2 kiểu cắt ghép thì không thể bắt chước hết muôn hình vạn trạng lỗi ngoài đời thực. Những kiểu lỗi chưa được dạy tạo thành một "lỗ hổng kiến thức" (*Diversity Gap*). Nếu tập Train có lỗi nhiễm bẩn rơi vào lỗ hổng này, AI vẫn bị lừa!
  2. **Bất thường giả vô hại (*False Anomaly / Pseudo-normal*):** Vì máy tính cắt ghép hoàn toàn ngẫu nhiên (chọn bừa đoạn thời gian để làm mượt hoặc cắt dán), có những lúc đoạn bị biến đổi xong **nhìn chẳng khác gì dữ liệu bình thường**! Bài báo gọi đây là **False Anomaly** (mang tiếng là mẫu lỗi giả nhưng thực chất vẫn bình thường). Nếu bắt AI phải tìm cách phân biệt mẫu này với mẫu bình thường, AI sẽ bị tẩu hỏa nhập ma (*Overfitting*) và **báo động giả liên tục (False Positive cao)**.
  3. **Quá tự tin vào nhãn cứng (*Overconfidence on Labels*):** Gán nhãn cứng tuyệt đối `0` (100% bình thường) cho dữ liệu gốc và `1` (100% lỗi) cho dữ liệu cắt ghép là sai lầm, vì dữ liệu gốc có thể chứa lỗi nhiễm bẩn, còn dữ liệu cắt ghép có thể chứa *False Anomaly*.

### 💡 Giải pháp của REDLAMP: Giả định Bất thường Đa lớp (*Multiclass Anomaly Assumption*)
REDLAMP giải quyết trọn vẹn cả 4 tử huyệt trên bằng 4 vũ khí:
1. **Tạo 12 Lớp Đa dạng (1 lớp Bình thường + 11 lớp Lỗi giả chuyên biệt):** Lấp kín *Diversity Gap* và giúp không gian học được có khả năng **giải thích tên loại lỗi (XAI)**.
2. **Mặt nạ che lỗi (*Anomaly Mask* $W$):** Che phần lỗi giả lại khi dạy Decoder vẽ, để Decoder tuyệt đối chỉ học vẽ phần bình thường.
3. **Nhãn mềm (*Backward Correction* $\tilde{y}$):** Không bao giờ tin 100% vào nhãn. Luôn dặn AI: *"Mẫu lỗi giả này vẫn có 11% khả năng là bình thường đấy nhé!"* $\implies$ Trị dứt điểm *Overconfidence*.
4. **Bộ lọc tần suất lúc kiểm thử (*FAA*):** Lúc chạy thực tế, lớp lỗi giả nào bị AI gọi tên quá nhiều ($> 5\%$ tổng thời gian) thì đích thị là lớp *False Anomaly* vô hại $\implies$ Tự động gạch điểm lớp đó về `0`!

---

## PHẦN 4: GIẢI PHẪU TỪNG CÔNG THỨC TOÁN HỌC BẰNG VÍ DỤ SỐ CỤ THỂ

### 4.1. Cửa sổ trượt (Sliding Window — Mục 3.1)
* **Công thức trong bài báo:**
  $$X^{(t)} = \{x_{t-L+1}, \dots, x_t\} \in \mathbb{R}^{D \times L}$$
* **Hiểu đơn giản:** Chuỗi thời gian ban đầu rất dài (ví dụ $T = 25,000$ phút). Mạng AI không đọc cả 25,000 phút cùng lúc mà dùng một khung cửa sổ dài $L = 100$ phút liên tiếp.
  * Tại phút thứ $t = 100$, cửa sổ $X^{(100)}$ lấy từ phút $1$ đến phút $100$.
  * Tại phút thứ $t = 101$, cửa sổ $X^{(101)}$ trượt sang phải 1 nấc, lấy từ phút $2$ đến phút $101$.
* **Mục tiêu:** Với mỗi cửa sổ $X^{(t)}$, AI phải chấm ra một con số điểm bất thường $s(t) \ge 0$. Con số $s(t)$ càng lớn thì khả năng tại thời điểm $t$ đang có sự cố càng cao.

---

### 4.2. Giải thích 12 Phép Biến Đổi Tạo Lỗi Giả & Mặt nạ $W$ (Mục 3.3.2)

Giả sử ta có một đoạn sóng bình thường gồm 6 điểm thời gian:
$$\text{Gốc: } [10, \; 12, \; 14, \; 16, \; 18, \; 20]$$
Máy tính chọn ngẫu nhiên khúc giữa (từ vị trí `st = 2` đến `ed = 5`, tức đoạn `[14, 16, 18]`) để tiêm lỗi giả. Khi đó:
* **Mặt nạ $W$** sẽ bằng `1` ở đúng chỗ bị sửa và bằng `0` ở chỗ giữ nguyên: $W = [0, \; 0, \; \mathbf{1, \; 1, \; 1}, \; 0]$.

Dưới đây là cách 12 phép biến đổi làm thay đổi đoạn `[14, 16, 18]` đó:

| Thứ tự | Tên lớp ($k$) | Công thức trong bài báo | Giải thích bằng lời & Ví dụ số minh họa |
| :---: | :--- | :--- | :--- |
| **1** | **Normal** ($k=1$) | $X_{\text{dim}}^{(t,1)} = X_{\text{dim}}^{(t)}$ | **Giữ nguyên 100%** không sửa gì cả: `[10, 12, 14, 16, 18, 20]`. Toàn bộ mặt nạ $W = [0, 0, 0, 0, 0, 0]$. |
| **2** | **Spike** ($k=2$) | $X_{\text{dim},\text{st}}^{(t,k)} = X_{\text{dim},\text{st}}^{(t)} + a$ | **Cộng một gai nhọn đột biến** tại đúng 1 điểm `st`. Ví dụ cộng $+50$ vào số `14` $\to$ `[10, 12, 64, 16, 18, 20]`. Mặt nạ $W$ chỉ bật số `1` tại đúng điểm gai nhọn đó. |
| **3** | **Flip** ($k=3$) | $X_{\text{dim},\text{st}:\text{ed}}^{(t,k)} = X_{\text{dim},\text{st}:\text{ed}}^{(t)}[::-1]$ | **Lật ngược thứ tự thời gian** (đầu thành đuôi). Đoạn `[14, 16, 18]` bị lật ngược thành `[18, 16, 14]` $\to$ `[10, 12, 18, 16, 14, 20]`. |
| **4** | **Speedup** ($k=4$) | Lấy mẫu cách quãng `[::2]` ($2\times$) hoặc nội suy giãn ra ($0.5\times$) | **Tua nhanh gấp đôi hoặc làm chậm đi một nửa** nhịp điệu của sóng (giống như nhịp tim đang đập đều tự nhiên đập nhanh gấp đôi hoặc chậm đi một nửa). |
| **5** | **Noise** ($k=5$) | $X_{\text{dim},\text{st}:\text{ed}}^{(t,k)} + a, \; a \sim \mathcal{N}(0, 0.1I)$ | **Cộng nhiễu rè (nhiễu trắng)** làm đường sóng bị rung lắc lăn tăn: ví dụ `[14, 16, 18]` $\to$ `[14.3, 15.6, 18.4]`. |
| **6** | **Cutoff** ($k=6$) | Thay cả đoạn bằng 1 hằng số ngẫu nhiên giữa `min` và `max` | **Làm kẹt cảm biến (đường thẳng ngang)**: Đoạn `[14, 16, 18]` bị đơ cứng ở một giá trị cố định, ví dụ `[15, 15, 15]` $\to$ `[10, 12, 15, 15, 15, 20]`. |
| **7** | **Average** ($k=7$) | $\text{MovingAverage}(X_{\text{dim},\text{st}:\text{ed}}^{(t)}, w=L/5)$ | **Làm trơn / làm phẳng sóng** bằng cách lấy trung bình cộng của $w=20$ điểm lân cận, làm mất đi các đỉnh nhọn tự nhiên của sóng. |
| **8** | **Scale** ($k=8$) | $a \times X_{\text{dim},\text{st}:\text{ed}}^{(t)}, \; a \sim \mathcal{N}(1, 1)$ | **Phóng đại hoặc thu nhỏ biên độ**: Nhân cả đoạn với một hệ số $a$ (ví dụ nhân đôi $a=2$: `[14, 16, 18]` $\to$ `[28, 32, 36]`). |
| **9** | **Wander** ($k=9$) | Cộng dốc tuyến tính `linspace(0, a)` rồi giữ nguyên độ lệch $a$ đến cuối cửa sổ | **Trôi đường cơ sở (Baseline Drift)**: Đoạn giữa bị kéo trượt dốc dần lên $+10$, và **toàn bộ phần đuôi phía sau cũng bị kênh lên $+10$**. Vì phần đuôi cũng bị lệch nên mặt nạ $W = 1$ từ `st` đến tận cuối cửa sổ! |
| **10** | **Contextual** ($k=10$) | $a \times X_{\text{dim},\text{st}:\text{ed}}^{(t)} + b$ | **Biến đổi ngữ cảnh**: Kết hợp vừa nhân biên độ ($a$) vừa cộng độ lệch nền ($b$). |
| **11** | **Upsidedown** ($k=11$) | $2 \times \text{Mean} - X_{\text{dim},\text{st}:\text{ed}}^{(t)}$ | **Lật úp ngược đỉnh xuống đáy** qua đường trung bình cộng của đoạn (đỉnh sóng biến thành hố sâu và ngược lại). |
| **12** | **Mixture** ($k=12$) | $X_{\text{dim},\text{st}:\text{ed}}^{(t,k)} = X_{\text{dim},\text{st}:\text{ed}}^{(t')}$ | **Râu ông nọ cắm cằm bà kia**: Cắt một đoạn từ cửa sổ ở thời điểm $t'$ khác ghép đè vào cửa sổ hiện tại $t$. |

---

### 4.3. Công thức (1) — Nhãn Mềm Hiệu Chỉnh Ngược (Backward Correction)

* **Công thức trong bài báo:**
  $$\tilde{y}_i^{(t,k)} = \begin{cases} (1 - \alpha - K\beta)y_i^{(t,k)} + \alpha + \beta, & \text{nếu } i = 1 \text{ (vị trí lớp Normal)}, \\ (1 - \alpha - K\beta)y_i^{(t,k)} + \beta, & \text{nếu } i \in \{2, \dots, 12\} \text{ (vị trí 11 lớp Lỗi)}, \end{cases} \quad (1)$$

* **Giải nghĩa từng chữ cái:**
  * $K = 12$: Tổng số lớp (Lớp số 1 là `Normal`, Lớp số 2 đến 12 là 11 loại lỗi giả).
  * $y^{(t,k)}$: **Nhãn cứng (One-hot)** ban đầu — là danh sách 12 con số, chỉ có đúng vị trí lớp $k$ bằng `1`, còn lại 11 vị trí kia bằng `0`.
  * $\tilde{y}^{(t,k)}$: **Nhãn mềm (Soft label)** sau khi đã được làm dịu đi.
  * $\alpha = 0.1$ ($10\%$): Xác suất dự phòng rằng **"Một mẫu mang tiếng là lỗi giả thực chất nhìn vẫn giống Bình thường (False Anomaly)"**. Vì thế ta luôn cộng bù $\alpha = 0.1$ vào vị trí số 1 (lớp `Normal`)!
  * $\beta = 0.01$ ($1\%$): Xác suất dự phòng rằng **"Một lớp bất kỳ có thể bị nhầm lẫn sang một trong 12 lớp"**.
  * Cụm $(1 - \alpha - K\beta)$: Phần độ tin cậy còn lại dành cho lớp chính. Thay số vào: $1 - 0.1 - 12 \times 0.01 = 1 - 0.1 - 0.12 = \mathbf{0.78}$.

* **🔢 Ví dụ tính tay cụ thể (Hiểu ngay trong 10 giây):**
  * **Trường hợp A — Mẫu đang xét là Lớp 2 (`Spike`, nên $y_2 = 1$ và các vị trí khác $y_i = 0$):**
    * Nhãn cứng ban đầu: $y = [0, \; \mathbf{1}, \; 0, \; 0, \dots, 0]$.
    * Tại ô số 1 (Lớp `Normal`, $y_1 = 0$): $\tilde{y}_1 = 0.78 \times 0 + 0.10 + 0.01 = \mathbf{0.11}$ ($11\%$).
    * Tại ô số 2 (Lớp `Spike`, $y_2 = 1$): $\tilde{y}_2 = 0.78 \times 1 + 0.01 = \mathbf{0.79}$ ($79\%$).
    * Tại 10 ô lỗi còn lại ($y_i = 0$): $\tilde{y}_i = 0.78 \times 0 + 0.01 = \mathbf{0.01}$ ($1\%$).
    * $\implies$ **Nhãn mềm thu được:** $\tilde{y} = [\mathbf{0.11}, \; \mathbf{0.79}, \; 0.01, \; 0.01, \dots, 0.01]$ *(Tổng 12 ô cộng lại $= 0.11 + 0.79 + 10 \times 0.01 = 1.00$)*.
    * **Ý nghĩa:** Khi dạy AI học mẫu `Spike`, ta bảo AI: *"Mẫu này 79% là Spike, nhưng vẫn có 11% khả năng nó là Normal thôi nhé, đừng học cực đoan 100%!"*
  * **Trường hợp B — Mẫu đang xét là Lớp 1 (`Normal`, nên $y_1 = 1$):**
    * Tại ô số 1 (`Normal`, $y_1 = 1$): $\tilde{y}_1 = 0.78 \times 1 + 0.10 + 0.01 = \mathbf{0.89}$ ($89\%$).
    * Tại 11 ô lỗi còn lại ($y_i = 0$): $\tilde{y}_i = \mathbf{0.01}$ ($1\%$ mỗi lớp).
    * **Ý nghĩa:** Ngay cả mẫu trong tập Train gốc (gán nhãn `Normal`), ta cũng chỉ tin nó $89\%$ là bình thường, chừa lại $11\%$ ($11 \times 1\%$) khả năng nó là **lỗi nhiễm bẩn (Anomaly Contamination)** trà trộn vào!

---

### 4.4. Công thức (2), (3), (4) — Hàm Mất Mát Khi Huấn Luyện (Objective Function)

Trong lúc huấn luyện (Training), AI vừa phải tập **vẽ lại sóng bình thường** (qua nhánh Decoder), vừa phải tập **đoán tên 12 lớp** (qua nhánh Classifier).

#### 1) Công thức (2) — Hàm mất mát tái tạo có mặt nạ ($\mathcal{L}_{\text{MSE}}$):
$$\mathcal{L}_{\text{MSE}} = \left\| \left(1 - W^{(t,k)}\right) \odot \left(X^{(t,k)} - \hat{X}^{(t,k)}\right) \right\|_2^2 \quad (2)$$

* **Tại sao phải nhân với $(1 - W^{(t,k)})$?**
  * Nhớ lại: Mặt nạ $W$ bằng `1` ở vùng vừa bị chèn lỗi giả, và bằng `0` ở vùng bình thường.
  * Vậy $(1 - W)$ sẽ bằng **`0` ở vùng lỗi giả** và bằng **`1` ở vùng bình thường**!
* **🔢 Ví dụ tính tay 4 con số:**
  * Sóng bình thường gốc: $[10, \; 10, \; 10, \; 10]$.
  * Ta chèn lỗi `Spike` vào vị trí thứ 3 $\implies$ Sóng đưa vào $X^{(t,k)} = [10, \; 10, \; \mathbf{50}, \; 10]$, và mặt nạ $W = [0, \; 0, \; \mathbf{1}, \; 0]$.
  * Giả sử Decoder vẽ ra đường sóng hoàn toàn bình thường: $\hat{X}^{(t,k)} = [10, \; 9, \; \mathbf{10}, \; 10]$.
  * Hiệu số $(X - \hat{X}) = [0, \; 1, \; \mathbf{40}, \; 0]$. *(Chỗ số 40 rất lớn vì Decoder không chịu vẽ cái gai nhọn 50).*
  * Nhân với $(1 - W) = [1, \; 1, \; \mathbf{0}, \; 1]$:
    $$(1 - W) \odot (X - \hat{X}) = [1\times 0, \; 1\times 1, \; \mathbf{0\times 40}, \; 1\times 0] = [0, \; 1, \; \mathbf{0}, \; 0]$$
  * Bình phương và cộng lại: $\mathcal{L}_{\text{MSE}} = 0^2 + 1^2 + 0^2 + 0^2 = 1$.
  * **Bài học rút ra:** Nhờ nhân với `0` ở vị trí có lỗi giả, **Decoder KHÔNG bị phạt khi nó từ chối vẽ cái gai nhọn 50**! Nhờ đó, Decoder chỉ tập trung học vẽ những phần bình thường. Đến lúc chạy thật (Test), gặp lỗi thật, Decoder sẽ tiếp tục vẽ đường bình thường $\implies$ lệch lớn với lỗi thật $\implies$ Bắt được lỗi!

#### 2) Công thức (3) — Hàm mất mát phân loại nhãn mềm ($\mathcal{L}_{\text{CE}}$):
$$\mathcal{L}_{\text{CE}} = -\sum_{i=1}^{K} \tilde{y}_i^{(t,k)} \log\left(\hat{y}_i^{(t,k)}\right) \quad (3)$$
* **Giải thích đơn giản:**
  * $\hat{y}_i^{(t,k)}$ là xác suất mà nhánh Classifier dự đoán cho lớp thứ $i$ (từ $1$ đến $12$).
  * $\tilde{y}_i^{(t,k)}$ là nhãn mềm mục tiêu vừa tính ở Công thức (1).
  * Hàm $-\log(\hat{y}_i)$ có tính chất: Nếu AI đoán đúng xác suất cao ($\hat{y}_i$ gần $1.0$), thì $-\log(\hat{y}_i)$ rất nhỏ (gần $0$ = không bị phạt). Nếu AI đoán sai ($\hat{y}_i$ gần $0$), thì $-\log(\hat{y}_i)$ vọt lên rất lớn (= bị phạt nặng).

#### 3) Công thức (4) — Tổng hợp hàm mất mát ($\mathcal{L}$):
$$\mathcal{L} = \gamma \mathcal{L}_{\text{CE}} + (1 - \gamma) \mathcal{L}_{\text{MSE}} \quad (4)$$
* Với $\gamma = 0.1$, tổng mức phạt khi huấn luyện bằng **$10\%$ điểm phạt phân loại ($\mathcal{L}_{\text{CE}}$) + $90\%$ điểm phạt vẽ lại sóng ($\mathcal{L}_{\text{MSE}}$)**. Cả hai nhánh cùng rèn luyện chung bộ não Encoder!

---

### 4.5. Công thức (5), (6), (7), (8) — Chấm Điểm Bất Thường Lúc Kiểm Thử (Test Time) & Bộ Lọc FAA

Khi mang mô hình đã học xong ra chạy trên tập dữ liệu thực tế $X^{\text{test}}$ (lúc này không còn chèn lỗi giả nữa mà đi tìm lỗi thật):

#### Bước 1: Tính điểm sai số vẽ lại sóng — Công thức (5)
$$s_{\text{MSE}}(t) = \left\| X^{(t)} - \hat{X}^{(t)} \right\|_2^2 \quad (5)$$
* Lấy cửa sổ thực tế $X^{(t)}$ trừ đi hình vẽ lại $\hat{X}^{(t)}$ của Decoder rồi bình phương cộng tổng lại. Nếu cửa sổ có lỗi thật, Decoder không biết vẽ lỗi nên $s_{\text{MSE}}(t)$ sẽ cao vọt lên.

#### Bước 2: Lọc bỏ "Lớp giả bình thường" bằng bộ lọc FAA — Công thức (6) & (7)
$$\text{FAA}\left(\hat{y}_i^{(t)}\right) = \begin{cases} 0, & \text{nếu } \frac{1}{T - L + 1}\sum_{j=L}^{T} \hat{y}_i^{(j)} > \delta, \\ \hat{y}_i^{(t)}, & \text{ngược lại}, \end{cases} \quad (6)$$
$$s_{\text{CE}}(t) = \sum_{i=2}^{K} \text{FAA}\left(\hat{y}_i^{(t)}\right) \quad (7)$$

* **Giải thích từng ký hiệu trong Công thức (6):**
  * Cụm $\frac{1}{T - L + 1}\sum_{j=L}^{T} \hat{y}_i^{(j)}$: Chính là **trung bình cộng xác suất dự đoán của lớp lỗi thứ $i$ trên toàn bộ chuỗi thời gian kiểm thử**.
  * $\delta = 0.05$ ($5\%$): Ngưỡng trần cho phép.
* **🔢 Ví dụ thực tế về FAA:**
  * Giả sử trên một chuỗi dữ liệu có chu kỳ rất dài, phép biến đổi `Average` (Lớp 7) lúc huấn luyện tạo ra các mẫu nhìn y hệt dữ liệu bình thường (*False Anomaly*).
  * Hệ quả là khi chạy trên tập Test, lúc nào máy đang chạy bình thường thì Classifier cũng tưởng nhầm là lớp `Average`! Trung bình trên toàn bộ chuỗi Test, xác suất của lớp `Average` đạt tới **$0.18$ ($18\%$)** — lớn hơn ngưỡng $\delta = 0.05$ ($5\%$).
  * Bộ lọc **FAA** lập tức phán: *"Không có cái lỗi thật nào lại xuất hiện tới 18% toàn thời gian cả! Lớp Average này là lớp giả bình thường (Pseudo-normal class)!"* $\implies$ FAA **gạch điểm của lớp `Average` về `0`** ở mọi thời điểm!
  * Sau khi đã lọc sạch các lớp báo động giả, **Công thức (7)** cộng tổng xác suất của các lớp lỗi còn lại (từ lớp $i=2$ đến $12$, bỏ qua lớp $i=1$ là Normal) để ra điểm phân loại $s_{\text{CE}}(t)$.

#### Bước 3: Cộng gộp thành Điểm Bất Thường Cuối Cùng — Công thức (8)
$$s(t) = \frac{1}{2}\text{MinMax}_T\left(s_{\text{MSE}}(t)\right) + \frac{1}{2}\text{MinMax}_T\left(s_{\text{CE}}(t)\right) \quad (8)$$
* **Tại sao phải có $\text{MinMax}_T(\cdot)$ trước khi cộng?**
  * Vì điểm sai số vẽ lại $s_{\text{MSE}}(t)$ có thể lên tới hàng chục (ví dụ từ `0.2` đến `45.0`), trong khi điểm tổng xác suất $s_{\text{CE}}(t)$ chỉ nằm trong khoảng từ `0.0` đến `1.0`. Nếu cộng trực tiếp thì $s_{\text{MSE}}$ sẽ lấn át hoàn toàn $s_{\text{CE}}$.
  * Hàm $\text{MinMax}_T$ kéo cả hai thang điểm về cùng một thang chuẩn từ **`0.0` (thấp nhất chuỗi) đến `1.0` (cao nhất chuỗi)** theo công thức:
    $$\text{MinMax}_T(s(t)) = \frac{s(t) - \min(s)}{\max(s) - \min(s)}$$
  * Sau đó lấy trung bình cộng ($\frac{1}{2}$ điểm tái tạo chuẩn hóa + $\frac{1}{2}$ điểm phân loại chuẩn hóa) để thu được điểm bất thường cuối cùng $s(t) \in [0, 1]$.

---

## PHẦN 5: TẠI SAO BÀI BÁO DÙNG THƯỚC ĐO `VUS-PR` VÀ `VUS-ROC`?

Nếu bạn mới học Machine Learning, bạn sẽ quen với `Accuracy` (Độ chính xác) hoặc `F1-Score`. Nhưng trong Chuỗi thời gian (TSAD), các thước đo cũ gặp 2 vấn đề lớn:
1. **Bất thường rất hiếm (Mất cân bằng dữ liệu nặng):** Trong tập UCR, lỗi chỉ chiếm `0.85%` thời gian (còn `99.15%` là bình thường). Một mô hình "lười biếng" lúc nào cũng đoán Bình thường sẽ đạt ngay `Accuracy = 99.15%` dù không bắt được bất kỳ lỗi nào!
2. **Lỗi kéo dài theo khoảng thời gian (Range) và phụ thuộc ngưỡng cắt (Threshold):** Khi sự cố xảy ra từ giây thứ 100 đến giây thứ 120, nếu AI báo động cao ở giây 98 (sớm hơn 2 giây do cửa sổ trượt) thì về mặt thực tế vẫn là rất tốt.

Vì vậy, KDD '25 sử dụng bộ thước đo chuẩn mới nhất:
* **Range-AUC-ROC (`R-R`) & Range-AUC-PR (`R-P`):** Mở rộng vùng lỗi thật ra hai đầu một khoảng đệm (buffer) bằng nửa cửa sổ ($L/2 = 50$ bước) và tính diện tích dưới đường cong ROC / PR trên mọi ngưỡng cắt.
* **VUS-ROC (`V-R`) & VUS-PR (`V-P` — *Volume Under the Surface*):** Thay vì chỉ tính với 1 độ dài vùng đệm cố định, VUS thay đổi độ dài vùng đệm liên tục để tạo thành một mặt cong 3D rồi tính thể tích dưới mặt cong đó. Trong đó, **`VUS-PR` là thước đo khắt khe và đáng tin cậy số 1 hiện nay** (càng gần `1.0` càng tốt).

### Tóm tắt Kết quả Thực nghiệm Nổi bật của REDLAMP (Mục 4.2 – 4.6)
1. **Đứng đầu trên 5 bộ dữ liệu (Q1):** Đạt trung bình **`VUS-ROC = 0.803`**, **`VUS-PR = 0.430`** (vượt phương pháp đứng thứ hai là *CutAddPaste* tới **+17.4%**).
2. **Chứng minh vai trò của từng thành phần (Q2 - Ablation Study):** Nếu tắt bất kỳ thành phần nào (`w/o AM` - tắt mặt nạ $W$, `w/o BC` - tắt nhãn mềm, `w/o FAA` - tắt bộ lọc FAA, hoặc gộp 11 lớp lỗi về phân loại nhị phân `Binary`), điểm `VUS-PR` lập tức tụt giảm mạnh (giảm tới **24.5%** trên SMD khi quay về phân loại nhị phân!).
3. **Sức đề kháng trước dữ liệu bẩn (Q4 - Robustness):** Khi cố tình trộn thêm **5% lỗi vào tập Train** của UCR để làm bẩn dữ liệu, REDLAMP chỉ giảm nhẹ **1.8%** điểm `VUS-PR`, trong khi *CutAddPaste* sụt giảm tới **15.5%** và các mô hình Autoencoder/Transformer thuần túy sụp đổ nặng nề.
4. **Khả năng giải thích nguyên nhân lỗi (Q5 - Explainability):** Nhìn vào biểu đồ phân cụm 2D của các vector `Embedding` 128 chiều, các điểm lỗi thật trên tập Test nằm sát cụm lớp lỗi giả nào sẽ chỉ ra ngay hình thái vật lý của sự cố đó (ví dụ: lỗi nhịp sinh học côn trùng EPG nằm sát cụm `Speedup`; lỗi nhiễu sóng huyết áp ABP nằm sát cụm `Noise`; lỗi máy chủ SMD nằm gần `Spike` và `Cutoff`).

---

## PHẦN 6: BẢNG TỔNG HỢP SIÊU THAM SỐ CỐ ĐỊNH CỦA MÔ HÌNH (TABLE 4)

Toàn bộ các tập dữ liệu trong bài báo đều dùng chung một bộ thông số duy nhất (không cần chỉnh tay cho từng tập):

| Tên tham số | Ký hiệu | Giá trị mặc định | Ý nghĩa dễ hiểu |
| :--- | :---: | :---: | :--- |
| Kích thước cửa sổ trượt | $L$ | `100` | Mỗi lần AI nhìn một đoạn gồm 100 điểm thời gian liên tiếp. |
| Số chiều không gian ẩn (Embedding) | `emb_dim` | `128` | Đoạn sóng 100 điểm được Encoder tóm tắt thành đúng 128 con số. |
| Tổng số lớp phân loại | $K$ | `12` | Gồm 1 lớp `Normal` + 11 lớp bất thường giả. |
| Xác suất mẫu lỗi giả thực ra là bình thường | $\alpha$ | `0.10` ($10\%$) | Dùng trong Công thức (1) để làm mềm nhãn về phía lớp Normal. |
| Xác suất nhầm lẫn giữa các lớp | $\beta$ | `0.01` ($1\%$) | Dùng trong Công thức (1) để chia đều xác suất dự phòng cho mọi lớp. |
| Trọng số hàm mất mát phân loại | $\gamma$ | `0.10` | Dùng trong Công thức (4): $\mathcal{L} = 0.1\mathcal{L}_{\text{CE}} + 0.9\mathcal{L}_{\text{MSE}}$. |
| Ngưỡng lọc lớp giả bình thường FAA | $\delta$ | `0.05` ($5\%$) | Dùng trong Công thức (6): Lớp lỗi nào xuất hiện $> 5\%$ toàn thời gian Test sẽ bị gạch điểm về 0. |
