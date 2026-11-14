# 📋 CẨM NANG KHÁI NIỆM & LỜI DẪN THUYẾT TRÌNH WEB DEMO REDLAMP (TỪ TRÊN XUỐNG DƯỚI)

* **Bài báo gốc:** *REDLAMP — Robust and Explainable Detector of Time Series Anomaly via Augmenting Multiclass Pseudo-Anomalies* (ACM SIGKDD '25, [`2505.20765v1.pdf`](./2505.20765v1.pdf))
* **Bản gộp riêng Lời dẫn thuyết trình (đọc liền mạch):** [`RedLamp_Demo_Script.md`](./RedLamp_Demo_Script.md) | **Tính tay chi tiết:** [`RedLamp_ConvAEC_StepByStep.md`](./RedLamp_ConvAEC_StepByStep.md) | **Tổng quan lý thuyết:** [`RedLamp_Guide.md`](./RedLamp_Guide.md)

> [!IMPORTANT]
> **2 Bí quyết "sống còn" để thao tác mượt tuyệt đối trên sân khấu:**
> 1. **Mẹo thử nghiệm dự đoán báo XANH / ĐỎ chuẩn mực ngay trong miền `[0, 1]` ở Tab 3:**
>    * Khi vừa mở Tab 3 (hoặc bấm nút **`Mẫu chuẩn`**), hệ thống đã điền sẵn bộ thông số vận hành chuẩn và báo **MÀU XANH — `BÌNH THƯỜNG`** (`Normal 99.9%`, `Combined = 0.0009 < Ngưỡng μ+3σ = 0.4368`).
>    * Để giả lập đột biến ngay trong miền chuẩn hóa `[0, 1]` trên **`SMD (38 đặc trưng)`**: bấm nút **`⚡ Mẫu đột biến`** *(hoặc tự gõ **`1`** vào **hàng đầu tiên `D1..D5`** rồi bấm `⚡ Dự đoán bất thường`)* $\rightarrow$ cả hàng `D1..D5` sáng viền đỏ (`Đã sửa 5 kênh`), bảng kết quả lập tức nổ **MÀU ĐỎ — `BẤT THƯỜNG`** với **`Spike (Đột biến đỉnh nhọn) - 99.9%`** (`Combined = 0.5008 > 0.4368`)! *(Hoặc trên tập 1 kênh `AIOps` / `UCR`: chỉ cần gõ `1` vào ô duy nhất `D1` hoặc bấm `⚡ Mẫu đột biến`).*
> 2. **Danh sách Thực thể nạp tức thì `0.2 giây` Offline (đã lưu sẵn trong `result/`, không cần Wi-Fi):**
>    * **`SMD (38 đặc trưng)`:** `ALL` *(đại diện bởi `machine-1-1`)*, `machine-1-1` *(Seed 0)*, `machine-1-2` *(Seed 0, 1)*, `machine-1-3` *(Seed 0)*, `machine-1-4` *(Seed 0)*.
>    * **`AIOps / IOPS (1 đặc trưng)`:** `ALL` và thực thể đầu tiên `KPI-05f10d3a-239c-3bef-9bdc-a2feeb0037aa` *(Seed 0)*.
>    * **`MSL (55 đặc trưng)`:** `ALL`, `C-1`, `D-14`, `D-15`, `D-16`, `F-8` *(Seed 0)*.
>    * **`SMAP (25 đặc trưng)`:** `ALL`, `A-1`, `A-2`, `A-3` *(Seed 0)*.
>    * **`UCR (250 thực thể - 1 đặc trưng)`:** `ALL`, `001` *(Seed 0)*.

---

## 1. TAB `1. Tổng quan` (Đi lần lượt từ trên xuống dưới)

### 1.1. Khối trên cùng: `Bộ chọn Dataset & Entity`

> **🗣️ Lời dẫn thuyết trình:**  
> *"Kính thưa thầy và các bạn, tiếp nối phần lý thuyết, nhóm xin phép trình bày hệ thống **Web Demo trực quan hóa mô hình REDLAMP** — công trình vừa được công bố tại hội nghị **ACM SIGKDD 2025** về bài toán **phát hiện bất thường chuỗi thời gian không giám sát**. Toàn bộ trọng số và kết quả trên hệ thống này đều do nhóm trực tiếp huấn luyện tái lập trên **5 bộ dữ liệu chuẩn quốc tế** đa lĩnh vực: từ giám sát máy chủ công nghiệp `SMD`, chỉ số KPI đám mây `AIOps`, dữ liệu tàu tự hành Sao Hỏa `MSL` và vệ tinh `SMAP` của NASA, cho tới kho dữ liệu đa ngành `UCR`.*
>
> *Nhìn vào thanh điều khiển trên cùng, hệ thống tổ chức dữ liệu theo 3 cấp từ lớn đến nhỏ: **Tập dữ liệu (`Dataset`)**, **Thực thể (`Entity`)** và **Kênh cảm biến (`Channel`)**. Mặc định trên màn hình đang là tập **`SMD`**, bao gồm **28 thực thể máy chủ độc lập**, trong đó mỗi máy chủ được gắn **38 kênh cảm biến** theo dõi đồng thời CPU, bộ nhớ RAM, ổ cứng và lưu lượng mạng. Khi nhóm để chế độ **`Entity = ALL`**, hệ thống sẽ tổng hợp bức tranh toàn cảnh trên cả 28 máy chủ; đồng thời ô **`Seed huấn luyện (0 – 4)`** bên phải cho phép kiểm chứng độ ổn định của mô hình qua **5 lần huấn luyện độc lập** với các vị trí chèn lỗi giả và khởi tạo trọng số ngẫu nhiên khác nhau."*

1. **Phân biệt 3 cấp độ `Dataset` $\supset$ `Entity` $\supset$ `Channel (Đặc trưng)`:**
   * **`Dataset` (Tập dữ liệu):** 5 bộ dữ liệu chuẩn gồm **`SMD`** (28 máy chủ, `38` kênh), **`AIOps / IOPS`** (29 chuỗi KPI web đám mây, `1` kênh), **`MSL`** (23 thực thể tàu tự hành Sao Hỏa NASA, `55` kênh), **`SMAP`** (49 thực thể vệ tinh đo độ ẩm đất NASA, `25` kênh), và **`UCR`** (250 chuỗi đa lĩnh vực y sinh/công nghiệp, `1` kênh).
   * **`Entity` (Thực thể):** Là **1 đối tượng / máy móc độc lập** bên trong tập dữ liệu (ví dụ `machine-1-1` là máy chủ số 1 của `SMD`).
   * **`Channel / Feature` (Kênh cảm biến đầu vào):** Là từng cột thông số đo bên trong 1 thực thể (ví dụ `machine-1-1` có `38` kênh `D1..D38` theo dõi CPU, RAM, Disk, Network...).
2. **Chế độ `Entity = ALL` vs. Chọn `1 Thực thể` cụ thể:**
   * **`ALL`:** Bảng Metrics và biểu đồ cột tổng hợp điểm trung bình trên **toàn bộ các thực thể** của bộ dữ liệu đó (ví dụ trung bình cả 28 máy của `SMD`).
   * **`1 Thực thể` (ví dụ `machine-1-1`):** Hiển thị kết quả, mô hình và toàn bộ chuỗi thời gian kiểm thử của riêng thực thể đó *(có thể nhấp đúp trực tiếp vào cột trên biểu đồ `ALL` để nhảy vào thực thể đó)*.
3. **`Seed huấn luyện (0 - 4)`:**
   * Dữ liệu `Train` và `Test` của chuỗi thời gian luôn giữ cố định theo thứ tự thời gian, không xáo trộn.
   * Con số `Seed 0..4` đại diện cho **5 lần huấn luyện độc lập** với 3 yếu tố ngẫu nhiên khác nhau: *(1)* vị trí, kênh và cường độ chèn lỗi giả lúc tạo dữ liệu tự giám sát, *(2)* khởi tạo trọng số ban đầu của mạng `ConvAEC`, và *(3)* thứ tự chia gói *batch* qua từng *epoch*.

---

### 1.2. Khối giữa: `Kết quả đánh giá chất lượng (Metrics)`

> **🗣️ Lời dẫn thuyết trình:**  
> *"Nhìn xuống khung **Kết quả đánh giá chất lượng** ở ngay bên dưới: theo đúng chuẩn nghiêm ngặt của hội nghị KDD '25, nhóm không sử dụng độ chính xác `Accuracy` hay chỉ số `F1-PA` truyền thống. Lý do là trong chuỗi thời gian thực tế, hơn 96% thời lượng máy chạy bình thường nên đoán toàn bộ là bình thường cũng đạt 96% `Accuracy`; còn `F1-PA` có nhược điểm là chỉ cần chạm trúng 1 giây trong cả đợt sự cố dài hàng trăm giây là tự tính đúng cả đoạn, tạo ra điểm số cao giả tạo.*
>
> *Thay vào đó, hệ thống đánh giá bằng bộ 3 thước đo theo phân đoạn liên tục là **`VUS-ROC`**, **`VUS-PR`** và **`RF (Range F-score)`** — xét đồng thời mọi ngưỡng cắt và vùng đệm dung sai quanh rìa sự cố. Trong ba chỉ số này, **`VUS-PR` là thước đo chính khắt khe nhất**: các bạn thấy `VUS-ROC` đạt rất cao (`0.82`), nhưng `VUS-PR` chỉ khoảng `0.37` — không phải vì mô hình yếu, mà vì sự cố rất hiếm nên chỉ cần có một ít báo động giả là độ chuẩn xác `Precision` bị phạt sụt ngay lập tức. Trên toàn bộ 28 máy của tập `SMD`, kết quả tái lập trung bình qua 5 seed của nhóm đạt **`VUS-PR = 0.374 ± 0.013`**, bám rất sát và nhỉnh hơn nhẹ so với mức `0.366` công bố trong bài báo gốc."*

1. **Lý do dùng bộ 3 chỉ số theo phân đoạn (*Range-based*) thay cho `Accuracy` và `F1-PA`:**
   * `Accuracy` cổ điển vô nghĩa vì `96%–99%` thời gian là bình thường (đoán toàn `0` vẫn đạt `99%`).
   * `F1-PA` (*Point Adjustment*) bị điểm ảo vì quy định chỉ cần đoán trúng `1` giây trong đoạn sự cố dài là tự sửa cả đoạn thành đúng `100%`.
2. **`VUS-ROC` (*Volume Under the ROC Surface*, miền `[0, 1]`):**
   * Thay vì chọn 1 ngưỡng cắt cố định và 1 vùng đệm cố định quanh rìa sự cố, `VUS-ROC` quét qua **mọi ngưỡng cắt** đồng thời mở rộng **độ dài vùng đệm mềm từ `0` đến `100` bước** quanh mép sự cố để tính **thể tích dưới mặt cong 3D** `(FPR, TPR, độ dài đệm)`. Đo khả năng phân tách tổng quát giữa vùng lỗi và vùng chuẩn.
3. **`VUS-PR` (*Volume Under the PR Surface*, miền `[0, 1]` — Thước đo chính của bài báo):**
   * Tính thể tích dưới mặt cong 3D trong không gian `(Recall, Precision, độ dài đệm)`.
   * **Tại sao `VUS-PR` (`0.374` trên `SMD`) luôn thấp hơn nhiều so với `VUS-ROC` (`0.823`)?** Không phải do mô hình kém, mà vì sự cố rất hiếm (`1%–4%` chuỗi): nếu có vài chục điểm báo động giả, khi chia cho hàng chục nghìn điểm bình thường thì `FPR` gần như bằng `0` nên `VUS-ROC` vẫn rất cao; nhưng trong `Precision = TP / (TP + FP)` chỉ tính riêng những lúc mô hình hú còi nên bị phạt sụt mạnh. Vì vậy `VUS-PR` là thước đo khắt khe nhất để xếp hạng các mô hình.
4. **`RF (Range F-score)` (miền `[0, 1]`):**
   * Cắt ngưỡng nhị phân tự động bằng quy tắc thống kê `mean + 3·std`, sau đó tính điểm `F1` trực tiếp trên các đoạn thời gian sự cố (`20%` điểm thưởng phát hiện tồn tại + `80%` điểm tỷ lệ độ phủ thời gian) và chia phạt `1/k` nếu 1 sự cố liền mạch bị mô hình báo băm vụn thành `k` đoạn rời rạc.
5. **Ý nghĩa 2 dòng số trong mỗi ô Metric:**
   * **Con số lớn in đậm** (ví dụ `0.3569`): Kết quả thực tế của `Seed` đang chọn (nếu đang ở `ALL` là trung bình qua 28 máy ở `Seed 0`).
   * **Dòng chữ nhỏ bên dưới** (`Trung bình 5 seed: 0.374 ± 0.013`): Giá trị trung bình ± độ lệch chuẩn tính qua cả 5 lần chạy `Seed 0..4` — dùng để đối chiếu trực tiếp 1–1 với bảng kết quả trong bài báo gốc (`0.366`).

> [!NOTE]
> **📌 Kiến thức nền dự phòng khi hội đồng hỏi sâu về siêu tham số huấn luyện** *(đã ẩn khỏi giao diện Tab 1 để màn hình tinh gọn)*:
> * **Kiến trúc `ConvAEC`:** Mạng chữ Y gồm `Encoder` (4 khối `1D-CNN` nén về `z = 128`) tách ra `Decoder` khôi phục tín hiệu và `Classifier` (`128 → 32 → 12` lớp: `1` chuẩn + `11` lỗi giả tự sinh).
> * **Bước trượt cửa sổ `W = 100`:** Lúc `Train` nhảy `10` bước/lần (giảm trùng lặp thừa, vẫn gối đầu `90/100` điểm, nhanh gấp 10 lần); lúc `Test` trượt sát `1` bước/lần để chấm điểm mọi thời điểm.
> * **3 cơ chế phòng vệ cốt lõi:** *(1)* Nhãn mềm `Backward Correction` (`α = 0.10, β = 0.01` — chống lỗi giả vô tình giống bình thường và chống nhiễm bẩn tập train); *(2)* Hàm mất mát `0.9·MSE (Mask) + 0.1·CE` (dùng `Anomaly Mask = 0` che vùng lỗi giả để `Decoder` chỉ học quy luật bình thường); *(3)* Lọc `FAA` (`δ = 0.05` — loại các lớp xuất hiện `> 5%` chuỗi) và làm mượt trung bình trượt `MA = 50` bước (`W/2`).

---

### 1.3. Khối dưới cùng: `Biểu đồ tổng quan`

> **🗣️ Lời dẫn thuyết trình:**  
> *"Nhìn xuống biểu đồ lớn ở dưới cùng: khi đang ở chế độ `ALL`, biểu đồ cột đối chiếu điểm `VUS-PR` của từng máy trong 28 máy chủ với đường trung bình toàn tập màu đỏ nét đứt. Khi nhóm **rê chuột lên từng cột**, bảng chú thích nổi sẽ báo ngay máy đó đứng trên hay dưới mức trung bình tập, hiển thị đồng thời điểm của Seed đang chọn, trung bình 5 seed và trọn bộ cả 3 chỉ số `ROC · PR · RF`. Bây giờ, nhóm sẽ **nhấp đúp trực tiếp vào cột đầu tiên `m1-1`** để đi sâu vào chuỗi thời gian thực tế dài **`28,381` bước** của máy chủ `machine-1-1`.*
>
> *Trên đồ thị vừa mở ra, đường **màu xám đen** biểu diễn `Độ lệch Max` trên cả 38 cảm biến để ta quan sát biến động tổng thể của máy trên một đường duy nhất (góc trên phải vẫn cho phép chọn xem riêng từng kênh `D1..D38`), còn đường **màu đỏ** là điểm bất thường đầu ra của REDLAMP. Ở đoạn đầu từ `0` đến `5,500`, mặc dù tín hiệu máy chủ nhấp nhô lên `0.85` theo chu kỳ tải công việc bình thường, mô hình hiểu đúng quy luật nên **đường màu đỏ nằm phẳng lặng sát mức `0.00`**, không hề báo giả. Ngược lại, tại cả **8 đợt sự cố thật**, đường màu đỏ đều lập tức bật cao vượt `0.50` — chạm đỉnh **`1.00`** tại sự cố lớn nhất **`t = 17,407`**. Đặc biệt khi **rê chuột dọc theo đồ thị**, vạch gióng sẽ chỉ đích danh ngay thời điểm `t` đó đang là `Bình thường` hay `Bất thường (> μ + 3σ)`, cảm biến nào trong 38 kênh đang biến động mạnh nhất, kèm tỷ lệ điểm thô `Raw Combined / μ + 3σ`; và nhấp đúp vào bất kỳ đỉnh nào cũng sẽ nhảy thẳng sang Tab 2 để mổ xẻ thời điểm đó."*

1. **Khi chọn `Entity = ALL`:**
   * Hiển thị biểu đồ cột so sánh chỉ số (`VUS-PR`, `VUS-ROC` hoặc `RF`) của toàn bộ các thực thể đối chiếu với đường ngang đứt nét đỏ **`Trung bình toàn tập`**.
   * **🖱️ Khi rê chuột & nhấp chuột vào từng cột:** Hộp thông tin nổi (*Tooltip*) hiện tên thực thể (ví dụ `Entity: machine-1-1`), gắn huy hiệu xếp hạng (`★ Cao nhất tập` / `≥ Trung bình tập` / `< Trung bình tập`), điểm của Seed hiện tại, `Trung bình 5 seed ± độ lệch chuẩn`, và liệt kê đủ cả 3 chỉ số `ROC · PR · RF` cùng lúc. **Nhấp đúp chuột** vào cột bất kỳ sẽ mở chi tiết chuỗi thời gian của thực thể đó.
2. **Khi chọn `1 Thực thể` (ví dụ `machine-1-1`, dài `28,381` bước):**
   * **Hộp `Tín hiệu hiển thị`:** `Độ lệch Max` (lấy kênh biến động mạnh nhất trong 38 kênh tại mỗi bước để bao quát mọi lỗi trên 1 đường), `Trung bình cộng` (trung bình cả 38 kênh), hoặc xem riêng từng kênh `D1 .. D38`.
   * **Ý nghĩa các đường & vùng màu:** Đường **xám đen** = tín hiệu cảm biến; Dải nền **màu hồng** = khoảng thời gian xảy ra sự cố thật (*Ground Truth*); Đường **màu đỏ** = điểm bất thường tổng hợp của REDLAMP đã chuẩn hóa về `[0, 1]`.
   * **🖱️ Khi rê chuột & nhấp đúp dọc chuỗi thời gian:** Vạch gióng dọc hiện bảng tra cứu tức thời gồm: `Vị trí t = ...`, huy hiệu trạng thái `⚠️ Bất thường (> μ + 3σ)` hoặc `✓ Bình thường`, biên độ tín hiệu kèm **tên kênh cảm biến đang lệch lớn nhất** (ví dụ `tại D7`), `Điểm tương đối [0, 1]` và tỷ lệ `Raw Combined / μ + 3σ`. **Nhấp đúp chuột** tại bất kỳ vị trí `t` nào sẽ chuyển thẳng sang Tab 2 để giải thích ngay cửa sổ đó.
   * **Giải thích hiện tượng trên đồ thị `machine-1-1` (nhiều đỉnh đỏ dừng ở `~0.51`, chỉ 1 đỉnh `t = 17,407` chạm `1.00`):** Công thức tổng hợp lấy trung bình `0.5 × MinMax(MSE) + 0.5 × MinMax(CE)`. Do sự cố tại `t = 17,407` có biên độ vọt quá lớn làm `MinMax(MSE)` của các đợt lỗi nhỏ khác bị ép xuống `~0.02–0.05`, nhưng nhờ nhánh **`Classifier` vẫn nhận diện trúng biến dạng sóng với xác suất `CE ≈ 1.00`**, điểm tổng hợp vẫn vọt lên `(0.03 + 1.00) / 2 ≈ 0.515` để báo động đầy đủ cả 8 đợt sự cố.

---

## 2. TAB `2. Giải thích mô hình` (Mổ xẻ 5 bước từ trái sang phải)

### 2.0. Thanh điều khiển trên cùng

> **🗣️ Lời dẫn thuyết trình:**  
> *"Một câu hỏi rất quan trọng trong thực tế là: tại sao tại `t = 17,407` mô hình lại kết luận là bất thường mà không phải một hộp đen chỉ trả ra con số? Để giải đáp điều đó, nhóm chuyển sang **Tab 2 — Giải thích mô hình**.*
>
> *Trên thanh điều khiển đầu trang, luồng xử lý bên trong mạng `ConvAEC` được bóc tách minh bạch thành đúng **5 bước tuần tự từ trái sang phải**: `Đầu vào` $\rightarrow$ `Encoder` $\rightarrow$ `Decoder` $\rightarrow$ `Classifier` $\rightarrow$ `Kết hợp`. Hệ thống đã tự động định vị ngay tại **đỉnh sự cố cao nhất chuỗi (`t = 17,407`)**, đồng thời chọn sẵn một **`Cửa sổ chuẩn` ở vùng máy đang vận hành êm ả ổn định** để đặt song song đối chiếu trực tiếp qua từng bước."*

* **5 nút bước `1. Đầu vào → 2. Encoder → 3. Decoder → 4. Classifier → 5. Kết hợp`:** Tương ứng đúng 5 chặng dữ liệu đi qua kiến trúc chữ **Y** của `ConvAEC` *(nhấn phím mũi tên `←` / `→` để lật bước)*.
* **Nhãn giữa `Mẫu (ALL): machine-1-1` / `Entity: machine-1-1`:** Cho biết thực thể đang được giải thích (khi đẩy một điểm tự nhập từ Tab 3 sang, nhãn này đổi sang nền đỏ `Điểm nạp: t = 100` kèm nút `◀ Chuỗi mẫu` để quay lại).
* **Thanh trượt `t` + 2 nút chọn nhanh `Đỉnh lỗi` & `Cửa sổ chuẩn` *(Mẹo nhớ 1 câu để trả lời hội đồng)*:**
  * **`Đỉnh lỗi`:** Tự động nhảy tới thời điểm có **điểm bất thường cao nhất chuỗi** (`t = 17,407`).
  * **`Cửa sổ chuẩn`:** Tự động lấy **một mốc ở khúc giữa của vùng có điểm bất thường gần bằng `0`** (chọn khúc giữa để tránh mép đầu chuỗi khi máy mới khởi động chưa đủ lịch sử ổn định). *Mẹo nhớ 4 chữ: **"Điểm ≈ 0 · Lấy khúc giữa"**.*

---

### 2.1. Bước 1/5 (`1. Đầu vào`): `Chuỗi thời gian & Input Window (W = 100)`

> **🗣️ Lời dẫn thuyết trình:**  
> *"Bắt đầu tại **Bước 1 — Đầu vào**: biểu đồ trên màn hình đang phóng to khoảng 300 bước thời gian xung quanh sự cố để ta nhìn rõ bối cảnh trước và sau khi máy phát sinh lỗi. Thay vì chỉ nhìn rời rạc một con số đơn lẻ tại thời điểm `t = 17,407`, REDLAMP dùng một khung cửa sổ trượt độ dài **`W = 100` bước**, cắt trọn đoạn tín hiệu liên tiếp từ **`t = 17,308` đến `t = 17,407`** — chính là vùng tô nền hồng có viền đỏ nét đứt trên đồ thị — trên toàn bộ **`38` kênh cảm biến**. Khi nhóm **rê chuột dọc qua đồ thị**, hộp thông tin tự động phân biệt rõ điểm đang trỏ nằm `Ngoài cửa sổ`, `Trong Input Window`, hay chính là mũi nhọn `Đầu cửa sổ (t)` kèm biên độ chính xác; và nhấp đúp chuột vào vị trí bất kỳ sẽ dời khung 100 bước tới ngay mốc thời gian đó."*

* **📖 Hướng dẫn đọc biểu đồ & tương tác rê chuột:**
  * **Trục ngang `Thời gian (t)`:** Phóng to khoảng 300 bước xung quanh mốc đang xét (ví dụ `17,257 → 17,557`) để nhìn rõ bối cảnh sóng trước và sau cửa sổ.
  * **Đường màu xám đen:** Tín hiệu cảm biến gốc; **Khung chữ nhật tô nền hồng, viền đỏ nét đứt `[t - 99, t]`:** Chính là **cửa sổ trượt 100 bước liên tiếp** (`[17308, 17407]`) cắt trên cả `38` kênh cảm biến tạo thành khối đầu vào `38 × 100` đưa vào Bước 2.
  * **🖱️ Khi rê chuột & nhấp đúp trên đồ thị:** Hộp thông tin nổi hiện mốc `t`, giá trị biên độ gốc và gắn nhãn tự động phân biệt 3 vùng: `Ngoài cửa sổ` (xám), `Trong Input Window` (xanh dương), hoặc `Đầu cửa sổ (t)` (đỏ tại đúng mốc kết thúc cửa sổ). **Nhấp đúp chuột** vào bất kỳ điểm nào sẽ dời cửa sổ khảo sát 100 bước về vị trí đó.

---

### 2.2. Bước 2/5 (`2. Encoder`): `Encoder & Bottleneck Embedding (128 chiều)`

#### A. Cột bên trái — `Kiến trúc Encoder (4 khối 1D-CNN)`

> **🗣️ Lời dẫn thuyết trình (Cột trái):**  
> *"Nhấn sang **Bước 2 — Encoder**, màn hình chia làm hai nửa để theo dõi quá trình nén đặc trưng. Nhìn vào sơ đồ 7 tầng từ trên xuống dưới ở **cột bên trái**, nhãn góc phải của mỗi hộp biểu diễn kích thước khối dữ liệu theo chuẩn **`[B, C, T]`**: gồm `1` cửa sổ đang xét, số kênh đặc trưng `C`, và độ dài thời gian `T`.*
>
> *Khối đầu vào **`[1, 38, 100]`** lần lượt chảy qua **4 khối tích chập 1 chiều (`ConvBlock 0` đến `3`)** màu xanh dương, mỗi khối dùng bộ lọc ngắn **`k = 4` bước thời gian** và nhảy bước **`s = 2`**. Các bạn nhìn thanh tiến trình ngang bên dưới mỗi hộp ngắn dần đi một nửa sau mỗi tầng (khi rê chuột vào thanh này sẽ hiện rõ tỷ lệ nén thời gian từ `100/100` xuống `51/100 → 26/100 → 14/100 → 8/100`). Ngược lại, chiều kênh mở rộng từ `38` lên **`128` rồi `256` kênh** — lưu ý đây không phải là cộng thêm kênh vào 38 cảm biến cũ, mà mạng dùng **`128` rồi `256` bộ lọc đa biến hoàn toàn khác nhau** cùng quét qua cả 38 cảm biến để săn tìm các dấu hiệu riêng biệt như gai nhọn, độ dốc tăng gấp hay lệch pha cảm biến. Cuối cùng, tầng `MaxPool1D` màu tím rút gọn 8 mốc thời gian về đúng 1 giá trị cực đại trên mỗi kênh, rồi qua tầng `Conv1x1` gom về nút cổ chai **`Bottleneck Embedding z` gồm đúng `128` con số**."*

1. **📖 Hướng dẫn đọc sơ đồ khối 7 tầng & tương tác rê chuột (từ trên xuống dưới):**
   * **Đường kẻ dọc bên trái & màu các chấm tròn:** Biểu diễn luồng dữ liệu chảy tuần tự qua **7 tầng**:
     * **Chấm xám (`Input Window`):** Dữ liệu thô đầu vào `[1, 38, 100]`.
     * **4 chấm xanh dương (`ConvBlock 0 → 3`):** 4 khối tích chập trích xuất đặc trưng nối tiếp nhau.
     * **Chấm tím (`MaxPool1D`):** Gộp cực đại toàn cục rút trục thời gian về `1`.
     * **Hộp viền & nền xanh đậm cuối cùng (`Conv1x1 → Embedding z`):** Nút cổ chai `Bottleneck` `[1, 128, 1]` trước khi rẽ sang `Decoder` & `Classifier`.
   * **Hàng trên của mỗi hộp:** Bên trái là **tên tầng xử lý**, góc phải là **nhãn kích thước khối dữ liệu `[B, C, T]`** ngay sau khi đi qua tầng đó.
   * **Hàng dưới của mỗi hộp & 🖱️ khi rê chuột vào thanh ngang:** Thanh tiến trình ngang màu xanh co ngắn dần minh họa trực quan **độ dài trục thời gian `T` đang bị nén ngắn dần sau mỗi bước** (`100% → 51% → 26% → 14% → 8% → 1%`). Khi rê chuột lên thanh ngang, hệ thống hiện chú thích `Độ dài trục thời gian: 51/100`, `26/100`, ..., `1/100`.
2. **Ký hiệu `[B, C, T]` ở góc phải tiêu đề là gì?**
   Là 3 chiều của khối dữ liệu (Tensor 3D) đi qua các tầng tích chập 1D trong PyTorch:
   * **`B` (`Batch size`):** Số lượng cửa sổ xử lý trong 1 lượt. Khi soi từng cửa sổ trên Demo thì **`B = 1`**; khi huấn luyện trên GPU thì `B = 128`.
   * **`C` (`Channels`):** Số kênh cảm biến hoặc số kênh đặc trưng ẩn (`38` ở đầu vào $\to$ `128 → 128 → 256 → 256` qua 4 khối CNN $\to$ `128` ở `Embedding z`).
   * **`T` (`Time steps`):** Độ dài trục thời gian của cửa sổ (`100` bước nén dần `→ 51 → 26 → 14 → 8 → 1`).
3. **Tại sao từ `Input Window [1, 38, 100]` qua `ConvBlock 0` lại biến thành `[1, 128, 51]`? (`128` kênh này là gì?)**
   * **Không phải** giữ lại `38` cảm biến gốc rồi cộng thêm `90` kênh mới!
   * Toàn bộ **`128` kênh đều là `128` đặc trưng ẩn hoàn toàn mới (*Feature Maps*)**, được sinh ra bởi **`128` bộ lọc (*filters*) khác nhau** cùng quét song song qua đồng thời cả `38` cảm biến.
   * Mỗi bộ lọc tự học cách dò một kiểu hình thái thời gian hoặc tương quan chéo riêng giữa các cảm biến (ví dụ: bộ lọc dò gai nhọn, bộ lọc dò độ dốc tăng gấp, bộ lọc dò nhiễu rung, bộ lọc dò sự lệch pha khi CPU tăng vọt nhưng Network đứng im...). Kể cả với tập 1 cảm biến như `AIOps` (`[1, 1, 100]`), qua `ConvBlock 0` vẫn mở rộng thành `128` góc nhìn hình thái `[1, 128, 51]`.
4. **Ký hiệu `(k=4, s=2)` và kích thước thực sự của 1 bộ lọc:**
   * **`k = 4` (`kernel_size = 4`):** Chiều dài trượt theo thời gian của **1 bộ lọc** chỉ bằng **`4` bước** *(lưu ý: `128` là số lượng bộ lọc trong danh sách, không phải chiều dài của bộ lọc)*. Cụ thể ở `ConvBlock 0`, mỗi bộ lọc đơn lẻ là một bảng trọng số kích thước **`[38 cảm biến × 4 bước thời gian]`**; sang `ConvBlock 1` và `2` mỗi bộ lọc có kích thước **`[128 × 4]`**; sang `ConvBlock 3` là **`[256 × 4]`**.
   * **`s = 2` (`stride = 2`, kèm đệm `padding = 2`):** Bộ lọc nhảy 2 bước thời gian một lần nên chiều `T` co ngắn lại một nửa sau mỗi tầng theo công thức $\lfloor T_{\text{in}}/2 \rfloor + 1$: `100 → 51 → 26 → 14 → 8`. Trong mỗi `ConvBlock` đều có thêm `BatchNorm1d` (ổn định biên độ), `ReLU` (giữ tín hiệu dương, xóa số âm về `0`) và `Dropout = 0.2` (chỉ bật lúc Train để chống học vẹt, tắt lúc chạy Demo).
   * **Tại sao tăng kênh từ `128` lên `256` ở `ConvBlock 2, 3`?** Khi chiều thời gian `T` bị nén ngắn dần (`51 → 26 → 14 → 8`), mạng mở rộng số kênh lên `256` để ghép các nét ngắn hạn ở tầng nông thành `256` mẫu hành vi phức hợp của toàn cửa sổ ở tầng sâu.
5. **`MaxPool1D (8 → 1)` và `Conv1x1 → Embedding z [1, 128, 1]`:**
   * **`MaxPool1D (8 → 1)` (*Global Max Pooling*):** Quét hết 8 mốc thời gian còn lại trên từng kênh để giữ lại đúng **1 giá trị cực đại** (`[1, 256, 8] → [1, 256, 1]`), giúp mô hình bắt được bất thường bất kể nó xuất hiện ở đầu, giữa hay cuối cửa sổ 100 bước.
   * **`Conv1x1 → Embedding z [1, 128, 1]`:** Nhân ma trận chiếu tuyến tính rút từ `256` kênh về đúng cổ chai **`Bottleneck` gồm `128` con số (`d0 → d127`)**, từ đây rẽ làm 2 nhánh sang Bước 3 (`Decoder`) và Bước 4 (`Classifier`).

#### B. Cột bên phải — `Vector Embedding z (d0 → d127)`

> **🗣️ Lời dẫn thuyết trình (Cột phải):**  
> *"Kết quả của 128 con số đó được vẽ thành 'dấu vân tay' ở **biểu đồ cột bên phải** từ chiều `d0` đến `d127`. Ở phần trên cùng là **2 dải mã vạch màu**: hàng trên của `Cửa sổ chuẩn (Normal)` có màu trắng xám rất nhạt vì lúc bình thường cả 128 chiều đều nằm êm quanh mức `0`; trái lại ngay hàng dưới tại sự cố `t = 17,407`, dải mã vạch rực lên hàng loạt vạch xanh đậm và đỏ đậm. Nhìn xuống biểu đồ cột phân kỳ bên dưới càng thấy rõ: vượt hẳn ra ngoài bóng các cột xám thấp sát vạch `0.0` của trạng thái chuẩn, các chiều đặc trưng tại điểm lỗi bung vọt dữ dội về cả hai phía dương và âm — như `d35 (+13.9)` hay `d96 (-12.4)`. Đặc biệt khi nhóm **rê chuột lướt ngang qua từng cột từ `d0` đến `d127`**, vạch gióng xuyên qua cả hai dải mã vạch và hiện bảng đối chiếu chi tiết tại từng chiều `dk`: giá trị của cửa sổ đang xét mang dấu `Dương (+)` hay `Âm (-)`, giá trị `Baseline Normal` của cửa sổ chuẩn, và độ chênh lệch trực tiếp `z - z_norm` — minh chứng rõ ràng cơ chế đẩy mẫu bất thường tách xa khỏi vùng chuẩn trước khi rẽ sang Bước 3 và Bước 4."*

* **📖 Hướng dẫn đọc biểu đồ cột phải ("Dấu vân tay 128 vạch") & tương tác rê chuột:**
  * **Ý nghĩa dữ liệu vẽ trên biểu đồ:** Toàn bộ cửa sổ `3,800` con số sau khi đi hết cột `Encoder` bên trái đã được tóm gọn thành **1 danh sách gồm đúng `128` con số** $\mathbf{z} = [z_0, z_1, \dots, z_{127}]$ (mang cả dấu dương và âm vì tầng `Conv1x1` cuối không có `ReLU`). Biểu đồ đặt chồng **cửa sổ đang soi (`t=17407`)** lên **cửa sổ bình thường đối chứng (`Normal`)** để so sánh trực tiếp:
  * **Trục ngang (`Latent Dimension d0 → d127`):** Thứ tự từ chiều đặc trưng thứ `0` đến chiều thứ `127`.
  * **Trục dọc bên trái & đường ngang `0.0` ở giữa:** Độ lớn âm/dương của từng chiều đặc trưng.
  * **Phần trên cùng (2 thanh mã vạch màu `Heatmap`):**
    * Thanh trên **`Normal`**: Màu trắng xám rất nhạt vì ở trạng thái bình thường cả 128 chiều đều nằm êm quanh `0`.
    * Thanh dưới **`t=17407`**: Nổi rực các vạch **xanh đậm** (giá trị dương lớn) và **đỏ đậm** (giá trị âm lớn), cho thấy bằng mắt thường "mã vạch" đặc trưng đã biến dạng hoàn toàn so với chuẩn.
  * **Phần dưới (Biểu đồ 128 cột phân kỳ qua vạch `0.0`):**
    * **Các cột XÁM MỜ thấp sát vạch `0.0` (`Chuẩn`):** Biên độ gốc của `Cửa sổ chuẩn` bình thường làm mốc nền.
    * **Các cột XANH DƯƠNG hướng lên trên `0.0` (`Dương +`)** và **cột ĐỎ cắm xuống dưới `0.0` (`Âm -`):** Biên độ thực tế tại cửa sổ đang soi, kèm nhãn tự động chỉ điểm các chiều vọt cực đại (ví dụ `d35 (+13.9)` ở phía trên và `d96 (-12.4)` ở phía dưới).
  * **🖱️ Khi rê chuột lướt ngang qua 128 chiều (`d0 → d127`):** Một vạch sáng dọc chạy xuyên qua cả 2 thanh mã vạch lẫn cột tương ứng, bật hộp thông tin nổi (*Tooltip*) hiển thị 4 thông số của riêng chiều đó:
    1. Thứ tự chiều `Chiều dk / d127` kèm nhãn dấu `Dương (+)` (xanh) hoặc `Âm (-)` (đỏ).
    2. Giá trị chính xác của cửa sổ đang xét (ví dụ `t=17407: +13.9124`).
    3. Giá trị chuẩn đối chứng `Baseline Normal` của cửa sổ bình thường tại cùng chiều `dk`.
    4. `Độ lệch (z - z_norm)`: Hiệu số chênh lệch trực tiếp giữa hai cửa sổ tại đúng chiều `dk`.

---

### 2.3. Bước 3/5 (`3. Decoder`): `Tái tạo qua Decoder, mặt nạ Anomaly Mask & sai số MSE`

> **🗣️ Lời dẫn thuyết trình:**  
> *"Tiếp tục sang **Bước 3 — Nhánh thứ nhất `Decoder` & cơ chế mặt nạ `Anomaly Mask`**: từ vector 128 chiều ở Bước 2, mạng dùng 4 khối tích chập chuyển vị để giải nén ngược lại thành cửa sổ `38 × 100`. Trên màn hình đặt song song 2 biểu đồ để đối chiếu, trong đó đường **xám đen** là tín hiệu cảm biến gốc, đường **xanh dương** là sóng do `Decoder` vẽ lại, còn vùng tô **màu đỏ nhạt** kẹp giữa hai đường chính là độ lệch tái tạo `|x − x̂|`.*
>
> *"Như công thức **`ANOMALY MASK (EQ. 2)`** bên dưới: $L_{\text{MSE}} = \|\text{Mask} \odot (X - \hat{X})\|_2^2$ — khi nhóm **bấm nút `🎭 Anomaly Mask` ở góc trên để BẬT thử mặt nạ**, ta thấy rõ bên trong Mask là một ma trận nhị phân `{0, 1}` kích thước `38 × 100 = 3,800 ô`: các ô bình thường mang giá trị **`1` (màu xanh — giữ lại)**, còn **`1,724` ô vượt chuẩn** mang giá trị **`0` (dải màu vàng & màn sọc chéo trên đồ thị — nhân `× 0` khóa sai số về `0`)**, làm `MSE` sụt ngay từ `0.1230` xuống chỉ còn `0.0009` để `Decoder` lúc huấn luyện tuyệt đối không học vẹt gai nhiễu! Ngược lại khi **bấm TẮT Mask về chế độ `Test`**, toàn bộ `3,800` ô đều mở bằng `1`, sóng tái tạo giữ nguyên quỹ đạo chuẩn làm bùng lên vùng lệch đỏ với **`MSE = 0.1230`** — **cao gấp hơn 220 lần** cửa sổ chuẩn (`0.0005`)! Khi rê chuột trên một biểu đồ, vạch gióng chạy đồng bộ trên cả hai khung trái – phải, đối chiếu từng bước cả tín hiệu `Gốc`, `Tái tạo`, giá trị `Mask (0 / 1)` và `MSE` trước/sau che Mask."*

* **Cơ chế hoạt động & Tương tác Bật/Tắt `Anomaly Mask {0, 1}` (`meta.py` / Eq. 2):**
  * Nhánh `ConvDecoder` nhận vector `128` chiều từ Bước 2, dùng tầng `Linear(1 → 8)` và 4 khối tích chập chuyển vị `ConvTranspose1d` (`8 → 14 → 26 → 51 → 100`) để khôi phục lại cửa sổ kích thước `38 × 100`.
  * **Bên trong `Anomaly Mask` có gì? (Nút bật/tắt `🎭 Anomaly Mask` trên giao diện):**

    $$L_{\text{MSE}} = \|\text{Mask} \odot (X - \hat{X})\|_2^2, \quad \text{Mask} \in \{0, 1\}^{D \times 100}$$

    * **Cấu trúc ma trận:** Trong mã nguồn `RedLamp/models/meta.py` (`inputs * anomaly_mask`, `predicted * anomaly_mask`), `Mask` là ma trận nhị phân cùng kích thước cửa sổ (`38 × 100 = 3,800 ô` trên `SMD`) chỉ gồm số **`1` (Giữ nguyên sai số)** và **`0` (Nhân `× 0` khóa sai số về `0`)**.
    * **Khi BẬT nút `🎭 Anomaly Mask` (Mô phỏng lúc Train):** Thanh barcode `Mask:` phía trên đồ thị đổi sang **vàng (`0`) / xanh (`1`)**, màn sọc vàng phủ lên các bước `Mask = 0` xóa vùng lệch đỏ, 3 thẻ chip dưới chân báo rõ số lượng ô `Mask = 1` vs `Mask = 0` (ví dụ `2,076 ô giữ` / `1,724 ô che`), và `MSE` giảm ngay từ `0.1230` xuống `0.0009`.
    * **Khi TẮT nút `🎭 Anomaly Mask` (Mặc định lúc Test):** Toàn bộ `3,800 / 3,800` ô mở bằng `1`, làm lộ trọn vẹn vùng lệch màu đỏ `|x − x̂|` với `MSE = 0.1230` gấp hơn `220×` cửa sổ chuẩn (`0.0005`).
* **📖 Hướng dẫn đọc 2 biểu đồ song song & tương tác rê chuột đồng bộ:**
  * **Trục ngang (`1 → 100`):** 100 điểm thời gian bên trong cửa sổ (bên trái là `Cửa sổ khảo sát`, bên phải là `Cửa sổ chuẩn đối chứng`).
  * **2 đường & vùng tô màu trên mỗi đồ thị:** Đường **màu xám đen (`Gốc`)** là dữ liệu cảm biến thật; đường **màu xanh dương (`Tái tạo`)** là sóng do `Decoder` vẽ lại; **vùng tô đỏ nhạt (`Độ lệch |x − x̂|`)** thể hiện khoảng cách sai lệch trực quan giữa hai đường; huy hiệu **`MSE = ...`** đóng khung góc trên phải là sai số bình phương trung bình toàn cửa sổ `38 × 100`.
  * **🖱️ Khi rê chuột (Đồng bộ vạch gióng đôi — `Synced Crosshair`):** Chỉ cần rê chuột trên 1 trong 2 biểu đồ, vạch gióng dọc sẽ **tự động chạy đồng bộ trên cả hai biểu đồ trái và phải cùng lúc** ở đúng vị trí `Bước cửa sổ k/100`. Hộp nổi hiển thị đồng thời: giá trị `Gốc`, giá trị `Tái tạo`, `MSE bước này (Khảo sát)` đặt ngay cạnh `MSE bước này (Chuẩn)` để thấy chính xác sai số bùng nổ từ bước thứ mấy trong 100 bước.

---

### 2.4. Bước 4/5 (`4. Classifier`): `Phân loại 12 lớp Pseudo-Anomaly & lọc FAA`

> **🗣️ Lời dẫn thuyết trình:**  
> *"Sang **Bước 4 — Nhánh thứ hai `Classifier` 12 lớp**: nhánh này giúp bắt gọn cả những biến dạng sóng tinh vi mà riêng sai số biên độ có thể bỏ sót. Trên hai biểu đồ cột 12 lớp, các cột **tô màu xám** là lớp `normal` cùng các lớp xuất hiện tràn lan trên 5% chuỗi đã bị bộ lọc **`FAA`** tự động loại trừ để chống báo giả; chỉ các cột **màu đỏ** mới được cộng vào tổng điểm phân loại bất thường `Raw CE`.*
>
> *Ở **`Cửa sổ chuẩn`** bên phải, mô hình khẳng định dứt khoát tới **`99.88%` là lớp `normal`**, tổng xác suất lỗi chỉ `0.0012`. Ngược lại ở **`Cửa sổ sự cố `t = 17,407`**** bên trái, cột `normal` sụt thẳng về **`0.01%`**, đồng thời các cột màu đỏ bật cao chỉ đích danh hình thái biến dạng của sóng: **`flip` (`40.3%`)**, kết hợp **`mixture` (`27.8%`)** và **`scale` (`17.8%`)**, đẩy tổng xác suất bất thường hợp lệ lên tới **`99.99%`**. Khi nhóm **rê chuột vào bất kỳ cột nào trong 12 lớp**, cột tương ứng ở cả hai khung trái – phải sẽ sáng lên đồng bộ, bật hộp chú thích giải thích bằng tiếng Việt kiểu biến dạng sóng đó là gì (hoặc nêu rõ lý do bị `FAA` lọc bỏ) và đặt cạnh nhau xác suất `%` giữa cửa sổ đang xét và cửa sổ chuẩn."*

* **Cơ chế hoạt động:** Nhánh `NonLinear Classifier` đưa cùng vector `128` chiều qua mạng `Classifier (128 → 32 → 12)` + `Softmax` để tính xác suất cửa sổ đang giống lớp nào nhất trong 12 lớp (tổng 12 cột bằng `100%`).
* **📖 Hướng dẫn đọc 2 biểu đồ cột 12 lớp & tương tác rê chuột đồng bộ:**
  * **Trục ngang:** Tên 12 lớp (`normal` + 11 kiểu lỗi giả); **Trục dọc (`0.00 → 1.00`):** Xác suất dự đoán của `Softmax`; **Huy hiệu `Top: ...`** ở góc trên phải nêu tên lớp chiến thắng cao nhất.
  * **Ý nghĩa 2 màu cột:**
    * **Cột màu xám (`Đã lọc FAA: Normal / Mean Prob > 0.05`):** Gồm cột `normal` và bất kỳ lớp lỗi giả nào xuất hiện tràn lan `> 5%` toàn chuỗi kiểm thử — không được cộng vào điểm bất thường.
    * **Cột màu đỏ (`Hợp lệ`):** Các lớp bất thường hợp lệ; cộng chiều cao tất cả các cột màu đỏ lại chính là điểm phân loại thô **`Raw CE`** (ở khung phải `normal = 99.88%` màu xám nên `Raw CE = 0.0012`; ở khung trái các cột đỏ `flip`, `mixture`, `scale`... cộng lại đạt **`Raw CE = 0.9999`**).
  * **🖱️ Khi rê chuột vào từng cột lớp (Đồng bộ làm sáng cột đôi):** Cột cùng tên ở cả hai biểu đồ trái và phải sẽ **sáng đồng bộ cùng lúc**; hộp thông tin nổi hiển thị: tên tiếng Việt của lớp, trạng thái (`Top Class` / `Hợp lệ` / `Đã lọc FAA`), so sánh trực tiếp `% xác suất ở Cửa sổ khảo sát` với `% xác suất ở Cửa sổ chuẩn`, kèm câu giải thích ngắn gọn kiểu biến dạng sóng (hoặc lý do cụ thể bị `FAA` loại trừ).
* **Ý nghĩa 12 nhãn lớp trên trục ngang** *(lưu ý: đây là **giải thích hình thái sóng toán học — morphological explainability**, giúp kỹ sư biết tín hiệu đang biến dạng kiểu gì chứ không khẳng định nguyên nhân hư hỏng vật lý phần cứng)*:
  1. `normal`: Tín hiệu ổn định bình thường.
  2. `spike`: Đột biến gai nhọn cục bộ 1–2 bước thời gian.
  3. `flip`: Đoạn sóng bị đảo ngược chiều thời gian.
  4. `speedup`: Tần số / chu kỳ dao động tăng hoặc giảm đột ngột.
  5. `noise`: Nhiễu trắng ngẫu nhiên cường độ lớn.
  6. `cutoff`: Tín hiệu bị cắt phẳng / bão hòa ở một mức cố định.
  7. `average`: Dao động chi tiết bị san phẳng trơn bất thường.
  8. `scale`: Biên độ đoạn sóng bị phóng đại hoặc thu nhỏ đột ngột.
  9. `wander`: Đường nền (*baseline*) trôi lệch khỏi mức cân bằng.
  10. `contextual`: Giá trị vẫn nằm trong ngưỡng cho phép nhưng lệch pha với ngữ cảnh xung quanh.
  11. `upsidedown`: Đỉnh và đáy sóng bị lật đối xứng ngược chiều biên độ.
  12. `mixture`: Tổ hợp biến dạng / chèn ghép phân đoạn lệch pha phức tạp.

---

### 2.5. Bước 5/5 (`5. Kết hợp`): `Combined Score (Reconstruction + Classification)`

> **🗣️ Lời dẫn thuyết trình:**  
> *"Cuối cùng, tại **Bước 5 — Kết hợp điểm phán quyết**: biểu đồ trên màn hình vẽ chồng cả 3 đường theo thời gian — gồm đường xám đen `Raw MSE` của nhánh tái tạo, đường xanh dương `Raw CE` của nhánh phân loại, và đường **màu đỏ đậm `Raw Combined Score`** là trung bình cộng 50%–50% của hai nhánh. Khi nhóm **rê chuột dọc theo đồ thị**, hộp thông tin hiển thị đồng thời cả 4 con số tại từng thời điểm `t`: `Raw MSE`, `Raw CE`, `Raw Combined Score` và ngưỡng thống kê tự động `Ngưỡng μ + 3σ` kèm nhãn phán quyết tức thì `BẤT THƯỜNG` hay `BÌNH THƯỜNG`.*
>
> *Theo đúng hộp **Công thức** bên dưới, tại mốc `t = 17,407` (chấm tròn đỏ trên đồ thị), phép tính trung bình cho ra `(0.1230 + 0.9999) / 2 = 0.5615`. Khi chấm điểm trực tuyến trên từng cửa sổ đơn lẻ chưa có toàn bộ chuỗi tương lai để chuẩn hóa `MinMax`, hệ thống đối chiếu trực tiếp con số thô này với vạch ngang nét đứt **`Ngưỡng μ + 3σ = 0.4368`** (quy tắc 3-sigma chuẩn `mean + 3·std` trong bài báo REDLAMP), xác nhận vượt ngưỡng và kết luận chính xác **`BẤT THƯỜNG`**."*

* **📖 Hướng dẫn đọc biểu đồ kết hợp & tương tác rê chuột:**
  * **Đường màu xám đen (`Raw MSE`):** Sai số tái tạo thô từ nhánh `Decoder` (Bước 3).
  * **Đường màu xanh dương (`Raw CE`):** Tổng xác suất bất thường hợp lệ từ nhánh `Classifier` (Bước 4).
  * **Đường màu đỏ đậm (`Raw Combined Score`):** Trung bình cộng của đường đen và đường xanh tại từng thời điểm, kèm **chấm tròn** đánh dấu đúng mốc `t = 17407` đang xét và huy hiệu `Combined = 0.561` ở góc phải (màu đỏ khi vượt ngưỡng, màu xanh lá khi dưới ngưỡng).
  * **Đường ngang nét đứt màu đỏ (`Ngưỡng μ+3σ: 0.437`):** Ngưỡng thống kê tự động `mean + 3·std` nhất quán với cách tính chỉ số `RF` trong mã nguồn `evaluate.py` của bài báo gốc — cứ thời điểm nào đường `Raw Combined Score` vượt lên trên vạch nét đứt này thì mô hình kết luận **`BẤT THƯỜNG`**.
  * **🖱️ Khi rê chuột & nhấp đúp dọc đồ thị:** Hộp thông tin nổi hiển thị đầy đủ cả 4 giá trị tại thời điểm `t` đang trỏ (`Raw MSE`, `Raw CE`, `Raw Combined Score`, `Ngưỡng μ + 3σ`) kèm huy hiệu kết luận `⚠️ BẤT THƯỜNG` hay `✓ BÌNH THƯỜNG` *(nhấp đúp chuột tại điểm bất kỳ để cập nhật lại toàn bộ 5 bước giải thích tại thời điểm đó)*.
* **Hộp `CÔNG THỨC`:**

  $$\text{Raw Combined Score} = \frac{\text{Raw MSE} + \text{Raw CE}}{2}$$

  Kết hợp trung bình 50%–50% giữa sai số biên độ của `Decoder` (`Raw MSE`) và tổng xác suất biến dạng hình thái của `Classifier` (`Raw CE`) — giúp mô hình bắt được cả lỗi vọt biên độ lẫn lỗi biến dạng hình dáng sóng tinh vi.
* **Ghi chú phân biệt `MinMax [0, 1]` và ngưỡng tự động `μ + 3σ`:**
  * Khi đánh giá toàn bộ chuỗi kiểm thử ở Tab 1, hai dãy điểm được làm mượt `MA = 50` và chuẩn hóa `MinMax` trên toàn chuỗi về `[0, 1]`.
  * Khi xét **1 cửa sổ đơn lẻ** (ở Bước 5 này và ở Tab 3), không có toàn bộ chuỗi tương lai để chạy `MinMax`, nên hệ thống đối chiếu trực tiếp **`Raw Combined Score`** với ngưỡng thống kê không giám sát **`μ + 3σ`** (`mean + 3·std`, quy tắc 3-sigma chuẩn bài báo gốc: ví dụ `0.5615 > 0.4368` $\Rightarrow$ **`BẤT THƯỜNG`**).

---

## 3. TAB `3. Thử nghiệm dự đoán` (Giao diện tối giản 2 cột từ trái sang phải)

### 3.1. Cột bên trái: `Nhập điểm mới (t = 100)` — Đối chứng `Mẫu chuẩn` (Báo Xanh)

> **🗣️ Lời dẫn thuyết trình:**  
> *"Để khép lại phần trình diễn, nhóm xin chuyển sang **Tab 3 — Thử nghiệm dự đoán** nhằm mô phỏng kịch bản giám sát trực tuyến thời gian thực khi có một gói dữ liệu cảm biến mới vừa truyền về tại thời điểm `t = 100`.*
>
> *Ở cột bên trái, vì mạng `ConvAEC` luôn đánh giá theo cửa sổ trượt 100 bước thời gian, hệ thống tự động ghép nối **99 bước ngữ cảnh chuẩn ổn định (`t = 1..99`)** với bộ **`38` thông số cảm biến (`D1..D38`)** nằm trong miền chuẩn hóa `[0, 1]` tại thời điểm mới nhất **`t = 100`**. Mặc định ngay khi mở Tab (hoặc bấm nút **`Mẫu chuẩn`**), hệ thống nạp sẵn bộ thông số ở trạng thái vận hành bình thường: bảng kết quả bên phải lập tức hiển thị huy hiệu **MÀU XANH — `BÌNH THƯỜNG`**, hình thái dự đoán 12 lớp khẳng định **`Normal (Bình thường)` với độ tự tin `99.9%`**, điểm kết hợp `Combined = 0.0009` nằm rất sâu bên dưới ngưỡng tự động `μ + 3σ = 0.4368`, và đồ thị phía dưới chấm điểm xanh lá bám khít đường tái tạo."*

1. **Tại sao cần ghép 99 bước ngữ cảnh chuẩn (`t = 1..99`) với điểm mới (`t = 100`)?**
   * Một con số rời rạc tại 1 thời điểm không đủ để kết luận bất thường chuỗi thời gian: ví dụ giá trị `1.0` ở kênh vốn luôn chạy tải cao `0.91` (như `D6`) là hoàn toàn bình thường, nhưng giá trị `1.0` ở kênh đang chạy nền thấp `0.02` suốt 99 bước trước đó (như `D1..D5`) lại là một cú vọt đột biến (`Spike`). Ghép với 99 bước từ `Cửa sổ chuẩn` giúp bài thử nghiệm luôn có mốc nền sạch, nhất quán trực tiếp với Tab 2.
2. **Lưới `Thông số cảm biến (D1 .. D38)` chia 5 cột & 2 nút nạp nhanh:**
   * **Hàng đầu tiên là trọn bộ 5 kênh `D1..D5`**; bất kỳ ô nào người dùng gõ sửa khác với nền chuẩn sẽ **tự động tô sáng viền đỏ** kèm nhãn đếm `Đã sửa k kênh`.
   * **Nút `Mẫu chuẩn`:** Khôi phục toàn bộ cảm biến về thông số vận hành bình thường chuẩn và chạy dự đoán ngay.
   * **Nút `⚡ Mẫu đột biến`:** Tự động đẩy các kênh đầu (`D1..D5` trên tập đa biến, hoặc `D1` trên tập đơn biến) lên biên đột biến `1.0` ngay trong miền `[0, 1]` và chạy dự đoán tức thì (cũng có thể tự gõ tay số `1` vào `D1..D5` rồi nhấn `Enter` hoặc bấm `⚡ Dự đoán bất thường`).

---

### 3.2. Cột bên phải: `Kết quả phân tích`, Rê chuột kiểm tra `t = 100` & Đồng bộ ngược về Tab 2

> **🗣️ Lời dẫn thuyết trình (Tiêm đột biến `D1..D5 = 1` & Chốt hạ bài trình bày):**  
> *"Bây giờ, nhóm sẽ thử giả lập một sự cố đột biến ngay trong miền chuẩn hóa `[0, 1]` tại thời điểm `t = 100`. Trong thực tế vận hành máy chủ `SMD`, khi xảy ra nghẽn tải đột ngột, một cụm cảm biến liên quan như CPU, RAM và I/O thường vọt kịch trần cùng lúc — nhóm nhập giá trị cực đại **`1`** vào **5 kênh hàng đầu tiên từ `D1` đến `D5`** (hoặc bấm nhanh nút **`⚡ Mẫu đột biến`**) rồi chạy dự đoán.*
>
> *Các bạn có thể thấy hệ thống phản ứng tức thì: 5 ô `D1..D5` sáng đỏ bên trái, đồng thời bảng kết quả bên phải chuyển ngay sang **MÀU ĐỎ — `BẤT THƯỜNG`** với điểm kết hợp `Combined = 0.5008` vượt qua ngưỡng tự động `μ + 3σ = 0.4368`; hộp hình thái 12 lớp gọi đúng tên sự cố vừa nhập là **`Spike (Đột biến đỉnh nhọn)` với độ tự tin `99.9%`**. Đặc biệt, nhìn vào 4 thẻ điểm số bên dưới, ta thấy rõ ưu thế cốt lõi của kiến trúc 2 nhánh REDLAMP: vì cú giật chỉ diễn ra ở 1 điểm cuối trên cửa sổ 3,800 điểm nên sai số tái tạo trung bình `MSE` mới chỉ nhích nhẹ lên `0.0017`, nếu dùng Autoencoder đơn thuần rất dễ bỏ lọt; nhưng **nhánh `Classifier` đã bắt gọn gai nhọn với `CE = 1.0000`**, kéo điểm kết hợp vượt ngưỡng để báo động chính xác! Ở biểu đồ phía dưới, chế độ `Tự động (kênh nổi bật)` cũng ưu tiên kênh có sóng dao động thực tế (như **`D2`** đang dao động `0.03–0.19` suốt `t = 1..99` rồi vọt dựng đứng lên `1.00` tại `t = 100`) — khi **rê chuột vào điểm đỏ cuối cùng**, hộp thông tin đối chiếu trực tiếp tín hiệu gốc, giá trị tái tạo và sai số bình phương `(x - x̂)²`.*
>
> *Và điểm khép kín trọn vẹn nhất của ứng dụng là khi nhóm bấm nút **`🔍 Giải thích điểm này ở Tab 2 (5 bước)`** ở góc phải: toàn bộ cửa sổ 100 bước chứa điểm đột biến vừa tự tay nhập được chuyển ngược sang Tab 2, cho phép kỹ sư kiểm tra minh bạch từ đầu đến cuối cả 5 bước xử lý của mạng. Đó cũng là toàn bộ quy trình phát hiện bất thường bền vững và có khả năng giải thích của mô hình REDLAMP. Nhóm xin chân thành cảm ơn thầy và các bạn đã chú ý lắng nghe, và rất mong nhận được các câu hỏi góp ý ạ!"*

1. **Hộp `KẾT LUẬN` & `HÌNH THÁI DỰ ĐOÁN (12 LỚP)`:**
   * Đặt song song phán quyết nhị phân **`BÌNH THƯỜNG`** (xanh lá, khi `Combined <= Ngưỡng μ+3σ`) hoặc **`BẤT THƯỜNG`** (đỏ, khi `Combined > Ngưỡng μ+3σ`) với nhãn hình thái có xác suất cao nhất trong 12 lớp (ví dụ `Normal 99.9%` hoặc `Spike 99.9%`) kèm mô tả đặc trưng sóng và cơ sở kết luận.
2. **4 thẻ chỉ số công thức trực diện (`Nhánh Decoder (MSE)`, `Nhánh Classifier (CE)`, `Combined = (MSE + CE) / 2`, `Ngưỡng tự động (μ + 3σ)`):**
   * Trình bày minh bạch đóng góp của từng nhánh vào phép trung bình cộng `(MSE + CE) / 2` đối chiếu trực tiếp với ngưỡng thống kê tự động `μ + 3σ` — không cần thanh kéo ngưỡng thủ công gây mâu thuẫn với tính chất "không giám sát".
   * **💡 Điểm nhấn vàng nếu hội đồng hỏi *"Tại sao `D1..D5 = 1` có `MSE = 0.0017` rất nhỏ mà vẫn báo Bất thường?"*:** Do đột biến chỉ xuất hiện tại đúng 1 bước thời gian `t = 100` trên 5 kênh (`5 / 3,800` giá trị trong cửa sổ) và Autoencoder không che mặt nạ lúc kiểm thử có xu hướng chép lại một phần gai nhọn (*over-reconstruction*), nên `MSE` trung bình toàn cửa sổ rất nhỏ (`0.0017`). Nhờ REDLAMP có thêm nhánh **`Classifier` nhận diện hình thái `Spike` với `CE = 1.0000`**, điểm kết hợp `(0.0017 + 1.0000) / 2 = 0.5008` vượt ngưỡng `0.4368` — chứng minh thực nghiệm trực tiếp lý do bài báo KDD '25 kết hợp 2 nhánh thay vì chỉ dùng Autoencoder truyền thống!
   * **💡 Nếu hội đồng hỏi *"Dữ liệu đầu vào nằm trong `[0, 1]`, vậy cứ nhập `= 1` ở bất kỳ kênh nào cũng phải là bất thường chứ?"*:** Phép chuẩn hóa `Min-Max [0, 1]` được khớp trên **tập huấn luyện bình thường** (`train`), nghĩa là mức `1.0` chính là mức đỉnh hoàn toàn hợp lệ từng xuất hiện lúc máy chạy bình thường. Với kênh vốn đang chạy nền cao sát `1.0` (như `D6 ≈ 0.908`), lên `1.0` là dao động bình thường; trái lại với các kênh đang chạy nền thấp `0.00–0.09` suốt 99 bước trước đó (như `D1..D5`), vọt dựng đứng lên `1.0` tại `t = 100` mới là **đột biến theo ngữ cảnh thời gian (*Contextual Spike*)**.
3. **Biểu đồ `Tái tạo cửa sổ 100 điểm & điểm mới (t = 100)`, bộ chọn kênh & 🖱️ tương tác rê chuột:**
   * Tự động ưu tiên hiển thị kênh cảm biến có biến động mạnh nhất tại `t = 100` (kèm hộp chọn góc phải cho phép đổi xem riêng bất kỳ kênh `D1..D38` nào nếu thầy yêu cầu kiểm tra), đổi màu chấm đánh dấu `Điểm mới (t=100)` sang **xanh lá** khi bình thường và **đỏ** khi bất thường.
   * **🖱️ Khi rê chuột dọc 100 điểm trên biểu đồ:** Hộp thông tin nổi phân biệt rõ các bước `Ngữ cảnh chuẩn` (`t = 1..99`) với `Điểm nạp mới (t = 100)`, hiển thị song song `Tín hiệu gốc`, `Tái tạo Decoder`, và **sai số bình phương từng bước `(x - x̂)²`**.
4. **Nút `🔍 Giải thích điểm này ở Tab 2 (5 bước)` (ngay góc phải tiêu đề `Kết quả phân tích`):**
   * Chuyển nguyên cửa sổ 100 bước vừa dự đoán quay ngược về **Tab `2. Giải thích mô hình`**, mở đầy đủ cả 5 bước (`Đầu vào → Encoder → Decoder → Classifier 12 lớp & lọc FAA → Combined Score`) cho chính điểm dữ liệu vừa tự tay nhập vào.
