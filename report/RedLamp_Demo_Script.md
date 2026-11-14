# 🎤 KỊCH BẢN LỜI DẪN THUYẾT TRÌNH CHI TIẾT — LIVE WEB DEMO REDLAMP

* **Dành cho người thuyết trình trực tiếp trên sân khấu** *(văn phong nói tự nhiên, dẫn dắt chi tiết từng vùng giao diện + hiệu ứng rê chuột tương tác trực tiếp trên biểu đồ, thời lượng ~8–10 phút)*.
* **Tra cứu giải thích khái niệm & phản biện hội đồng:** [`RedLamp_Demo_Concepts.md`](./RedLamp_Demo_Concepts.md) | **Tính tay từng bước:** [`RedLamp_ConvAEC_StepByStep.md`](./RedLamp_ConvAEC_StepByStep.md)

---

## PHẦN 1 — TAB `1. Tổng quan` (~3 phút)

### 1.1. Khối trên cùng: `Bộ chọn Dataset & Entity`
* **🖱️ Thao tác:** Mở sẵn Tab `1. Tổng quan` ở chế độ mặc định `SMD` · `ALL` · `Seed 0`. Rê chuột dọc theo 3 ô chọn trên cùng.

> *"Kính thưa thầy và các bạn, tiếp nối phần lý thuyết, em xin phép trình bày hệ thống **Web Demo trực quan hóa mô hình REDLAMP**. Toàn bộ trọng số và kết quả trên hệ thống này đều do nhóm trực tiếp huấn luyện tái lập trên **5 bộ dữ liệu chuẩn quốc tế** đa lĩnh vực: từ giám sát máy chủ công nghiệp `SMD`, chỉ số KPI đám mây `AIOps`, dữ liệu tàu tự hành Sao Hỏa `MSL` và vệ tinh `SMAP` của NASA, cho tới kho dữ liệu đa ngành `UCR`.*

Hệ thống tổ chức dữ liệu theo 3 cấp từ lớn đến nhỏ: **Tập dữ liệu (`Dataset`)**, **Thực thể (`Entity`)** và **Kênh cảm biến (`Channel`)**. Mặc định trên màn hình đang là tập **`SMD`**, bao gồm **28 thực thể máy chủ độc lập**, trong đó mỗi máy chủ được gắn **38 kênh cảm biến** theo dõi đồng thời CPU, bộ nhớ RAM, ổ cứng và lưu lượng mạng. Khi nhóm để chế độ **`Entity = ALL`**, hệ thống sẽ tổng hợp bức tranh toàn cảnh trên cả 28 máy chủ; đồng thời ô **`Seed huấn luyện (0 – 4)`** bên phải cho phép kiểm chứng độ ổn định của mô hình qua **5 lần huấn luyện độc lập** với các vị trí chèn lỗi giả và khởi tạo trọng số ngẫu nhiên khác nhau."*

---

### 1.2. Khối giữa: `Kết quả đánh giá chất lượng (Metrics)`
* **🖱️ Thao tác:** Chỉ xuống khung ngang `Kết quả đánh giá chất lượng (Metrics)` ở giữa màn hình (`VUS-ROC`, `VUS-PR`, `RF`).

> *"Nhìn xuống khung **Kết quả đánh giá chất lượng** ở ngay bên dưới: theo đúng chuẩn nghiêm ngặt của hội nghị KDD '25, nhóm không sử dụng độ chính xác `Accuracy` hay chỉ số `F1-PA` truyền thống. Lý do là trong chuỗi thời gian thực tế, hơn 96% thời lượng máy chạy bình thường nên đoán toàn bộ là bình thường cũng đạt 96% `Accuracy`; còn `F1-PA` có nhược điểm là chỉ cần chạm trúng 1 giây trong cả đợt sự cố dài hàng trăm giây là tự tính đúng cả đoạn, tạo ra điểm số cao giả tạo.*
>
> *Thay vào đó, hệ thống đánh giá bằng bộ 3 thước đo theo phân đoạn liên tục là **`VUS-ROC`**, **`VUS-PR`** và **`RF (Range F-score)`** — xét đồng thời mọi ngưỡng cắt và vùng đệm dung sai quanh rìa sự cố. Trong ba chỉ số này, **`VUS-PR` là thước đo chính khắt khe nhất**: các bạn thấy `VUS-ROC` đạt rất cao (`0.82`), nhưng `VUS-PR` chỉ khoảng `0.37` — không phải vì mô hình yếu, mà vì sự cố rất hiếm nên chỉ cần có một ít báo động giả là độ chuẩn xác `Precision` bị phạt sụt ngay lập tức. Trên toàn bộ 28 máy của tập `SMD`, kết quả tái lập trung bình qua 5 seed của nhóm đạt **`VUS-PR = 0.374 ± 0.013`**, bám rất sát và nhỉnh hơn nhẹ so với mức `0.366` công bố trong bài báo gốc."*

---

### 1.3. Khối dưới cùng: `Biểu đồ tổng quan` (`ALL` $\rightarrow$ `machine-1-1`)
* **🖱️ Thao tác:** Rê chuột qua vài cột trên biểu đồ `ALL` để hiện hộp thông tin nổi (*Tooltip*), sau đó **nhấp đúp vào cột đầu tiên `m1-1`** để mở chuỗi thời gian của `machine-1-1` và rê chuột dọc theo các đỉnh trên đồ thị.

> *"Nhìn xuống biểu đồ lớn ở dưới cùng: khi đang ở chế độ `ALL`, biểu đồ cột đối chiếu điểm `VUS-PR` của từng máy trong 28 máy chủ với đường trung bình toàn tập màu đỏ nét đứt. Khi nhóm **rê chuột lên từng cột**, bảng chú thích nổi sẽ báo ngay máy đó đứng trên hay dưới mức trung bình tập, hiển thị đồng thời điểm của Seed đang chọn, trung bình 5 seed và trọn bộ cả 3 chỉ số `ROC · PR · RF`. Bây giờ, nhóm sẽ **nhấp đúp trực tiếp vào cột đầu tiên `m1-1`** để đi sâu vào chuỗi thời gian thực tế dài **`28,381` bước** của máy chủ `machine-1-1`.*
>
> *Trên đồ thị vừa mở ra, đường **màu xám đen** biểu diễn `Độ lệch Max` trên cả 38 cảm biến để ta quan sát biến động tổng thể của máy trên một đường duy nhất (góc trên phải vẫn cho phép chọn xem riêng từng kênh `D1..D38`), còn đường **màu đỏ** là điểm bất thường đầu ra của REDLAMP. Ở đoạn đầu từ `0` đến `5,500`, mặc dù tín hiệu máy chủ nhấp nhô lên `0.85` theo chu kỳ tải công việc bình thường, mô hình hiểu đúng quy luật nên **đường màu đỏ nằm phẳng lặng sát mức `0.00`**, không hề báo giả. Ngược lại, tại cả **8 đợt sự cố thật**, đường màu đỏ đều lập tức bật cao vượt `0.50` — chạm đỉnh **`1.00`** tại sự cố lớn nhất **`t = 17,407`**. Đặc biệt khi **rê chuột dọc theo đồ thị**, vạch gióng sẽ chỉ đích danh ngay thời điểm `t` đó đang là `Bình thường` hay `Bất thường (> μ + 3σ)`, cảm biến nào trong 38 kênh đang biến động mạnh nhất, kèm tỷ lệ điểm thô `Raw Combined / μ + 3σ`; và nhấp đúp vào bất kỳ đỉnh nào cũng sẽ nhảy thẳng sang Tab 2 để mổ xẻ thời điểm đó."*

---

## PHẦN 2 — TAB `2. Giải thích mô hình` (~4 phút)

### 2.0. Mở đầu Tab 2 & Thanh điều khiển trên cùng
* **🖱️ Thao tác:** Bấm chuyển sang Tab **`2. Giải thích mô hình`** trên thanh Header. Chỉ qua 5 nút bước và thanh trượt thời gian.

> *"Một câu hỏi rất quan trọng trong thực tế là: tại sao tại `t = 17,407` mô hình lại kết luận là bất thường mà không phải một hộp đen chỉ trả ra con số? Để giải đáp điều đó, nhóm chuyển sang **Tab 2 — Giải thích mô hình**.*
>
> *Trên thanh điều khiển đầu trang, luồng xử lý bên trong mạng `ConvAEC` được bóc tách minh bạch thành đúng **5 bước tuần tự từ trái sang phải**: `Đầu vào` $\rightarrow$ `Encoder` $\rightarrow$ `Decoder` $\rightarrow$ `Classifier` $\rightarrow$ `Kết hợp`. Hệ thống đã tự động định vị ngay tại **đỉnh sự cố cao nhất chuỗi (`t = 17,407`)**, đồng thời chọn sẵn một **`Cửa sổ chuẩn` ở vùng máy đang vận hành êm ả ổn định** để đặt song song đối chiếu trực tiếp qua từng bước."*

---

### 2.1. Bước 1/5 (`1. Đầu vào`)
* **🖱️ Thao tác:** Đang ở `1. Đầu vào`, rê chuột từ vùng ngoài vào bên trong khung chữ nhật màu hồng viền đỏ nét đứt `[17308, 17407]`.

> *"Bắt đầu tại **Bước 1 — Đầu vào**: biểu đồ trên màn hình đang phóng to khoảng 300 bước thời gian xung quanh sự cố để ta nhìn rõ bối cảnh trước và sau khi máy phát sinh lỗi. Thay vì chỉ nhìn rời rạc một con số đơn lẻ tại thời điểm `t = 17,407`, REDLAMP dùng một khung cửa sổ trượt độ dài **`W = 100` bước**, cắt trọn đoạn tín hiệu liên tiếp từ **`t = 17,308` đến `t = 17,407`** — chính là vùng tô nền hồng có viền đỏ nét đứt trên đồ thị — trên toàn bộ **`38` kênh cảm biến**. Khi nhóm **rê chuột dọc qua đồ thị**, hộp thông tin tự động phân biệt rõ điểm đang trỏ nằm `Ngoài cửa sổ`, `Trong Input Window`, hay chính là mũi nhọn `Đầu cửa sổ (t)` kèm biên độ chính xác; và nhấp đúp chuột vào vị trí bất kỳ sẽ dời khung 100 bước tới ngay mốc thời gian đó."*

---

### 2.2. Bước 2/5 (`2. Encoder`)
* **🖱️ Thao tác:** Nhấn phím **`→`** sang `2. Encoder`. Rê chuột dọc sơ đồ 7 tầng bên trái trước, sau đó rê chuột lướt ngang qua biểu đồ `Vector Embedding z (d0 → d127)` bên phải để hiện hộp so sánh từng chiều.

> **Cột trái (`Kiến trúc Encoder`):**  
> *"Nhấn sang **Bước 2 — Encoder**, màn hình chia làm hai nửa để theo dõi quá trình nén đặc trưng. Nhìn vào sơ đồ 7 tầng từ trên xuống dưới ở **cột bên trái**, nhãn góc phải của mỗi hộp biểu diễn kích thước khối dữ liệu theo chuẩn **`[B, C, T]`**: gồm `1` cửa sổ đang xét, số kênh đặc trưng `C`, và độ dài thời gian `T`.*
>
> *Khối đầu vào **`[1, 38, 100]`** lần lượt chảy qua **4 khối tích chập 1 chiều (`ConvBlock 0` đến `3`)** màu xanh dương, mỗi khối dùng bộ lọc ngắn **`k = 4` bước thời gian** và nhảy bước **`s = 2`**. Các bạn nhìn thanh tiến trình ngang bên dưới mỗi hộp ngắn dần đi một nửa sau mỗi tầng (khi rê chuột vào thanh này sẽ hiện rõ tỷ lệ nén thời gian từ `100/100` xuống `51/100 → 26/100 → 14/100 → 8/100`). Ngược lại, chiều kênh mở rộng từ `38` lên **`128` rồi `256` kênh** — lưu ý đây không phải là cộng thêm kênh vào 38 cảm biến cũ, mà mạng dùng **`128` rồi `256` bộ lọc đa biến hoàn toàn khác nhau** cùng quét qua cả 38 cảm biến để săn tìm các dấu hiệu riêng biệt như gai nhọn, độ dốc tăng gấp hay lệch pha cảm biến. Cuối cùng, tầng `MaxPool1D` màu tím rút gọn 8 mốc thời gian về đúng 1 giá trị cực đại trên mỗi kênh, rồi qua tầng `Conv1x1` gom về nút cổ chai **`Bottleneck Embedding z` gồm đúng `128` con số**."*
>
> **Cột phải (`Vector Embedding z`):**  
> *"Kết quả của 128 con số đó được vẽ thành 'dấu vân tay' ở **biểu đồ cột bên phải** từ chiều `d0` đến `d127`. Ở phần trên cùng là **2 dải mã vạch màu**: hàng trên của `Cửa sổ chuẩn (Normal)` có màu trắng xám rất nhạt vì lúc bình thường cả 128 chiều đều nằm êm quanh mức `0`; trái lại ngay hàng dưới tại sự cố `t = 17,407`, dải mã vạch rực lên hàng loạt vạch xanh đậm và đỏ đậm. Nhìn xuống biểu đồ cột phân kỳ bên dưới càng thấy rõ: vượt hẳn ra ngoài bóng các cột xám thấp sát vạch `0.0` của trạng thái chuẩn, các chiều đặc trưng tại điểm lỗi bung vọt dữ dội về cả hai phía dương và âm — như `d35 (+13.9)` hay `d96 (-12.4)`. Đặc biệt khi nhóm **rê chuột lướt ngang qua từng cột từ `d0` đến `d127`**, vạch gióng xuyên qua cả hai dải mã vạch và hiện bảng đối chiếu chi tiết tại từng chiều `dk`: giá trị của cửa sổ đang xét mang dấu `Dương (+)` hay `Âm (-)`, giá trị `Baseline Normal` của cửa sổ chuẩn, và độ chênh lệch trực tiếp `z - z_norm` — minh chứng rõ ràng cơ chế đẩy mẫu bất thường tách xa khỏi vùng chuẩn trước khi rẽ sang Bước 3 và Bước 4."*

---

### 2.3. Bước 3/5 (`3. Decoder` & Nút bật/tắt `🎭 Anomaly Mask`)
* **🖱️ Thao tác:** Nhấn phím **`→`** sang `3. Decoder`. Bấm nút **`🎭 Anomaly Mask`** ở thanh trên để **BẬT rồi TẮT thử mặt nạ `{0, 1}`** ngay trên đồ thị, sau đó rê chuột trên đồ thị bên trái để thấy **vạch gióng chạy đồng bộ trên cả 2 khung trái – phải**.

> *"Tiếp tục sang **Bước 3 — Nhánh thứ nhất `Decoder` & cơ chế mặt nạ `Anomaly Mask`**: từ vector 128 chiều ở Bước 2, mạng dùng 4 khối tích chập chuyển vị để giải nén ngược lại thành cửa sổ `38 × 100`. Trên màn hình đặt song song 2 biểu đồ để đối chiếu, trong đó đường **xám đen** là tín hiệu cảm biến gốc, đường **xanh dương** là sóng do `Decoder` vẽ lại, còn vùng tô **màu đỏ nhạt** kẹp giữa hai đường chính là độ lệch tái tạo `|x − x̂|`.*
>
> *Khi nhóm **bấm nút `🎭 Anomaly Mask` ở phía trên để BẬT thử mặt nạ**, ta thấy rõ bên trong Mask là một ma trận nhị phân `{0, 1}` kích thước `38 × 100 = 3,800 ô` theo công thức $L_{\text{MSE}} = \|\text{Mask} \odot (X - \hat{X})\|_2^2$: các ô bình thường mang giá trị **`1` (màu xanh — giữ lại)**, còn **`1,724` ô vượt chuẩn** bị gán **`0` (màn sọc vàng trên đồ thị — nhân `× 0` khóa sai số về `0`)**, kéo `MSE` sụt ngay từ `0.1230` xuống `0.0009` để `Decoder` lúc huấn luyện không bao giờ học vẹt gai nhiễu! Và khi **bấm TẮT Mask về chế độ `Test`**, toàn bộ `3,800` ô mở bằng `1`, sóng tái tạo giữ nguyên quỹ đạo chuẩn làm lộ vùng lệch đỏ với **`MSE = 0.1230`** — **cao gấp hơn 220 lần** cửa sổ chuẩn (`0.0005`)! Khi rê chuột trên một biểu đồ, vạch gióng chạy đồng bộ trên cả hai khung trái – phải, đối chiếu từng bước cả tín hiệu `Gốc`, `Tái tạo`, giá trị `Mask (0 / 1)` và `MSE`."*

---

### 2.4. Bước 4/5 (`4. Classifier`)
* **🖱️ Thao tác:** Nhấn phím **`→`** sang `4. Classifier`. Rê chuột qua cột `normal` rồi qua các cột `flip`, `mixture`, `scale` để thấy **2 biểu đồ cột tự động sáng đồng bộ** và hiện mô tả từng lớp lỗi.

> *"Sang **Bước 4 — Nhánh thứ hai `Classifier` 12 lớp**: nhánh này giúp bắt gọn cả những biến dạng sóng tinh vi mà riêng sai số biên độ có thể bỏ sót. Trên hai biểu đồ cột 12 lớp, các cột **tô màu xám** là lớp `normal` cùng các lớp xuất hiện tràn lan trên 5% chuỗi đã bị bộ lọc **`FAA`** tự động loại trừ để chống báo giả; chỉ các cột **màu đỏ** mới được cộng vào tổng điểm phân loại bất thường `Raw CE`.*
>
> *Ở **`Cửa sổ chuẩn`** bên phải, mô hình khẳng định dứt khoát tới **`99.88%` là lớp `normal`**, tổng xác suất lỗi chỉ `0.0012`. Ngược lại ở **`Cửa sổ sự cố `t = 17,407`**** bên trái, cột `normal` sụt thẳng về **`0.01%`**, đồng thời các cột màu đỏ bật cao chỉ đích danh hình thái biến dạng của sóng: **`flip` (`40.3%`)**, kết hợp **`mixture` (`27.8%`)** và **`scale` (`17.8%`)**, đẩy tổng xác suất bất thường hợp lệ lên tới **`99.99%`**. Khi nhóm **rê chuột vào bất kỳ cột nào trong 12 lớp**, cột tương ứng ở cả hai khung trái – phải sẽ sáng lên đồng bộ, bật hộp chú thích giải thích bằng tiếng Việt kiểu biến dạng sóng đó là gì (hoặc nêu rõ lý do bị `FAA` lọc bỏ) và đặt cạnh nhau xác suất `%` giữa cửa sổ đang xét và cửa sổ chuẩn."*

---

### 2.5. Bước 5/5 (`5. Kết hợp`)
* **🖱️ Thao tác:** Nhấn phím **`→`** sang `5. Kết hợp`. Rê chuột dọc qua vùng đỉnh chấm tròn đỏ `t = 17,407` trên đồ thị rồi chỉ xuống hộp `CÔNG THỨC`.

> *"Cuối cùng, tại **Bước 5 — Kết hợp điểm phán quyết**: biểu đồ trên màn hình vẽ chồng cả 3 đường theo thời gian — gồm đường xám đen `Raw MSE` của nhánh tái tạo, đường xanh dương `Raw CE` của nhánh phân loại, và đường **màu đỏ đậm `Raw Combined Score`** là trung bình cộng 50%–50% của hai nhánh. Khi nhóm **rê chuột dọc theo đồ thị**, hộp thông tin hiển thị đồng thời cả 4 con số tại từng thời điểm `t`: `Raw MSE`, `Raw CE`, `Raw Combined Score` và ngưỡng thống kê tự động `Ngưỡng μ + 3σ` kèm nhãn phán quyết tức thì `BẤT THƯỜNG` hay `BÌNH THƯỜNG`.*
>
> *Theo đúng hộp **Công thức** bên dưới, tại mốc `t = 17,407` (chấm tròn đỏ trên đồ thị), phép tính trung bình cho ra `(0.1230 + 0.9999) / 2 = 0.5615`. Khi chấm điểm trực tuyến trên từng cửa sổ đơn lẻ chưa có toàn bộ chuỗi tương lai để chuẩn hóa `MinMax`, hệ thống đối chiếu trực tiếp con số thô này với vạch ngang nét đứt **`Ngưỡng μ + 3σ = 0.4368`** (quy tắc 3-sigma chuẩn `mean + 3·std` trong bài báo REDLAMP), xác nhận vượt ngưỡng và kết luận chính xác **`BẤT THƯỜNG`**."*

---

## PHẦN 3 — TAB `3. Thử nghiệm dự đoán` & Chốt Bài (~2 phút)

### 3.1. Đối chứng `Mẫu chuẩn` mặc định (Báo Xanh ngay khi mở Tab)
* **🖱️ Thao tác:** Bấm chuyển sang Tab **`3. Thử nghiệm dự đoán`**, chỉ vào lưới cảm biến `D1..D38` bên trái (đang nạp sẵn `Mẫu chuẩn`) và bảng kết quả màu xanh bên phải.

> *"Để khép lại phần trình diễn, nhóm xin chuyển sang **Tab 3 — Thử nghiệm dự đoán** nhằm mô phỏng kịch bản giám sát trực tuyến thời gian thực khi có một gói dữ liệu cảm biến mới vừa truyền về tại thời điểm `t = 100`.*
>
> *Ở cột bên trái, vì mạng `ConvAEC` luôn đánh giá theo cửa sổ trượt 100 bước thời gian, hệ thống tự động ghép nối **99 bước ngữ cảnh chuẩn ổn định (`t = 1..99`)** với bộ **`38` thông số cảm biến (`D1..D38`)** nằm trong miền chuẩn hóa `[0, 1]` tại thời điểm mới nhất **`t = 100`**. Mặc định ngay khi mở Tab (hoặc bấm nút **`Mẫu chuẩn`**), hệ thống nạp sẵn bộ thông số ở trạng thái vận hành bình thường: bảng kết quả bên phải lập tức hiển thị huy hiệu **MÀU XANH — `BÌNH THƯỜNG`**, hình thái dự đoán 12 lớp khẳng định **`Normal (Bình thường)` với độ tự tin `99.9%`**, điểm kết hợp `Combined = 0.0009` nằm rất sâu bên dưới ngưỡng tự động `μ + 3σ = 0.4368`, và đồ thị phía dưới chấm điểm xanh lá bám khít đường tái tạo."*

---

### 3.2. Tiêm đột biến `D1..D5 = 1` ngay trong miền `[0, 1]` (Báo Đỏ), Rê chuột kiểm tra `t = 100` & Đồng bộ ngược về Tab 2
* **🖱️ Thao tác:** Bấm nút **`⚡ Mẫu đột biến`** *(hoặc tự gõ số **`1`** vào hàng đầu tiên **`D1..D5`** rồi bấm **`⚡ Dự đoán bất thường`**)* $\rightarrow$ rê chuột tới điểm đỏ cuối `t = 100` trên biểu đồ tái tạo bên dưới $\rightarrow$ bấm nút **`🔍 Giải thích điểm này ở Tab 2 (5 bước)`** ở góc trên phải.

> *"Bây giờ, nhóm sẽ thử giả lập một sự cố đột biến ngay trong miền chuẩn hóa `[0, 1]` tại thời điểm `t = 100`. Trong thực tế vận hành máy chủ `SMD`, khi xảy ra nghẽn tải đột ngột, một cụm cảm biến liên quan như CPU, RAM và I/O thường vọt kịch trần cùng lúc — nhóm nhập giá trị cực đại **`1`** vào **5 kênh hàng đầu tiên từ `D1` đến `D5`** (hoặc bấm nhanh nút **`⚡ Mẫu đột biến`**) rồi chạy dự đoán.*
>
> *Các bạn có thể thấy hệ thống phản ứng tức thì: 5 ô `D1..D5` sáng đỏ bên trái, đồng thời bảng kết quả bên phải chuyển ngay sang **MÀU ĐỎ — `BẤT THƯỜNG`** với điểm kết hợp `Combined = 0.5008` vượt qua ngưỡng tự động `μ + 3σ = 0.4368`; hộp hình thái 12 lớp gọi đúng tên sự cố vừa nhập là **`Spike (Đột biến đỉnh nhọn)` với độ tự tin `99.9%`**. Đặc biệt, nhìn vào 4 thẻ điểm số bên dưới, ta thấy rõ ưu thế cốt lõi của kiến trúc 2 nhánh REDLAMP: vì cú giật chỉ diễn ra ở 1 điểm cuối trên cửa sổ 3,800 điểm nên sai số tái tạo trung bình `MSE` mới chỉ nhích nhẹ lên `0.0017`, nếu dùng Autoencoder đơn thuần rất dễ bỏ lọt; nhưng **nhánh `Classifier` đã bắt gọn gai nhọn với `CE = 1.0000`**, kéo điểm kết hợp vượt ngưỡng để báo động chính xác! Ở biểu đồ phía dưới, chế độ `Tự động (kênh nổi bật)` cũng ưu tiên kênh có sóng dao động thực tế (như **`D2`** đang dao động `0.03–0.19` suốt `t = 1..99` rồi vọt dựng đứng lên `1.00` tại `t = 100`) — khi **rê chuột vào điểm đỏ cuối cùng**, hộp thông tin đối chiếu trực tiếp tín hiệu gốc, giá trị tái tạo và sai số bình phương `(x - x̂)²`.*
>
> *Và điểm khép kín trọn vẹn nhất của ứng dụng là khi nhóm bấm nút **`🔍 Giải thích điểm này ở Tab 2 (5 bước)`** ở góc phải: toàn bộ cửa sổ 100 bước chứa điểm đột biến vừa tự tay nhập được chuyển ngược sang Tab 2, cho phép kỹ sư kiểm tra minh bạch từ đầu đến cuối cả 5 bước xử lý của mạng. Đó cũng là toàn bộ quy trình phát hiện bất thường bền vững và có khả năng giải thích của mô hình REDLAMP. Nhóm xin chân thành cảm ơn thầy và các bạn đã chú ý lắng nghe, và rất mong nhận được các câu hỏi góp ý ạ!"*
