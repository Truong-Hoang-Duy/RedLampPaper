# 🎤 KỊCH BẢN THUYẾT TRÌNH — KẾT QUẢ THỰC NGHIỆM & DEMO REDLAMP

* **Dành cho người thuyết trình** *(lời nói thuần nội dung chuyên môn, tự nhiên, ngắn gọn, không đọc các câu chỉ trỏ thao tác giao diện)*.
* **Tra cứu giải thích khái niệm & phản biện hội đồng:** [`RedLamp_Demo_Concepts.md`](./RedLamp_Demo_Concepts.md) | **Tính tay từng bước:** [`RedLamp_ConvAEC_StepByStep.md`](./RedLamp_ConvAEC_StepByStep.md)

---

## PHẦN 1 — TỔNG QUAN DỮ LIỆU & KẾT QUẢ ĐÁNH GIÁ

### 1.1. Giới thiệu 5 bộ dữ liệu & cấu hình thực nghiệm trên `SMD`

> *"Em chào thầy và các bạn, hôm nay em xin đại diện **Nhóm 1** thuyết trình về kết quả mà nhóm đã trực tiếp huấn luyện lại mô hình **REDLAMP** trên **5 bộ dữ liệu chuẩn**: gồm dữ liệu máy chủ công nghiệp `SMD`, chỉ số KPI đám mây `AIOps`, dữ liệu tàu tự hành Sao Hỏa `MSL` và vệ tinh `SMAP` của NASA, cùng kho dữ liệu chuỗi thời gian đa lĩnh vực `UCR`.*
>
> *Em xin lấy ví dụ bộ dữ liệu **`SMD`**, gồm **28 máy chủ** độc lập, mỗi máy chủ có **38 đặc trưng cảm biến (`D1 – D38`)** theo dõi đồng thời CPU, RAM, ổ cứng và lưu lượng mạng. Nhóm tổng hợp kết quả chung trên toàn bộ 28 máy (`ALL`) và huấn luyện độc lập qua **5 seed (`Seed 0 – 4`)** để đánh giá độ ổn định của mô hình với các vị trí chèn lỗi giả lập và khởi tạo trọng số ngẫu nhiên khác nhau."*

---

### 1.2. Kết quả đánh giá chất lượng qua 3 thông số (`VUS-ROC`, `VUS-PR`, `RF`)

> *"Về **kết quả đánh giá chất lượng**, mô hình được đánh giá qua **3 thông số** trên thang điểm từ `0` đến `1` (càng gần `1` càng tốt):*
>
> * *`VUS-ROC`: đo **khả năng phân biệt giữa vùng bình thường và vùng sự cố** (có dung sai nhẹ quanh rìa sự cố), trung bình 5 seed đạt **`0.823 ± 0.005`**.*
> * *`VUS-PR`: đánh giá **độ chính xác mỗi khi mô hình phát cảnh báo**, đạt **`0.374 ± 0.013`** — nhỉnh hơn nhẹ mức `0.366` của bài báo gốc.*
> * *`RF (Range F-score)`: đo **độ phủ trùng khớp và liền mạch** trên đợt sự cố thật sau khi cắt ngưỡng tự động, đạt **`0.239 ± 0.009`**."*

---

### 1.3. So sánh 28 máy chủ & đi sâu vào máy `machine-2-8`

> *"Xét riêng thông số đo trên từng máy trong **28 máy chủ** của tập `SMD` đối chiếu với mức trung bình toàn tập (`0.3569` ở `Seed 0`), máy nổi bật nhất là **`machine-2-8` (`m2-8`)** với khả năng phát hiện bất thường đạt hiệu quả cao nhất: `VUS-PR` đạt **`0.9818`** và `VUS-ROC` đạt **`0.9999`**.*
>
> *Đi vào chuỗi thời gian thực tế dài **`23,605` bước** của **`machine-2-8`**, tín hiệu gốc của đặc trưng cảm biến (`D1`, trong tổng số 38 đặc trưng `D1 – D38`) được đặt cùng đường điểm bất thường tương đối màu đỏ do REDLAMP dự đoán. Suốt giai đoạn dài từ đầu chuỗi tới mốc `16,000`, khi máy vận hành bình thường, điểm bất thường giữ ở mức thấp ổn định và không báo giả. Ngay khi phát sinh sự cố quanh mốc **`t = 17,587`**, tín hiệu cảm biến biến động vọt cao và điểm bất thường của REDLAMP lập tức dựng đứng chạm đỉnh **`1.00`** — với điểm kết hợp thô đạt `0.5689`, vượt xa ngưỡng tự động (`0.3257`), xác nhận chính xác trạng thái bất thường."*

---

## PHẦN 2 — GIẢI THÍCH MÔ HÌNH QUA 5 BƯỚC

### 2.0. Mở đầu cơ chế giải thích 5 bước

> *"Tiếp theo, nhóm sẽ đi sâu vào phân tích từng bước bên trong mạng `ConvAEC` để làm rõ vì sao mô hình kết luận bất thường — thông qua **5 bước tuần tự**: `Đầu vào` → `Encoder` → `Decoder` → `Classifier` → `Kết hợp`, trong đó **cửa sổ sự cố** luôn được đặt đối chiếu trực tiếp với một **cửa sổ chuẩn bình thường**."*

---

### 2.1. Bước 1/5 (`Đầu vào`)

> *"Ở **Bước 1 — Đầu vào**: thay vì chỉ xét rời rạc một thời điểm đơn lẻ, REDLAMP sử dụng một cửa sổ trượt độ dài **`W = 100` bước** liên tiếp trên toàn bộ **`38` kênh cảm biến**, giúp mô hình nắm bắt trọn vẹn bối cảnh biến thiên trước và trong khi máy phát sinh lỗi."*

---

### 2.2. Bước 2/5 (`Encoder`)

> *"Ở **Bước 2 — Encoder**: cửa sổ `100` bước của `38` cảm biến lần lượt đi qua **4 khối tích chập 1 chiều (`1D-CNN`)**. Qua từng tầng, chiều thời gian được **thu ngắn dần** từ `100` về `1`, trong khi số bộ lọc **mở rộng lên `128` rồi `256` kênh** để quét tìm các dấu hiệu biến dạng sóng như gai nhọn, trượt nền hay lệch pha, trước khi cô đọng lại thành vector đặc trưng **`Embedding z` gồm `128` chiều**.*
>
> *Để dễ so sánh, hệ thống tự động chọn một **mốc chuẩn đối chứng tại `t = 7,332`** — tức là một cửa sổ `100` bước trích từ giai đoạn máy đang vận hành ổn định, bình thường nhất trên chính chuỗi này (hiển thị bằng các cột màu xám). Trong khi mốc chuẩn `t = 7,332` có phân bố đặc trưng nằm trong biên độ vừa phải, thì tại cửa sổ sự cố, hàng loạt chiều của `z` bị **đảo ngược dấu** và **vọt biên độ vượt xa** khỏi các cột xám chuẩn — chứng tỏ `Encoder` đã đẩy vector của mẫu bất thường tách xa khỏi vùng bình thường trong không gian ẩn."*

---

### 2.3. Bước 3/5 (`Decoder` & Mặt nạ `Anomaly Mask`)

* **Trạng thái 1 — Mặc định `TẮT Mask` (Chế độ Kiểm thử):**
> *"Ở **Bước 3 — Nhánh `Decoder` và `Anomaly Mask`**: từ vector `128` chiều, `Decoder` khôi phục lại cửa sổ `38` cảm biến trong `100` bước (`3,800` phần tử). Khi **kiểm thử thực tế (tắt `Anomaly Mask`)**, toàn bộ `3,800` phần tử đều được mở để tính sai số: ở **cửa sổ chuẩn**, đường tái tạo màu xanh bám sát tín hiệu gốc với `MSE` rất nhỏ (`0.0003`); còn ở **cửa sổ sự cố**, dù tín hiệu gốc giật vọt lên sát mức `1.0`, đường tái tạo vẫn giữ nền ổn định quanh `0.15`, làm lộ vùng lệch đỏ rất lớn và đẩy `MSE` vọt lên gần `0.14` — gấp khoảng `400` lần mức chuẩn."*

* **Trạng thái 2 — Chuyển sang `BẬT Mask` (Mô phỏng Huấn luyện):**
> *"Đường tái tạo giữ được nền ổn định mà không vẽ chạy theo đỉnh lỗi là nhờ cơ chế **bật `Anomaly Mask` trong lúc huấn luyện**: ở cửa sổ chuẩn mọi phần tử đều được giữ nguyên (`1` — dải xanh lá), nhưng ở cửa sổ có lỗi, mô hình chỉ giữ lại đoạn sóng êm ở giữa và **phủ mặt nạ che lỗi (`0` — vùng sọc vàng)** lên hơn `56%` số phần tử đang vọt cao bất thường. Khi khóa sai số vùng lỗi về `0`, `MSE` giảm mạnh từ `0.14` xuống chỉ còn **`0.0006`** — nhờ vậy `Decoder` không bị ép học theo gai nhiễu mà chỉ học quy luật sóng bình thường."*

---

### 2.4. Bước 4/5 (`Classifier` 12 lớp & Lọc `FAA`)

> *"Ở **Bước 4 — Nhánh `Classifier` 12 lớp**: nhánh này giúp nhận diện cả những biến dạng hình thái sóng tinh vi mà riêng sai số biên độ có thể bỏ sót. Trong 12 lớp đầu ra, lớp `normal` cùng các lớp xuất hiện tràn lan trên 5% chuỗi sẽ bị bộ lọc **`FAA`** tự động loại trừ để chống báo giả; chỉ các lớp bất thường hợp lệ còn lại mới được cộng vào tổng điểm phân loại `Raw CE`.*
>
> *Tại **`Cửa sổ chuẩn`**, mô hình dự đoán gần như tuyệt đối vào lớp `normal` (`~99.9%`), tổng xác suất lỗi xấp xỉ `0`. Ngược lại tại **`Cửa sổ sự cố`**, xác suất lớp `normal` sụt thẳng về `0%` và dồn sang các lớp biến dạng bất thường hợp lệ, đẩy điểm phân loại `Raw CE` lên sát mức tối đa `1.0`."*

---

### 2.5. Bước 5/5 (`Kết hợp điểm phán quyết`)

> *"Ở **Bước 5 — Kết hợp điểm phán quyết**: REDLAMP lấy trung bình cộng giữa sai số tái tạo `Raw MSE` của nhánh `Decoder` và xác suất bất thường `Raw CE` của nhánh `Classifier`.*
>
> *Khi giám sát trực tuyến trên từng cửa sổ đơn lẻ chưa có toàn bộ chuỗi tương lai để chuẩn hóa `MinMax`, điểm kết hợp thô này được đối chiếu trực tiếp với ngưỡng thống kê tự động. Tại thời điểm sự cố, điểm kết hợp vượt xa ngưỡng cảnh báo, từ đó mô hình đưa ra phán quyết chính xác là **`BẤT THƯỜNG`**."*

---

## PHẦN 3 — THỬ NGHIỆM DỰ ĐOÁN TRỰC TUYẾN & KẾT LUẬN

### 3.1. Đối chứng điểm dữ liệu ở trạng thái chuẩn (`BÌNH THƯỜNG`)

> *"Cuối cùng, để mô phỏng kịch bản giám sát thời gian thực khi có một điểm dữ liệu mới truyền về tại thời điểm `t = 100`, mô hình tự động ghép **99 bước ngữ cảnh chuẩn ổn định (`t = 1..99`)** với bộ **38 thông số cảm biến (`D1..D38`)** trong miền `[0, 1]` tại `t = 100`.*
>
> *Khi điểm mới mang bộ thông số vận hành bình thường, mô hình lập tức kết luận **`BÌNH THƯỜNG`**, nhánh 12 lớp nhận diện **`Normal` với độ tự tin gần `100%`**, điểm kết hợp nằm rất sâu bên dưới ngưỡng tự động, và tín hiệu tại `t = 100` bám khít đường tái tạo."*

---

### 3.2. Giả lập đột biến cảm biến & Đồng bộ giải thích 5 bước

> *"Ngược lại, khi giả lập một sự cố nghẽn tải đột ngột làm một cụm cảm biến đầu vào vọt kịch trần trong miền `[0, 1]` tại thời điểm `t = 100`, hệ thống chuyển ngay sang cảnh báo **`BẤT THƯỜNG`** và gọi đúng hình thái lỗi là **`Spike (Đột biến đỉnh nhọn)`**. Kết quả này thể hiện rõ ưu thế cốt lõi của kiến trúc hai nhánh trong REDLAMP: vì cú giật chỉ xảy ra tại đúng 1 điểm cuối trên toàn cửa sổ `3,800` phần tử nên sai số tái tạo trung bình `MSE` tăng rất nhẹ — nếu chỉ dùng Autoencoder đơn thuần sẽ dễ bỏ lọt; nhưng **nhánh `Classifier` đã bắt gọn gai nhọn với `CE ≈ 1.0`**, kéo điểm kết hợp vượt qua ngưỡng cảnh báo để báo động chính xác.*
>
> *Đồng thời, toàn bộ cửa sổ chứa điểm đột biến vừa dự đoán cũng có thể được đồng bộ ngược về quy trình 5 bước để kỹ sư kiểm tra minh bạch từ đầu đến cuối lý do mô hình báo lỗi. Đó là toàn bộ phần trình bày thực nghiệm của Nhóm 1 về tính hiệu quả, bền vững và khả năng giải thích của mô hình REDLAMP. Em xin chân thành cảm ơn thầy và các bạn đã lắng nghe ạ!"*
