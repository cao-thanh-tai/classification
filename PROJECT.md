# classification

> File ngữ cảnh của project. AI đọc đầu mỗi phiên và cập nhật khi xong một mốc.

## Bài toán
Bài toán phân loại trái cây, thuộc loại classification với 10 classes

## Dữ liệu
- Nguồn:https://www.kaggle.com/datasets/karimabdulnabi/fruit-classification10-class
- Đường dẫn: data/raw/
- Cấu trúc:
┣ 📂predict
┣ 📂test
┃ ┣ 📂apple
┃ ┣ 📂avocado
┃ ┣ 📂banana
┃ ┣ 📂cherry
┃ ┣ 📂kiwi
┃ ┣ 📂mango
┃ ┣ 📂orange
┃ ┣ 📂pinenapple
┃ ┣ 📂stawberries
┃ ┗ 📂watermelon
┗ 📂train
  ┣ 📂Apple
  ┣ 📂avocado
  ┣ 📂Banana
  ┣ 📂cherry
  ┣ 📂kiwi
  ┣ 📂mango
  ┣ 📂orange
  ┣ 📂pinenapple
  ┣ 📂strawberries
  ┗ 📂watermelon
- Số lượng ảnh / lớp: 3373/10
- Cách chia: train/val theo tỉ lệ 80/20, test giữ làm benchmark; lưu ở `data/splits/v1/`
- Ghi chú đặc biệt: tên nhãn raw không đồng nhất; ảnh processed được chuẩn hóa JPEG/RGB.

## Mục tiêu
- So sánh các model: resnet50, vit_b16, convnext_t
- Metric chính: f1
- Metric phụ: tốc độ suy luận, số tham số, thời gian train
- Điều kiện thành công:

## Cài đặt thí nghiệm (recipe chung)
- Kích thước ảnh: 224×224
- Augmentation: resize 224×224; train dùng random horizontal flip; val/test chỉ resize; chuẩn hóa theo ImageNet mean/std.
- Optimizer / learning rate / epoch: AdamW, learning rate 0.0003, weight decay 0.0001, batch size 32, 10 epoch.
- Seed sẽ chạy: 42, 43, 44
- Pretrained weights: ImageNet weights từ torchvision, áp dụng đồng nhất cho cả ba model.

## Trạng thái
- [x] Điền Bài toán và Dữ liệu
- [x] EDA
- [x] Chốt split (`data/splits/v1/`)
- [x] Pipeline chạy thông suốt với một model (baseline smoke run)
- [x] Tạo factory cho các model resnet50, vit_b16, convnext_t
- [ ] Chạy đủ 3 seed cho mỗi model
- [ ] Chạy `compare.py`, chốt bảng so sánh trong `reports/`

## Quyết định đã chốt
- 2026-11-03, recipe chung dùng ảnh 224×224, horizontal flip khi train, chuẩn hóa ImageNet, AdamW (lr 0.0003, weight decay 0.0001), batch size 32, 10 epoch và ImageNet pretrained weights.
- 2026-11-03, pipeline train dùng chung đọc CSV split v1; smoke run có thể giới hạn số dòng bằng `--limit` và ghi checkpoint/log/metrics trong `experiments/`.
- 2026-11-03, giữ nguyên `data/raw/` trong quá trình EDA; notebook chỉ đọc dữ liệu.
- 2026-11-03, phải xử lý/loại trừ exact duplicate giữa train và test trước khi chốt split vì EDA phát hiện 122 nhóm ảnh trùng nội dung giữa hai split.
- 2026-11-03, `src/data/dataset.py` chỉ nhận `root`, danh sách `images`, danh sách `labels` và `transform`; chuẩn hóa nhãn sẽ thực hiện ở bước tạo `processed/`.
- 2026-11-03, tạo `data/splits/v1/` bằng seed 42, validation 20% theo từng lớp; CSV trỏ tới ảnh processed trong `data/processed/images/`.
- 2026-11-03, loại ảnh có cạnh ngắn dưới 50 px; chuyển ảnh hợp lệ sang JPEG RGB trong `data/processed/images/`, không sửa `data/raw/`.

## Kết quả tóm tắt
- EDA: 2.301 ảnh train, 1.025 ảnh test, 48 ảnh predict; 10 lớp.
- Không có ảnh lỗi khi đọc/verify trong ba split.
- Train có 5 nhóm duplicate nội bộ và test có 5 nhóm duplicate nội bộ.
- Có 122 nhóm exact duplicate giữa train và test; đã xử lý trước khi chốt split.
- Dataset v1: train 1.743 ảnh, val 436 ảnh, test 1.020 ảnh; loại 1 ảnh quá nhỏ, 116 ảnh train trùng test, 5 ảnh duplicate nội bộ train và 5 ảnh duplicate nội bộ test.
- Notebook chuẩn bị dataset `notebooks/01.1_prepare_dataset.ipynb` đã chạy thành công; ảnh processed là JPEG/RGB, cạnh ngắn tối thiểu 50 px.

## Vấn đề đang mở
- Baseline smoke run ResNet-50 thành công với lệnh `python scripts/train.py --model resnet50 --epochs 1 --limit 64`; kết quả chỉ là kiểm tra pipeline, không dùng để so sánh model.
- Cần chạy đầy đủ 3 seed cho mỗi model, sau đó chạy `compare.py` và chốt báo cáo.
