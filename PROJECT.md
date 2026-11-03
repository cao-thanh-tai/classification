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
- [x] Hoàn thiện script tổng hợp comparison CSV/Markdown/biểu đồ
- [x] Chạy đủ 3 seed cho mỗi model
- [x] Chạy `compare.py`, xuất bảng so sánh trong `reports/`
- [ ] Đánh giá 9 checkpoint trên test split và xuất báo cáo test

## Quyết định đã chốt
- 2026-11-03, recipe chung dùng ảnh 224×224, horizontal flip khi train, chuẩn hóa ImageNet, AdamW (lr 0.0003, weight decay 0.0001), batch size 32, 10 epoch và ImageNet pretrained weights.
- 2026-11-03, pipeline train dùng chung đọc CSV split v1; smoke run có thể giới hạn số dòng bằng `--limit` và ghi checkpoint/log/metrics trong `experiments/`.
- 2026-11-03, hoàn tất 9 lượt train (3 model × seed 42/43/44), mỗi lượt 10 epoch trên cùng split v1 và recipe chung; bảng so sánh gộp trung bình ± độ lệch chuẩn theo seed.
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
- So sánh validation (mean ± std, 3 seed): ConvNeXt-T macro F1 0.959 ± 0.002, ResNet-50 0.953 ± 0.007, ViT-B/16 0.875 ± 0.024. ConvNeXt-T có F1 validation cao nhất trong các model đã chạy.
- Báo cáo đã tạo: `reports/comparison.csv`, `reports/comparison.md`, `reports/figures/comparison.png`; kết quả là trên validation, chưa phải benchmark test.

## Vấn đề đang mở
- Đã hoàn tất pipeline và so sánh validation. Script `python scripts/evaluate.py` đã sẵn sàng để đánh giá inference-only cả 9 checkpoint trên test split, lưu metrics theo từng run và xuất `reports/test_comparison.csv`, `reports/test_comparison.md`, `reports/figures/test_comparison.png`.
- Bước test chưa chạy; chỉ chạy sau khi đã chốt model/recipe từ validation. Test scores không được dùng để chọn lại model.
