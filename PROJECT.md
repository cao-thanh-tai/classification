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
- Số lượng ảnh / lớp:
- Cách chia: tỉ lệ train/val/test, lưu ở `data/splits/v1/`
- Ghi chú đặc biệt (mất cân bằng lớp, nhiễu, định dạng nhãn...):

## Mục tiêu
- So sánh các model: resnet50, vit_b16, convnext_t
- Metric chính:
- Metric phụ: tốc độ suy luận, số tham số, thời gian train
- Điều kiện thành công:

## Cài đặt thí nghiệm (recipe chung)
- Kích thước ảnh:
- Augmentation:
- Optimizer / learning rate / epoch:
- Seed sẽ chạy: 42, 43, 44
- Pretrained weights:

## Trạng thái
- [ ] Điền Bài toán và Dữ liệu
- [x] EDA
- [ ] Chốt split (`data/splits/v1/`)
- [ ] Pipeline chạy thông suốt với một model (baseline)
- [ ] Thêm các model còn lại
- [ ] Chạy đủ 3 seed cho mỗi model
- [ ] Chạy `compare.py`, chốt bảng so sánh trong `reports/`

## Quyết định đã chốt
- 2026-11-03, giữ nguyên `data/raw/` trong quá trình EDA; notebook chỉ đọc dữ liệu.
- 2026-11-03, phải xử lý/loại trừ exact duplicate giữa train và test trước khi chốt split vì EDA phát hiện 122 nhóm ảnh trùng nội dung giữa hai split.
- 2026-11-03, `src/data/dataset.py` dùng chung cho EDA và train, hỗ trợ cả class-folder và CSV manifest; nhãn được chuẩn hóa về 10 tên lớp canonical.

## Kết quả tóm tắt
- EDA: 2.301 ảnh train, 1.025 ảnh test, 48 ảnh predict; 10 lớp.
- Không có ảnh lỗi khi đọc/verify trong ba split.
- Train có 5 nhóm duplicate nội bộ và test có 5 nhóm duplicate nội bộ.
- Có 122 nhóm exact duplicate giữa train và test; chưa chốt split do vấn đề leakage này.

## Vấn đề đang mở
- Cần quyết định cách xử lý 122 nhóm exact duplicate train/test trước khi tạo `data/splits/v1/`.
- Cần tạo CSV manifest và split v1 sau khi xử lý 122 nhóm exact duplicate.
