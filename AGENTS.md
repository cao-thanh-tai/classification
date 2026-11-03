# Hướng dẫn cho AI (dùng chung cho mọi project trong CV/)

## 0. Thứ tự đọc (làm ở đầu mỗi phiên)

1. Đọc file này.
2. Đọc `STRUCTURE.md`.
3. Đọc `PROJECT.md` của project đang làm (bài toán, dữ liệu, trạng thái, quyết định đã chốt).
4. Nếu chưa rõ đang ở project nào, hỏi lại. Không đoán.

## 1. Môi trường

- Conda env: `cv`.
- Mọi lệnh chạy từ **gốc của project** (thư mục chứa `PROJECT.md`), không chạy từ trong `src/`.
- Cần thư viện mới thì thêm vào `dependencies` trong `pyproject.toml`, rồi cài lại. Không cài lẻ rồi quên ghi.
- Import theo kiểu: `from src.models.builder import build_model`.

## 2. Nguyên tắc cốt lõi

- **Model là cấu hình, không phải bản sao code.** Mỗi model là một thư mục `src/models/<tên>/` chứa `model.py` và `config.yaml`. KHÔNG tạo `train.py` riêng cho từng model.
- **Chỉ có một vòng train**, nằm ở `src/training/`. Mọi model dùng chung nó.
- **Phần dùng chung** (`src/data/`, `src/training/`, `src/evaluation/`, `src/comparison/`, `src/utils/`) không được sửa khi chỉ đang thêm model.
- **Tham số đưa vào `config/`**, không hardcode đường dẫn, learning rate, kích thước ảnh trong code.
- **`scripts/` chỉ là vỏ**: parse tham số rồi gọi hàm trong `src/`. Không đặt logic ở đây.
- Ưu tiên bọc model từ `timm` (gọi bằng tên) trước khi tự viết kiến trúc.

## 3. Quy trình thêm model mới

1. Tạo `src/models/<tên>/` gồm `__init__.py`, `model.py`, `config.yaml` (theo mẫu model đã có).
2. Đăng ký tên trong `src/models/registry.py`.
3. Chạy thử 1 epoch trên mẫu nhỏ: `python scripts/train.py --model <tên> --epochs 1 --limit 64`. Phải ra được `experiments/<tên>_seed<N>/metrics.json`.
4. Chỉ khi bước 3 thành công mới báo xong. Không báo "xong" khi chưa chạy.
5. Chỉ được thêm hoặc sửa file trong `src/models/<tên>/`, `src/models/registry.py` và config liên quan.

## 4. Quy trình train và so sánh

- Train: `python scripts/train.py --model <tên> --seed <N>`.
- Mỗi lần chạy ghi vào `experiments/<tên>_seed<N>/` gồm `checkpoints/`, `logs/`, `metrics.json`.
- `metrics.json` tối thiểu có: `model`, `seed`, metric chính, `params_m`, `infer_ms`, `train_min`.
- So sánh: `python scripts/compare.py`. Nó đọc mọi `metrics.json`, gộp theo model (trung bình các seed), xuất `reports/comparison.csv`, `reports/comparison.md`, `reports/figures/`.

## 5. Luật so sánh công bằng (bắt buộc)

- Mọi model dùng **cùng split** trong `data/splits/`.
- Mọi model dùng **cùng recipe** trong `config/train.yaml`. Chỉ đổi model.
- Chạy **ít nhất 3 seed** rồi lấy trung bình trước khi kết luận.
- Pretrained weights dùng **đồng nhất** (cùng nguồn, ví dụ cùng ImageNet) hoặc cùng train từ đầu.
- Báo cả tốc độ suy luận và số tham số, không chỉ độ chính xác.
- Nếu một model buộc phải khác recipe, ghi rõ lý do vào `PROJECT.md`, không âm thầm đổi.

## 6. Không được làm

- Không sửa hay xóa gì trong `data/raw/`.
- Không đổi `data/splits/` khi đang so sánh. Cần split mới thì tạo phiên bản mới (`splits/v2/`).
- Không commit `data/` và `experiments/`.
- Không tạo thư mục `misc/`, `temp/`, `old/`, `new/`, `backup/`.
- Không tự thêm, xóa, đổi tên thư mục cấp 1 hoặc cấp 2 ngoài `STRUCTURE.md`. Cần thư mục mới thì dừng lại và hỏi.
- Không bịa số liệu hoặc kết quả. Số nào chưa chạy thì ghi "chưa chạy".

## 7. Cách làm việc

- Tác vụ lớn: làm xong **một model chạy thông suốt** trước, rồi mới nhân ra các model còn lại.
- Sau khi chạy xong, báo ngắn gọn: đã làm gì, chạy lệnh nào, kết quả ra sao, chỗ nào chưa chắc.
- Khi kết thúc phiên hoặc xong một mốc, **cập nhật `PROJECT.md`** (mục Trạng thái và Quyết định đã chốt).
- Gặp lỗi: đọc kỹ thông báo lỗi, sửa nguyên nhân gốc. Không lặp lại cùng một lệnh hỏng nhiều lần.