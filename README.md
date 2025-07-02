# Hướng Dẫn Huấn Luyện Restormer trên nhiều GPU

### 1. Di Chuyển Vào Thư Mục Gốc
Đầu tiên, bạn cần di chuyển vào thư mục gốc của dự án:
```bash
cd Image-Enhance-AI
```
### 2. Tạo quyền thực thi cho `run_all.sh`
Trước khi chạy script, bạn cần đảm bảo rằng file `run_all.sh` có quyền thực thi. Chạy lệnh sau trong terminal:
```bash
chmod +x run_all.sh
```

### 3. Chạy Script Với Các Tác Vụ Tùy Chọn
Sau khi đã cấp quyền thực thi, chạy script với các tham số task_type để thực hiện các tác vụ cụ thể. Dưới đây là các tác vụ có thể có và cách chạy chúng:

#### Task: derain
```bash
./run_all.sh derain
```

#### Task: real_denoise
```bash
./run_all.sh real_denoise
```

#### Task: motion_deblur
```bash
./run_all.sh motion_deblur
```

#### Task: single_image_deblur
```bash
./run_all.sh single_image_deblur
```

#### Task: gaussian_denoise
```bash
./run_all.sh gaussian_denoise
```

### 4. Giải Thích Các Tác Vụ
- **derain**: Chạy tác vụ giảm mưa cho ảnh.
- **real_denoise**: Chạy tác vụ khử nhiễu ảnh thực.
- **motion_deblur**: Chạy tác vụ làm rõ ảnh bị mờ do chuyển động.
- **single_image_deblur**: Chạy tác vụ làm rõ ảnh bị mờ tiêu cự.
- **gaussian_denoise**: Chạy tác vụ khử nhiễu Gauss.

Link tham khảo: https://github.com/swz30/Restormer

