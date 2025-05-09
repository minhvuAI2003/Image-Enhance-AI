# Hướng Dẫn Sử Dụng `run_all.sh`

## Cài Đặt và Chạy Script

### 1. Tạo quyền thực thi cho `run_all.sh`
Trước khi chạy script, bạn cần đảm bảo rằng file `run_all.sh` có quyền thực thi. Bạn có thể làm điều này bằng cách chạy lệnh sau trong terminal:

```bash
chmod +x run_all.sh
```

### 2. Chạy Script Với Các Tác Vụ Tùy Chọn
Sau khi đã cấp quyền thực thi, bạn có thể chạy script với các tham số task_type để thực hiện các tác vụ cụ thể. Dưới đây là các tác vụ có thể có và cách chạy chúng:

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

### 3. Giải Thích Các Tác Vụ
- **derain**: Chạy tác vụ giảm mưa cho ảnh.
- **real_denoise**: Chạy tác vụ khử nhiễu ảnh thực, bao gồm tải dữ liệu từ SIDD với noise thực và tạo các patch.
- **motion_deblur**: Chạy tác vụ làm rõ ảnh bị mờ do chuyển động.
- **single_image_deblur**: Chạy tác vụ làm rõ ảnh bị mờ theo phương pháp đơn.
- **gaussian_denoise**: Chạy tác vụ khử nhiễu ảnh với noise Gaussian.
