# Hướng Dẫn Sử Dụng `run_all.sh`

## Chuẩn Bị

### 1. Di Chuyển Vào Thư Mục Gốc
Đầu tiên, bạn cần di chuyển vào thư mục gốc của dự án:
```bash
cd Image-Enhance-AI# Hướng Dẫn Sử Dụng `run_all.sh`

## Chuẩn Bị

### 1. Di Chuyển Vào Thư Mục Gốc
Đầu tiên, bạn cần di chuyển vào thư mục gốc của dự án:
```bash
cd Image-Enhance-AI
```

### 2. Chỉnh Sửa File utils.py
Mở file `utils.py` và bỏ comment dòng 88 bằng cách xóa dấu # ở đầu dòng:
```python
# Từ:
# self.backend=args.backend

# Thành:
self.backend=args.backend
```

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

## Chạy Server

### 1. Cài Đặt Dependencies
Trước khi chạy server, hãy đảm bảo bạn đã cài đặt tất cả các dependencies cần thiết:
```bash
pip install -r requirements.txt
```

### 2. Khởi Động Server
Để chạy server, sử dụng lệnh sau:
```bash
python api.py
```
Server sẽ tự động:
- Khởi động FastAPI server trên port 8000
- Tạo ngrok tunnel để có thể truy cập từ internet
- In ra public URL để truy cập API từ bên ngoài

### 3. Truy Cập API
Sau khi server đã chạy, bạn có thể truy cập các API endpoints sau:
- API Documentation: http://localhost:8000/docs
- API Information: http://localhost:8000/

### 4. Các Endpoints Chính
- `/add-noise`: Thêm nhiễu Gaussian vào ảnh (có thể chỉ định mức độ nhiễu từ 1-75)
- `/derain`: Xử lý ảnh bị mưa
- `/gaussian-denoise`: Khử nhiễu Gaussian
- `/real-denoise`: Khử nhiễu ảnh thực
- `/motion-deblur`: Làm rõ ảnh bị mờ do chuyển động
- `/single-image-deblur`: Làm rõ ảnh bị mờ đơn

### 5. Cách Sử Dụng API
1. Gửi ảnh dưới dạng file qua POST request
2. Đối với endpoint `/add-noise`, có thể thêm tham số `level` để chỉ định mức độ nhiễu (1-75)
3. Server sẽ trả về ảnh đã xử lý dưới dạng PNG

### 6. Dừng Server
Để dừng server, nhấn `Ctrl + C` trong terminal đang chạy server.

```

### 2. Chỉnh Sửa File utils.py
Mở file `utils.py` và bỏ comment dòng 88 bằng cách xóa dấu # ở đầu dòng:
```python
# Từ:
# self.backend=args.backend

# Thành:
self.backend=args.backend
```

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

## Chạy Server

### 1. Cài Đặt Dependencies
Trước khi chạy server, hãy đảm bảo bạn đã cài đặt tất cả các dependencies cần thiết:
```bash
pip install -r requirements.txt
```

### 2. Khởi Động Server
Để chạy server, sử dụng lệnh sau:
```bash
python api.py
```

### 3. Truy Cập API
Sau khi server đã chạy, bạn có thể truy cập các API endpoints sau:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 4. Các Endpoints Chính
- `/process/derain`: Xử lý ảnh bị mưa
- `/process/real_denoise`: Khử nhiễu ảnh thực
- `/process/motion_deblur`: Làm rõ ảnh bị mờ do chuyển động
- `/process/single_image_deblur`: Làm rõ ảnh bị mờ đơn
- `/process/gaussian_denoise`: Khử nhiễu Gaussian

### 5. Dừng Server
Để dừng server, nhấn `Ctrl + C` trong terminal đang chạy server.
