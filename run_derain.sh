#!/bin/bash

# Cài đặt các phụ thuộc từ requirements.txt
pip install -r requirements.txt

# Tải dữ liệu bằng script download_derain.py
python download_derain.py --data train-test

# Thêm quyền thực thi cho script run_ddp_train.sh
chmod +x run_ddp_train.sh

# Chạy script run_ddp_train.sh
./run_ddp_train.sh
