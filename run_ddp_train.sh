#!/bin/bash

# 🧠 Set số tiến trình bằng số GPU
NUM_PROCESSES=4

# ✅ Lấy IP thật thay vì 127.0.0.1
MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=29500

# 🚫 Xóa cấu hình cũ có thể gây lỗi NCCL
unset NCCL_SOCKET_IFNAME
unset NCCL_P2P_DISABLE

# 🧵 Giới hạn số luồng OpenMP để tránh overload CPU
export OMP_NUM_THREADS=4

# ⚙️ Thêm tên interface mạng nếu biết (ví dụ: eth0)
# export NCCL_SOCKET_IFNAME=eth0

# Nhận tham số task_type từ dòng lệnh
TASK_TYPE=$1  # Lấy giá trị tham số đầu tiên, ví dụ: "derain"

# Kiểm tra xem task_type có được cung cấp không
if [ -z "$TASK_TYPE" ]; then
  echo "Lỗi: Bạn cần cung cấp tham số task_type."
  echo "Ví dụ: ./run_ddp_train.sh derain"
  exit 1
fi

# 🚀 Chạy torchrun với backend NCCL và tham số --task_type
torchrun \
  --nproc_per_node=$NUM_PROCESSES \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train_ddp.py \
  --data_path ./ \
  --data_name Datasets \
  --backend nccl \
  --task_type $TASK_TYPE  # Thêm tham số --task_type vào lệnh
