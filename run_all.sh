#!/bin/bash

# Kiểm tra tham số task
TASK_TYPE=$1  # Lấy giá trị tham số đầu tiên, ví dụ: "derain"

# Cài đặt các phụ thuộc từ requirements.txt
pip install -r requirements.txt

# Kiểm tra giá trị của task và thực thi các lệnh tương ứng
if [ "$TASK_TYPE" == "derain" ]; then
    # Chạy code download dữ liệu cho derain
    python download_derain.py --data train-test
elif [ "$TASK_TYPE" == "real_denoise" ]; then
    # Chạy code download dữ liệu cho real denoise
    python download_denoise.py --data train-test --dataset SIDD --noise real
    # Chạy generate_patches_sidd.py
    python generate_patches_sidd.py
elif [ "$TASK_TYPE" == "motion_deblur" ]; then
    # Chạy code download dữ liệu cho motion deblur
    python download_motion_deblur.py --data train-test --dataset GoPro
    # Chạy generate_patch_GoPro.py
    python generate_patch_GoPro.py
elif [ "$TASK_TYPE" == "single_image_deblur" ]; then
    # Chạy code download dữ liệu cho single image deblur
    python download_defocus_deblur.py --data train-test
    # Chạy generate_patch_dpdd.py
    python generate_patch_dpdd.py
elif [ "$TASK_TYPE" == "gaussian_denoise" ]; then
    # Chạy code download dữ liệu cho gaussian denoise
    python download_denoise.py --data train-test --dataset SIDD --noise gaussian
    # Chạy generate_patches_dfwb.py
    python generate_patches_dfwb.py
else
    echo "Task type không hợp lệ. Vui lòng chọn một trong các task sau: derain, real_denoise, motion_deblur, single_image_deblur, gaussian_denoise."
    exit 1
fi

# Thêm quyền thực thi cho script run_ddp_train.sh
chmod +x run_ddp_train.sh

# Chạy script run_ddp_train.sh với tham số --task_type
./run_ddp_train.sh $TASK_TYPE
