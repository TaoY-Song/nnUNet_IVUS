#!/bin/bash
#预处理：nnUNetv2_plan_and_preprocess -d 002 --verify_dataset_integrity
# ==============================================================================
# nnU-Net 并行训练管理器（增强版-带NaN检测与断点续训）
# 功能：4个fold在4张GPU上并行，自动检测NaN，智能决定是否使用--c继续训练
# 使用 fold_X_completed 文件标记完成状态
# ==============================================================================

# ==================== 配置参数 ====================
DATASET_ID="002"
CONFIG="2d" 

# 2d：2D U-Net（用于 2D 和 3D 数据集）
# 3d_fullres：在高图像分辨率上运行的 3D U-Net（仅适用于 3D 数据集）
# 3d_lowres → 3d_cascade_fullres：3D U-Net 级联

AVAILABLE_GPUS=(0 1 2 3)
GPU_COUNT=${#AVAILABLE_GPUS[@]}
MAX_RETRIES=10

# ==================== GPU精确检测函数 ====================
get_gpu_process_count() {
    local gpu_id=$1
    nvidia-smi pmon -c 1 -i "$gpu_id" 2>/dev/null | grep -c "nnUNetv2_train" || echo "0"
}

get_total_gpu_usage() {
    local total=0
    for gpu_id in "${AVAILABLE_GPUS[@]}"; do
        total=$((total + $(get_gpu_process_count "$gpu_id")))
    done
    echo "$total"
}

# ==================== 核心：带NaN检测的fold启动函数 ====================
launch_fold() {
    local fold=$1
    local gpu_id=$2
    local state_file="fold_${fold}_completed"
    
    # 检查是否已完成
    if [[ -f "$state_file" ]]; then
        echo "[$(date)] Fold $fold 已标记完成，跳过"
        return 0
    fi
    
    # 在子shell中运行重试逻辑
    (
        local retry_count=0
        local use_continue_flag=""  # 用于记录下次是否使用 --c
        
        while [[ $retry_count -le $MAX_RETRIES ]]; do
            # 构建命令
            local cmd="nnUNetv2_train $DATASET_ID $CONFIG $fold"
            [[ -n "$use_continue_flag" ]] && cmd="$cmd $use_continue_flag"
            
            # 执行训练并捕获输出
            export CUDA_VISIBLE_DEVICES="$gpu_id"
            echo "[$(date)] Fold $fold 开始训练 (尝试 $retry_count/$MAX_RETRIES) 命令: $cmd"
            
            local output
            output=$(eval $cmd 2>&1)
            local exit_code=$?
            
            # 保存输出用于调试
            echo "$output" > "/tmp/fold_${fold}_attempt_${retry_count}.log"
            
            if [[ $exit_code -eq 0 ]]; then
                # 成功
                echo "[$(date)] Fold $fold 成功完成"
                touch "$state_file"
                exit 0
            else
                # 失败
                retry_count=$((retry_count + 1))
                
                # 重置 continue flag，根据本次失败原因决定是否设置
                use_continue_flag=""
                
                # 检查是否检测到NaN
                if echo "$output" | grep -q "NaN detected, current_epoch="; then
                    echo "[$(date)] Fold $fold 检测到 NaN"
                    
                    # 提取epoch数字
                    local current_epoch
                    current_epoch=$(echo "$output" | grep -oP "current_epoch=\K[0-9]+" | head -1)
                    
                    if [[ -n "$current_epoch" && "$current_epoch" -gt 10 ]]; then
                        echo "[$(date)] Fold $fold 在 epoch $current_epoch 出现 NaN，下次将使用 --c 继续"
                        use_continue_flag="--c"
                    else
                        echo "[$(date)] Fold $fold 在 epoch ${current_epoch:-unknown} 出现 NaN，下次将从头开始"
                        use_continue_flag=""
                    fi
                else
                    echo "[$(date)] Fold $fold 出现未知错误，下次将从头开始"
                    use_continue_flag=""
                fi
                
                # 检查是否达到最大重试次数
                if [[ $retry_count -gt $MAX_RETRIES ]]; then
                    echo "[$(date)] Fold $fold 达到最大重试次数 $MAX_RETRIES，放弃"
                    exit 1
                fi
                
                echo "[$(date)] Fold $fold 将在 2 分钟后重试..."
                sleep 120
            fi
        done
    ) &
    
    echo "[$(date)] Fold $fold 已提交到 GPU $gpu_id (PID: $!)"
}

# ==================== 主程序：并行启动所有folds ====================
echo "=========================================="
echo "nnU-Net 并行训练管理器 - 增强版"
echo "数据集: $DATASET_ID | 配置: $CONFIG"
echo "可用GPU: ${AVAILABLE_GPUS[*]} | 并行任务: $GPU_COUNT"
echo "最大重试: $MAX_RETRIES 次"
echo "NaN检测: 已启用 | 断点续训: 已启用"
echo "=========================================="

# 启动 fold 0-3（立即在4张GPU上运行）
for fold in {0..3}; do
    state_file="fold_${fold}_completed"
    if [[ -f "$state_file" ]]; then
        echo "[$(date)] Fold $fold 已标记完成，跳过"
        continue
    fi
    
    gpu_id=${AVAILABLE_GPUS[$fold]}
    launch_fold $fold $gpu_id
    sleep 10
done

# 等待GPU 0空闲后启动 fold 4
echo "[$(date)] 等待GPU 0空闲以启动 Fold 4..."
while [[ $(get_gpu_process_count 0) -gt 0 ]]; do
    sleep 30
done

# 检查 fold 4 是否已完成
if [[ ! -f "fold_4_completed" ]]; then
    launch_fold 4 0
else
    echo "[$(date)] Fold 4 已标记完成，跳过"
fi

echo "[$(date)] 所有fold已提交，训练在后台运行..."
echo "[$(date)] 可使用 'tail -f /tmp/fold_X_attempt_Y.log' 查看实时日志"
exit 0