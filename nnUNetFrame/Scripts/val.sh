#!/bin/bash
# 普通训练：bash val.sh
# 带最佳配置查找：bash val.sh --find-best

DATASET_ID="2"
CONFIG="2d"
# TRAINER="nnUNetTrainer"
# PLANS="nnUNetPlans"
AVAILABLE_GPUS=(0 1 2 3)
MAX_RETRIES=10

# BASE_RESULTS_DIR="/data/taoyusong/nnUNet_IVUS/nnUNet/nnUNetFrame/DATASET/nnUNet_results/Dataset002_IVUS/${TRAINER}__${PLANS}__${CONFIG}"

get_gpu_process_count() {
    local gpu_id=$1
    nvidia-smi pmon -c 1 -i "$gpu_id" 2>/dev/null | grep -c "nnUNetv2_train" || echo "0"
}

launch_fold() {
    local fold=$1
    local gpu_id=$2
    
    (
        local retry_count=0
        
        while [[ $retry_count -le $MAX_RETRIES ]]; do
            local cmd="nnUNetv2_train $DATASET_ID $CONFIG $fold --val --npz"
            [[ $retry_count -gt 0 ]] && cmd="$cmd --c"
            
            export CUDA_VISIBLE_DEVICES="$gpu_id"
            echo "[$(date)] Fold $fold 开始执行 (尝试次数: $retry_count)"
            $cmd
            
            if [[ $? -eq 0 ]]; then
                echo "[$(date)] Fold $fold 成功完成"
                exit 0
            else
                retry_count=$((retry_count + 1))
                
                if [[ $retry_count -gt $MAX_RETRIES ]]; then
                    echo "[$(date)] Fold $fold 达到最大重试次数($MAX_RETRIES)，放弃"
                    exit 1
                fi
                
                echo "[$(date)] Fold $fold 执行失败，120秒后进行第 $retry_count 次重试..."
                sleep 120
            fi
        done
    ) &
    
    echo "[$(date)] Fold $fold 已提交到 GPU $gpu_id (PID: $!)"
}

echo "=========================================="
echo "nnU-Net 并行训练 (with --val --npz)"
echo "数据集: $DATASET_ID | 配置: $CONFIG | GPU: ${AVAILABLE_GPUS[*]}"
echo "最大重试: $MAX_RETRIES 次"
echo "=========================================="

for fold in {0..3}; do
    launch_fold $fold ${AVAILABLE_GPUS[$fold]}
    sleep 10
done

echo "[$(date)] 等待GPU 0空闲以启动 Fold 4..."
while [[ $(get_gpu_process_count 0) -gt 0 ]]; do
    sleep 30
done
launch_fold 4 0

echo "[$(date)] 所有fold已提交..."

wait

if [[ "$1" == "--find-best" ]]; then
    echo "[$(date)] 开始寻找最佳配置..."
    nnUNetv2_find_best_configuration $DATASET_ID -c $CONFIG -f 0 1 2 3 4
    echo "[$(date)] 最佳配置查找完成。"
fi

exit 0