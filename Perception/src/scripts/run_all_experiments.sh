#!/bin/bash
# 三组对比实验依次运行脚本
# 默认使用 GPU 0。可在运行前设置 CUDA_VISIBLE_DEVICES 覆盖。

echo "============================================"
echo "  DR-SPAAM 伪标签对比实验 - 多GPU训练"
echo "============================================"
echo ""
echo "配置:"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0}"
echo ""
echo "开始时间: $(date)"
echo "============================================"

# 设置默认 GPU；外部环境变量优先。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 进入工作目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

echo ""
echo "============================================"
echo "[实验 1/3] Mask R-CNN 伪标签"
echo "============================================"
echo "开始: $(date)"
python bin/train.py --cfg cfgs/dr_spaam_jrdb_maskrcnn_pseudo.yaml
echo "完成: $(date)"
echo ""

echo "============================================"
echo "[实验 2/3] Fused 伪标签 (Mask R-CNN + YOLOv8)"
echo "============================================"
echo "开始: $(date)"
python bin/train.py --cfg cfgs/dr_spaam_jrdb_fused_pseudo.yaml
echo "完成: $(date)"
echo ""

echo "============================================"
echo "[实验 3/3] YOLOv8 伪标签"
echo "============================================"
echo "开始: $(date)"
python bin/train.py --cfg cfgs/dr_spaam_jrdb_yolov8_pseudo.yaml
echo "完成: $(date)"
echo ""

echo "============================================"
echo "所有实验完成！"
echo "结束时间: $(date)"
echo ""
echo "结果保存在 logs/ 目录:"
ls -lhd logs/*pseudo*/
echo "============================================"
