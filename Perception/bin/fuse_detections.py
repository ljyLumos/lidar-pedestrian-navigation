#!/usr/bin/env python
"""
融合 Mask R-CNN 和 YOLOv8 的检测结果

策略：用 Mask R-CNN 的高召回率 + YOLOv8 验证过滤
- 保留被 YOLOv8 验证的 Mask R-CNN 检测（高置信度）
- 对未被验证的 Mask R-CNN 检测降低置信度
- 可选：添加 YOLOv8 独有的检测
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def compute_iou(box1, box2):
    """计算两个边界框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0
    
    return inter_area / union_area


def fuse_detections_for_frame(maskrcnn_dets, yolov8_dets, 
                               iou_thresh=0.5,
                               verified_boost=1.0,
                               unverified_penalty=0.5,
                               add_yolo_unique=True,
                               min_conf=0.3):
    """
    融合单帧的检测结果
    
    Args:
        maskrcnn_dets: Mask R-CNN 检测列表 [{"box": [x1,y1,x2,y2], "score": float}, ...]
        yolov8_dets: YOLOv8 检测列表
        iou_thresh: IoU 阈值，超过此值认为是同一目标
        verified_boost: 被验证检测的置信度提升因子
        unverified_penalty: 未被验证检测的置信度惩罚因子
        add_yolo_unique: 是否添加 YOLOv8 独有的检测
        min_conf: 最小置信度阈值
    
    Returns:
        fused_dets: 融合后的检测列表
    """
    fused_dets = []
    yolo_matched = [False] * len(yolov8_dets)
    
    for mrcnn_det in maskrcnn_dets:
        mrcnn_box = mrcnn_det["box"]
        mrcnn_score = mrcnn_det["score"]
        
        # 查找与此 Mask R-CNN 检测重叠的 YOLOv8 检测
        best_iou = 0
        best_yolo_idx = -1
        best_yolo_score = 0
        
        for yolo_idx, yolo_det in enumerate(yolov8_dets):
            yolo_box = yolo_det["box"]
            iou = compute_iou(mrcnn_box, yolo_box)
            
            if iou > best_iou:
                best_iou = iou
                best_yolo_idx = yolo_idx
                best_yolo_score = yolo_det["score"]
        
        if best_iou >= iou_thresh:
            # 被 YOLOv8 验证的检测 - 提升置信度
            # 融合置信度：取两者的加权平均或最大值
            fused_score = min(1.0, max(mrcnn_score, best_yolo_score) * verified_boost)
            
            # 融合边界框：取两者的平均
            yolo_box = yolov8_dets[best_yolo_idx]["box"]
            fused_box = [
                (mrcnn_box[0] + yolo_box[0]) / 2,
                (mrcnn_box[1] + yolo_box[1]) / 2,
                (mrcnn_box[2] + yolo_box[2]) / 2,
                (mrcnn_box[3] + yolo_box[3]) / 2,
            ]
            
            fused_dets.append({
                "box": fused_box,
                "score": fused_score,
                "verified": True,
                "source": "fused"
            })
            
            yolo_matched[best_yolo_idx] = True
        else:
            # 未被 YOLOv8 验证的检测 - 降低置信度
            penalized_score = mrcnn_score * unverified_penalty
            
            if penalized_score >= min_conf:
                fused_dets.append({
                    "box": mrcnn_box,
                    "score": penalized_score,
                    "verified": False,
                    "source": "maskrcnn_only"
                })
    
    # 添加 YOLOv8 独有的检测
    if add_yolo_unique:
        for yolo_idx, yolo_det in enumerate(yolov8_dets):
            if not yolo_matched[yolo_idx]:
                # YOLOv8 精度高，独有检测也可能是真正的行人
                if yolo_det["score"] >= min_conf:
                    fused_dets.append({
                        "box": yolo_det["box"],
                        "score": yolo_det["score"],
                        "verified": False,
                        "source": "yolov8_only"
                    })
    
    return fused_dets


def load_detections(json_path):
    """加载检测结果 JSON 文件"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 转换为按帧组织的字典
    dets_by_frame = defaultdict(list)
    
    if "detections" in data:
        # 实际的 JRDB 检测格式: detections[frame_name] = [list of dets]
        for frame_name, dets in data["detections"].items():
            for det in dets:
                box = det["box"]  # [x1, y1, x2, y2]
                score = float(det["score"])  # score 是字符串格式
                
                dets_by_frame[frame_name].append({
                    "box": box,
                    "score": score
                })
    elif "labels" in data:
        # 备用格式
        for label in data["labels"]:
            frame_name = label["attributes"]["frame"]
            box = label["box2d"]
            x1 = box["x1"]
            y1 = box["y1"]
            x2 = box["x2"]
            y2 = box["y2"]
            score = label["attributes"].get("score", 1.0)
            
            dets_by_frame[frame_name].append({
                "box": [x1, y1, x2, y2],
                "score": score
            })
    
    return dets_by_frame


def save_fused_detections(fused_dets_by_frame, output_path, sequence_name):
    """保存融合后的检测结果（与原始格式兼容）"""
    detections = {}
    
    for frame_name, dets in fused_dets_by_frame.items():
        detections[frame_name] = []
        for det in dets:
            det_entry = {
                "box": [
                    det["box"][0],
                    det["box"][1],
                    det["box"][2],
                    det["box"][3]
                ],
                "file_id": frame_name,
                "label_id": "person:-1",
                "score": str(det["score"]),
                "verified": det.get("verified", False),
                "source": det.get("source", "unknown")
            }
            detections[frame_name].append(det_entry)
    
    output_data = {"detections": detections}
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="融合 Mask R-CNN 和 YOLOv8 检测结果")
    parser.add_argument("--data_dir", type=str, default="./data/JRDB",
                        help="JRDB 数据集路径")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"],
                        help="数据集划分")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录 (默认: detections_2d_stitched_fused)")
    parser.add_argument("--iou_thresh", type=float, default=0.5,
                        help="IoU 阈值")
    parser.add_argument("--verified_boost", type=float, default=1.0,
                        help="被验证检测的置信度提升因子")
    parser.add_argument("--unverified_penalty", type=float, default=0.5,
                        help="未被验证检测的置信度惩罚因子")
    parser.add_argument("--add_yolo_unique", action="store_true", default=True,
                        help="是否添加 YOLOv8 独有的检测")
    parser.add_argument("--no_yolo_unique", action="store_false", dest="add_yolo_unique",
                        help="不添加 YOLOv8 独有的检测")
    parser.add_argument("--min_conf", type=float, default=0.3,
                        help="最小置信度阈值")
    
    args = parser.parse_args()
    
    # 设置路径
    split_dir = f"{args.split}_dataset"
    det_base_dir = os.path.join(args.data_dir, split_dir, "detections")
    
    maskrcnn_dir = os.path.join(det_base_dir, "detections_2d_stitched")
    yolov8_dir = os.path.join(det_base_dir, "detections_2d_stitched_yolov8")
    
    if args.output_dir is None:
        output_dir = os.path.join(det_base_dir, "detections_2d_stitched_fused")
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查输入目录
    if not os.path.exists(maskrcnn_dir):
        print(f"错误: Mask R-CNN 检测目录不存在: {maskrcnn_dir}")
        return
    
    if not os.path.exists(yolov8_dir):
        print(f"错误: YOLOv8 检测目录不存在: {yolov8_dir}")
        return
    
    # 获取所有序列
    maskrcnn_files = sorted([f for f in os.listdir(maskrcnn_dir) if f.endswith('.json')])
    
    print(f"找到 {len(maskrcnn_files)} 个序列")
    print(f"融合参数:")
    print(f"  IoU 阈值: {args.iou_thresh}")
    print(f"  验证提升因子: {args.verified_boost}")
    print(f"  未验证惩罚因子: {args.unverified_penalty}")
    print(f"  添加 YOLOv8 独有检测: {args.add_yolo_unique}")
    print(f"  最小置信度: {args.min_conf}")
    print()
    
    # 统计信息
    total_stats = {
        "total_maskrcnn": 0,
        "total_yolov8": 0,
        "total_fused": 0,
        "verified": 0,
        "maskrcnn_only": 0,
        "yolov8_only": 0,
        "frames": 0
    }
    
    for json_file in tqdm(maskrcnn_files, desc="处理序列"):
        sequence_name = json_file.replace('.json', '')
        
        maskrcnn_path = os.path.join(maskrcnn_dir, json_file)
        yolov8_path = os.path.join(yolov8_dir, json_file)
        
        if not os.path.exists(yolov8_path):
            print(f"警告: YOLOv8 检测文件不存在: {yolov8_path}, 跳过")
            continue
        
        # 加载检测结果
        maskrcnn_dets = load_detections(maskrcnn_path)
        yolov8_dets = load_detections(yolov8_path)
        
        # 融合每一帧
        fused_dets_by_frame = {}
        
        all_frames = set(maskrcnn_dets.keys()) | set(yolov8_dets.keys())
        
        for frame_name in all_frames:
            mrcnn_frame_dets = maskrcnn_dets.get(frame_name, [])
            yolo_frame_dets = yolov8_dets.get(frame_name, [])
            
            fused_frame_dets = fuse_detections_for_frame(
                mrcnn_frame_dets,
                yolo_frame_dets,
                iou_thresh=args.iou_thresh,
                verified_boost=args.verified_boost,
                unverified_penalty=args.unverified_penalty,
                add_yolo_unique=args.add_yolo_unique,
                min_conf=args.min_conf
            )
            
            fused_dets_by_frame[frame_name] = fused_frame_dets
            
            # 统计
            total_stats["total_maskrcnn"] += len(mrcnn_frame_dets)
            total_stats["total_yolov8"] += len(yolo_frame_dets)
            total_stats["total_fused"] += len(fused_frame_dets)
            total_stats["frames"] += 1
            
            for det in fused_frame_dets:
                source = det.get("source", "unknown")
                if source == "fused":
                    total_stats["verified"] += 1
                elif source == "maskrcnn_only":
                    total_stats["maskrcnn_only"] += 1
                elif source == "yolov8_only":
                    total_stats["yolov8_only"] += 1
        
        # 保存融合结果
        output_path = os.path.join(output_dir, json_file)
        save_fused_detections(fused_dets_by_frame, output_path, sequence_name)
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("融合统计")
    print("=" * 60)
    print(f"总帧数: {total_stats['frames']}")
    print(f"Mask R-CNN 原始检测: {total_stats['total_maskrcnn']}")
    print(f"YOLOv8 原始检测: {total_stats['total_yolov8']}")
    print(f"融合后总检测: {total_stats['total_fused']}")
    print()
    print("检测来源分布:")
    print(f"  - 双方验证 (fused): {total_stats['verified']} ({100*total_stats['verified']/max(1,total_stats['total_fused']):.1f}%)")
    print(f"  - 仅 Mask R-CNN: {total_stats['maskrcnn_only']} ({100*total_stats['maskrcnn_only']/max(1,total_stats['total_fused']):.1f}%)")
    print(f"  - 仅 YOLOv8: {total_stats['yolov8_only']} ({100*total_stats['yolov8_only']/max(1,total_stats['total_fused']):.1f}%)")
    print()
    print(f"结果保存在: {output_dir}")


if __name__ == "__main__":
    main()

