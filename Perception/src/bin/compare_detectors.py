#!/usr/bin/env python3
"""
对比不同检测器 (Mask R-CNN vs YOLOv8) 生成的检测结果。

分析指标:
    - 检测数量统计
    - 置信度分布
    - 边界框大小分布
    - 与GT标签的匹配情况 (如果有GT)

使用方法:
    python compare_detectors.py --data_dir ./data/JRDB --split train
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm


def load_detections(det_dir, sequence_name):
    """加载检测结果"""
    det_file = os.path.join(det_dir, f"{sequence_name}.json")
    if not os.path.exists(det_file):
        return None
    
    with open(det_file, 'r') as f:
        data = json.load(f)
    
    return data.get("detections", {})


def load_labels(label_dir, sequence_name):
    """加载 2D 标签"""
    label_file = os.path.join(label_dir, f"{sequence_name}.json")
    if not os.path.exists(label_file):
        return None
    
    with open(label_file, 'r') as f:
        data = json.load(f)
    
    return data.get("labels", {})


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


def analyze_detections(detections, name="Detector"):
    """分析检测结果统计信息"""
    if detections is None:
        return None
    
    all_scores = []
    all_widths = []
    all_heights = []
    all_areas = []
    all_aspect_ratios = []
    total_dets = 0
    
    for frame_id, dets in detections.items():
        for det in dets:
            box = det["box"]
            score = float(det["score"])
            
            width = box[2] - box[0]
            height = box[3] - box[1]
            area = width * height
            aspect_ratio = width / max(height, 1)
            
            all_scores.append(score)
            all_widths.append(width)
            all_heights.append(height)
            all_areas.append(area)
            all_aspect_ratios.append(aspect_ratio)
            total_dets += 1
    
    if total_dets == 0:
        return None
    
    return {
        "name": name,
        "total_frames": len(detections),
        "total_detections": total_dets,
        "avg_dets_per_frame": total_dets / len(detections),
        "scores": np.array(all_scores),
        "widths": np.array(all_widths),
        "heights": np.array(all_heights),
        "areas": np.array(all_areas),
        "aspect_ratios": np.array(all_aspect_ratios),
    }


def compute_matching_stats(detections, labels, iou_thresh=0.5):
    """计算检测结果与标签的匹配统计"""
    if detections is None or labels is None:
        return None
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for frame_id in labels.keys():
        gt_boxes = []
        for label in labels.get(frame_id, []):
            if "box" in label:
                box = label["box"]
                if len(box) == 4:
                    # 标签格式可能是 [x, y, w, h]
                    if box[2] < box[0] or box[3] < box[1]:
                        # 是 [x, y, w, h] 格式
                        gt_boxes.append([box[0], box[1], box[0]+box[2], box[1]+box[3]])
                    else:
                        gt_boxes.append(box)
        
        det_boxes = []
        for det in detections.get(frame_id, []):
            det_boxes.append(det["box"])
        
        # 简单匹配
        gt_matched = [False] * len(gt_boxes)
        det_matched = [False] * len(det_boxes)
        
        for i, det_box in enumerate(det_boxes):
            best_iou = 0
            best_gt_idx = -1
            for j, gt_box in enumerate(gt_boxes):
                if gt_matched[j]:
                    continue
                iou = compute_iou(det_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_thresh and best_gt_idx >= 0:
                gt_matched[best_gt_idx] = True
                det_matched[i] = True
                total_tp += 1
            else:
                total_fp += 1
        
        total_fn += sum(1 for m in gt_matched if not m)
    
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    
    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def plot_comparison(stats_list, output_path):
    """绘制对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    colors = ['blue', 'orange', 'green', 'red']
    
    # 1. 置信度分布
    ax = axes[0, 0]
    for i, stats in enumerate(stats_list):
        if stats is not None:
            ax.hist(stats["scores"], bins=50, alpha=0.5, 
                   label=f'{stats["name"]} (μ={stats["scores"].mean():.3f})',
                   color=colors[i % len(colors)])
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution")
    ax.legend()
    
    # 2. 边界框宽度分布
    ax = axes[0, 1]
    for i, stats in enumerate(stats_list):
        if stats is not None:
            ax.hist(stats["widths"], bins=50, alpha=0.5,
                   label=f'{stats["name"]} (μ={stats["widths"].mean():.1f})',
                   color=colors[i % len(colors)])
    ax.set_xlabel("Box Width (pixels)")
    ax.set_ylabel("Count")
    ax.set_title("Box Width Distribution")
    ax.legend()
    
    # 3. 边界框高度分布
    ax = axes[0, 2]
    for i, stats in enumerate(stats_list):
        if stats is not None:
            ax.hist(stats["heights"], bins=50, alpha=0.5,
                   label=f'{stats["name"]} (μ={stats["heights"].mean():.1f})',
                   color=colors[i % len(colors)])
    ax.set_xlabel("Box Height (pixels)")
    ax.set_ylabel("Count")
    ax.set_title("Box Height Distribution")
    ax.legend()
    
    # 4. 宽高比分布
    ax = axes[1, 0]
    for i, stats in enumerate(stats_list):
        if stats is not None:
            ar = stats["aspect_ratios"]
            ax.hist(ar[ar < 2], bins=50, alpha=0.5,
                   label=f'{stats["name"]} (μ={ar.mean():.3f})',
                   color=colors[i % len(colors)])
    ax.set_xlabel("Aspect Ratio (W/H)")
    ax.set_ylabel("Count")
    ax.set_title("Aspect Ratio Distribution")
    ax.axvline(x=0.45, color='red', linestyle='--', label='Pseudo-label Threshold (0.45)')
    ax.legend()
    
    # 5. 检测数量对比
    ax = axes[1, 1]
    names = [s["name"] for s in stats_list if s is not None]
    counts = [s["total_detections"] for s in stats_list if s is not None]
    bars = ax.bar(names, counts, color=colors[:len(names)])
    ax.set_ylabel("Total Detections")
    ax.set_title("Total Detections Comparison")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
               f'{count:,}', ha='center', va='bottom')
    
    # 6. 每帧平均检测数
    ax = axes[1, 2]
    avg_counts = [s["avg_dets_per_frame"] for s in stats_list if s is not None]
    bars = ax.bar(names, avg_counts, color=colors[:len(names)])
    ax.set_ylabel("Avg Detections per Frame")
    ax.set_title("Avg Detections per Frame")
    for bar, count in zip(bars, avg_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
               f'{count:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"对比图已保存: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="对比不同检测器的结果")
    parser.add_argument("--data_dir", type=str, default="./data/JRDB")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="./comparison_results")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检测结果目录
    det_dir_maskrcnn = os.path.join(
        args.data_dir, f"{args.split}_dataset", "detections", "detections_2d_stitched"
    )
    det_dir_yolov8 = os.path.join(
        args.data_dir, f"{args.split}_dataset", "detections", "detections_2d_stitched_yolov8"
    )
    det_dir_fused = os.path.join(
        args.data_dir, f"{args.split}_dataset", "detections", "detections_2d_stitched_fused"
    )
    label_dir = os.path.join(
        args.data_dir, f"{args.split}_dataset", "labels", "labels_2d_stitched"
    )
    
    # 获取序列列表
    timestamp_dir = os.path.join(args.data_dir, f"{args.split}_dataset", "timestamps")
    sequences = sorted(os.listdir(timestamp_dir))
    
    print(f"共 {len(sequences)} 个序列")
    
    # 收集所有检测结果
    all_dets_maskrcnn = {}
    all_dets_yolov8 = {}
    all_dets_fused = {}
    all_labels = {}
    
    for seq_name in tqdm(sequences, desc="加载数据"):
        dets_maskrcnn = load_detections(det_dir_maskrcnn, seq_name)
        dets_yolov8 = load_detections(det_dir_yolov8, seq_name)
        dets_fused = load_detections(det_dir_fused, seq_name)
        labels = load_labels(label_dir, seq_name)
        
        if dets_maskrcnn:
            all_dets_maskrcnn.update(dets_maskrcnn)
        if dets_yolov8:
            all_dets_yolov8.update(dets_yolov8)
        if dets_fused:
            all_dets_fused.update(dets_fused)
        if labels:
            all_labels.update(labels)
    
    # 分析检测结果
    print("\n=== 检测结果分析 ===")
    
    stats_maskrcnn = analyze_detections(all_dets_maskrcnn, "Mask R-CNN")
    stats_yolov8 = analyze_detections(all_dets_yolov8, "YOLOv8")
    stats_fused = analyze_detections(all_dets_fused, "Fused")
    
    stats_list = [s for s in [stats_maskrcnn, stats_yolov8, stats_fused] if s is not None]
    
    for stats in stats_list:
        print(f"\n{stats['name']}:")
        print(f"  总帧数: {stats['total_frames']}")
        print(f"  总检测数: {stats['total_detections']}")
        print(f"  每帧平均检测数: {stats['avg_dets_per_frame']:.2f}")
        print(f"  置信度: μ={stats['scores'].mean():.4f}, σ={stats['scores'].std():.4f}")
        print(f"  边界框宽度: μ={stats['widths'].mean():.1f}, σ={stats['widths'].std():.1f}")
        print(f"  边界框高度: μ={stats['heights'].mean():.1f}, σ={stats['heights'].std():.1f}")
        print(f"  宽高比: μ={stats['aspect_ratios'].mean():.3f}, σ={stats['aspect_ratios'].std():.3f}")
        
        # 伪标签相关统计
        ar = stats['aspect_ratios']
        scores = stats['scores']
        valid_for_pl = (ar < 0.45) & (scores > 0.75)
        print(f"  符合伪标签条件 (AR<0.45, conf>0.75): {valid_for_pl.sum()} ({100*valid_for_pl.mean():.1f}%)")
    
    # 与 GT 标签对比 (如果有)
    if len(all_labels) > 0:
        print("\n=== 与 GT 标签匹配分析 (IoU=0.5) ===")
        
        if all_dets_maskrcnn:
            match_stats = compute_matching_stats(all_dets_maskrcnn, all_labels, iou_thresh=0.5)
            if match_stats:
                print(f"\nMask R-CNN:")
                print(f"  TP: {match_stats['tp']}, FP: {match_stats['fp']}, FN: {match_stats['fn']}")
                print(f"  Precision: {match_stats['precision']:.4f}")
                print(f"  Recall: {match_stats['recall']:.4f}")
                print(f"  F1: {match_stats['f1']:.4f}")
        
        if all_dets_yolov8:
            match_stats = compute_matching_stats(all_dets_yolov8, all_labels, iou_thresh=0.5)
            if match_stats:
                print(f"\nYOLOv8:")
                print(f"  TP: {match_stats['tp']}, FP: {match_stats['fp']}, FN: {match_stats['fn']}")
                print(f"  Precision: {match_stats['precision']:.4f}")
                print(f"  Recall: {match_stats['recall']:.4f}")
                print(f"  F1: {match_stats['f1']:.4f}")
        
        if all_dets_fused:
            match_stats = compute_matching_stats(all_dets_fused, all_labels, iou_thresh=0.5)
            if match_stats:
                print(f"\nFused (Mask R-CNN + YOLOv8):")
                print(f"  TP: {match_stats['tp']}, FP: {match_stats['fp']}, FN: {match_stats['fn']}")
                print(f"  Precision: {match_stats['precision']:.4f}")
                print(f"  Recall: {match_stats['recall']:.4f}")
                print(f"  F1: {match_stats['f1']:.4f}")
    
    # 绘制对比图
    if len(stats_list) > 0:
        plot_path = os.path.join(args.output_dir, "detector_comparison.png")
        plot_comparison(stats_list, plot_path)
    
    print(f"\n结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()

