#!/usr/bin/env python3
"""
使用 YOLOv8 生成 2D 行人检测结果，替代原有的 Mask R-CNN 检测结果。

用途：
    生成伪标签用于自监督训练 DR-SPAAM 激光雷达人员检测器。

输出格式与原 JRDB 数据集的 detections_2d_stitched 格式一致：
{
    "detections": {
        "000000.jpg": [
            {
                "box": [x1, y1, x2, y2],
                "file_id": "000000.jpg", 
                "label_id": "person:-1",
                "score": "0.95"
            }
        ]
    }
}

使用方法:
    python generate_yolov8_detections.py --data_dir ./data/JRDB --split train --output_dir ./data/JRDB/train_dataset/detections/detections_2d_stitched_yolov8
"""

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("请先安装 ultralytics: pip install ultralytics")
    exit(1)


class YOLOv8Detector:
    """YOLOv8 行人检测器封装"""
    
    # COCO 数据集中 person 类别的 ID
    PERSON_CLASS_ID = 0
    
    def __init__(self, model_name="yolov8x.pt", conf_thresh=0.25, iou_thresh=0.45, device=None):
        """
        初始化 YOLOv8 检测器
        
        Args:
            model_name: YOLOv8 模型名称，可选:
                - yolov8n.pt (最快，精度最低)
                - yolov8s.pt 
                - yolov8m.pt
                - yolov8l.pt
                - yolov8x.pt (最慢，精度最高)
            conf_thresh: 置信度阈值
            iou_thresh: NMS IOU 阈值
            device: 运行设备 ('cuda', 'cpu', 或 None 自动选择)
        """
        self.model = YOLO(model_name)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        
        print(f"已加载 YOLOv8 模型: {model_name}")
        print(f"置信度阈值: {conf_thresh}, IOU阈值: {iou_thresh}")
    
    def detect(self, image):
        """
        检测图像中的行人
        
        Args:
            image: BGR 格式图像 (numpy array) 或图像路径
            
        Returns:
            detections: list of dict, 每个 dict 包含:
                - box: [x1, y1, x2, y2]
                - score: float, 置信度
        """
        results = self.model(
            image, 
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=[self.PERSON_CLASS_ID],  # 只检测 person 类
            device=self.device,
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                conf = boxes.conf[i].cpu().numpy()
                
                detections.append({
                    "box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                    "score": float(conf)
                })
        
        return detections


def get_sequence_names(data_dir, split):
    """获取所有序列名称"""
    timestamp_dir = os.path.join(data_dir, f"{split}_dataset", "timestamps")
    if not os.path.exists(timestamp_dir):
        raise FileNotFoundError(f"找不到目录: {timestamp_dir}")
    
    sequences = sorted(os.listdir(timestamp_dir))
    return sequences


def get_image_files(data_dir, split, sequence_name):
    """获取序列中所有图像文件"""
    image_dir = os.path.join(
        data_dir, f"{split}_dataset", "images", "image_stitched", sequence_name
    )
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"找不到图像目录: {image_dir}")
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    return image_dir, image_files


def process_sequence(detector, data_dir, split, sequence_name, output_dir):
    """处理单个序列的所有图像"""
    image_dir, image_files = get_image_files(data_dir, split, sequence_name)
    
    detections_dict = {}
    
    for img_file in tqdm(image_files, desc=f"处理 {sequence_name}", leave=False):
        img_path = os.path.join(image_dir, img_file)
        
        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            print(f"警告: 无法读取图像 {img_path}")
            continue
        
        # 检测行人
        detections = detector.detect(image)
        
        # 转换为 JRDB 格式
        detections_formatted = []
        for det in detections:
            detections_formatted.append({
                "box": det["box"],
                "file_id": img_file,
                "label_id": "person:-1",
                "score": str(det["score"])  # JRDB 格式中 score 是字符串
            })
        
        detections_dict[img_file] = detections_formatted
    
    # 保存结果
    output_file = os.path.join(output_dir, f"{sequence_name}.json")
    with open(output_file, 'w') as f:
        json.dump({"detections": detections_dict}, f, indent=4)
    
    return len(image_files), sum(len(v) for v in detections_dict.values())


def main():
    parser = argparse.ArgumentParser(description="使用 YOLOv8 生成 2D 行人检测结果")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="./data/JRDB",
        help="JRDB 数据集根目录"
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="train",
        choices=["train", "test"],
        help="数据集分割 (train/test)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="输出目录 (默认: data_dir/split_dataset/detections/detections_2d_stitched_yolov8)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="yolov8x.pt",
        help="YOLOv8 模型路径或名称 (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)"
    )
    parser.add_argument(
        "--conf_thresh", 
        type=float, 
        default=0.25,
        help="置信度阈值"
    )
    parser.add_argument(
        "--iou_thresh", 
        type=float, 
        default=0.45,
        help="NMS IOU 阈值"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="运行设备 (cuda/cpu/None自动)"
    )
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="+",
        default=None,
        help="只处理指定的序列 (默认处理所有序列)"
    )
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(
            args.data_dir, 
            f"{args.split}_dataset", 
            "detections", 
            "detections_2d_stitched_yolov8"
        )
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"输出目录: {args.output_dir}")
    
    # 初始化检测器
    detector = YOLOv8Detector(
        model_name=args.model,
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh,
        device=args.device
    )
    
    # 获取序列列表
    if args.sequences:
        sequences = args.sequences
    else:
        sequences = get_sequence_names(args.data_dir, args.split)
    
    print(f"待处理序列数: {len(sequences)}")
    
    # 处理每个序列
    total_images = 0
    total_detections = 0
    
    for seq_name in tqdm(sequences, desc="处理序列"):
        try:
            n_images, n_dets = process_sequence(
                detector, args.data_dir, args.split, seq_name, args.output_dir
            )
            total_images += n_images
            total_detections += n_dets
        except Exception as e:
            print(f"处理序列 {seq_name} 时出错: {e}")
            continue
    
    print(f"\n完成! 共处理 {total_images} 张图像, 检测到 {total_detections} 个行人")
    print(f"结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()

