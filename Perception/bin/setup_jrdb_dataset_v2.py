#!/usr/bin/env python
"""
数据预处理脚本 - 使用 rosbags 库 (替代原始 rosbag 库)

功能:
1. 从 rosbag 提取 2D laser 扫描数据
2. 生成 frames_pc_im_laser.json 同步文件

使用方法:
    python bin/setup_jrdb_dataset_v2.py

作者: 自动生成
"""

import json
import numpy as np
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# 使用新的 rosbags 库
from rosbags.rosbag1 import Reader
from rosbags.serde import deserialize_cdr, ros1_to_cdr


# Set root dir to JRDB
_jrdb_dir = "./data/JRDB"

# Variables defining output location.
_output_laser_dir_name = "lasers"
_output_frames_laser_im_fname = "frames_img_laser.json"
_output_frames_laser_pc_fname = "frames_pc_laser.json"
_output_laser_timestamp_fname = "timestamps.txt"


def _laser_idx_to_fname(idx):
    return str(idx).zfill(6) + ".txt"


def extract_laser_from_rosbag(split):
    """Extract and save combined laser scan from rosbag using rosbags library."""
    data_dir = os.path.join(_jrdb_dir, split + "_dataset")

    timestamp_dir = os.path.join(data_dir, "timestamps")
    bag_dir = os.path.join(data_dir, "rosbags")
    
    if not os.path.exists(timestamp_dir):
        print(f"错误: timestamps 目录不存在: {timestamp_dir}")
        return
    
    sequence_names = sorted(os.listdir(timestamp_dir))

    laser_dir = os.path.join(data_dir, _output_laser_dir_name)
    if os.path.exists(laser_dir):
        shutil.rmtree(laser_dir)
    os.mkdir(laser_dir)

    for idx, seq_name in enumerate(tqdm(sequence_names, desc=f"提取 {split} laser 数据")):
        seq_laser_dir = os.path.join(laser_dir, seq_name)
        os.mkdir(seq_laser_dir)

        bag_file = os.path.join(bag_dir, seq_name + ".bag")
        
        if not os.path.exists(bag_file):
            print(f"警告: rosbag 文件不存在: {bag_file}")
            continue
            
        print(f"\n({idx + 1}/{len(sequence_names)}) 提取 laser: {seq_name}")

        # 使用 rosbags 库读取
        with Reader(bag_file) as reader:
            timestamp_list = []
            count = 0
            
            # 获取 laser topic 的连接 - 优先使用 scan_multi (合并的激光扫描)
            connections = [x for x in reader.connections if x.topic == 'segway/scan_multi']
            
            if not connections:
                # 尝试带斜杠的版本
                connections = [x for x in reader.connections if x.topic == '/segway/scan_multi']
            
            if connections:
                print(f"  使用 topic: {connections[0].topic}")
            
            if not connections:
                # 如果没有 scan_multi，尝试 filtered_scan
                connections = [x for x in reader.connections if 'filtered_scan' in x.topic and 'front' not in x.topic and 'rear' not in x.topic]
            
            if not connections:
                print(f"  警告: 找不到合并的 laser topic (scan_multi 或 filtered_scan)，跳过 {seq_name}")
                # 列出可用的 topics
                scan_topics = [x.topic for x in reader.connections if 'scan' in x.topic.lower()]
                print(f"  可用 scan topics: {scan_topics}")
                continue

            for connection, timestamp, rawdata in reader.messages(connections=connections):
                try:
                    # 反序列化消息
                    msg = deserialize_cdr(ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
                    
                    # 提取 ranges
                    scan = np.array(msg.ranges)
                    fname = _laser_idx_to_fname(count)
                    np.savetxt(os.path.join(seq_laser_dir, fname), scan, newline=" ")

                    # 时间戳转换为秒
                    timestamp_list.append(timestamp / 1e9)  # nanoseconds to seconds
                    count += 1
                except Exception as e:
                    print(f"  处理消息出错: {e}")
                    continue

            np.savetxt(
                os.path.join(seq_laser_dir, _output_laser_timestamp_fname),
                np.array(timestamp_list),
                newline=" ",
            )
            print(f"  提取了 {count} 帧 laser 数据")


def _match_pc_im_laser_one_sequence(split, sequence_name):
    """生成 frames_pc_im_laser.json 同步文件"""
    data_dir = os.path.join(_jrdb_dir, split + "_dataset")

    timestamp_dir = os.path.join(data_dir, "timestamps", sequence_name)
    laser_dir = os.path.join(data_dir, "lasers", sequence_name)

    # pc frames
    pc_frames_file = os.path.join(timestamp_dir, "frames_pc.json")
    with open(pc_frames_file, "r") as f:
        pc_frames = json.load(f)["data"]

    # im frames
    im_frames_file = os.path.join(timestamp_dir, "frames_img.json")
    with open(im_frames_file, "r") as f:
        im_frames = json.load(f)["data"]

    # synchronize pc and im frame
    pc_timestamp = np.array([float(f["timestamp"]) for f in pc_frames])
    im_timestamp = np.array([float(f["timestamp"]) for f in im_frames])

    pc_im_ft_diff = np.abs(pc_timestamp.reshape(-1, 1) - im_timestamp.reshape(1, -1))
    pc_im_matching_inds = pc_im_ft_diff.argmin(axis=1)

    # synchronize pc and laser
    laser_timestamp_file = os.path.join(laser_dir, _output_laser_timestamp_fname)
    if not os.path.exists(laser_timestamp_file):
        print(f"警告: laser timestamps 不存在: {laser_timestamp_file}")
        return False
        
    laser_timestamp = np.loadtxt(laser_timestamp_file, dtype=np.float64)
    
    # 确保 laser_timestamp 是 1D 数组
    if laser_timestamp.ndim == 0:
        laser_timestamp = np.array([laser_timestamp])
    
    pc_laser_ft_diff = np.abs(
        pc_timestamp.reshape(-1, 1) - laser_timestamp.reshape(1, -1)
    )
    pc_laser_matching_inds = pc_laser_ft_diff.argmin(axis=1)

    # create a merged frame dict
    output_frames = []
    for i in range(len(pc_frames)):
        frame = {
            "pc_frame": pc_frames[i],
            "im_frame": im_frames[pc_im_matching_inds[i]],
            "laser_frame": {
                "url": os.path.join(
                    _output_laser_dir_name,
                    sequence_name,
                    _laser_idx_to_fname(pc_laser_matching_inds[i]),
                ),
                "name": "laser_combined",
                "timestamp": float(laser_timestamp[pc_laser_matching_inds[i]]),
            },
            "timestamp": pc_frames[i]["timestamp"],
            "frame_id": pc_frames[i]["frame_id"],
        }

        # correct file url for pc and im
        for pc_dict in frame["pc_frame"]["pointclouds"]:
            f_name = os.path.basename(pc_dict["url"])
            pc_dict["url"] = os.path.join(
                "pointclouds", pc_dict["name"], sequence_name, f_name
            )

        for im_dict in frame["im_frame"]["cameras"]:
            f_name = os.path.basename(im_dict["url"])
            cam_name = (
                "image_stitched"
                if im_dict["name"] == "stitched_image0"
                else im_dict["name"][:-1] + "_" + im_dict["name"][-1]
            )
            im_dict["url"] = os.path.join("images", cam_name, sequence_name, f_name)

        output_frames.append(frame)

    # write to file
    output_dict = {"data": output_frames}
    f_name = os.path.join(timestamp_dir, "frames_pc_im_laser.json")
    with open(f_name, "w") as fp:
        json.dump(output_dict, fp)
    
    return True


def match_pc_im_laser(split):
    """为所有序列生成同步文件"""
    sequence_names = sorted(os.listdir(
        os.path.join(_jrdb_dir, split + "_dataset", "timestamps")
    ))
    
    success_count = 0
    for idx, seq_name in enumerate(tqdm(sequence_names, desc=f"生成 {split} 同步文件")):
        if _match_pc_im_laser_one_sequence(split, seq_name):
            success_count += 1
    
    print(f"\n{split}: 成功处理 {success_count}/{len(sequence_names)} 个序列")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="JRDB 数据集预处理")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "both"],
                        help="处理哪个数据集分割")
    parser.add_argument("--skip_laser", action="store_true",
                        help="跳过 laser 提取，只生成同步文件")
    args = parser.parse_args()
    
    if args.split == "both":
        splits = ["train", "test"]
    else:
        splits = [args.split]
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"处理 {split} 数据集")
        print(f"{'='*60}")
        
        if not args.skip_laser:
            extract_laser_from_rosbag(split)
        
        match_pc_im_laser(split)
    
    print("\n完成!")

