# lidar-pedestrian-navigation

### 2D LiDAR-based Pedestrian Detection and Dynamic Robot Navigation in ROS/Gazebo

This repository implements an end-to-end dynamic environment navigation system for mobile robots, integrating **high-precision perception** and **robust decision-making** through DR-SPAAM-based LiDAR pedestrian detection and TD3-based reinforcement learning.

### [**Project Report**](./docs/report.pdf)

## 📋 Overview

The system addresses safe and efficient navigation of robots in environments with dynamic pedestrians. It consists of two primary stages:

1. **Perception**: High-precision 2D LiDAR-based pedestrian detection enhanced with YOLOv8 pseudo-labeling, improving AP@0.5m from 0.417 → 0.566 (+35.8%).  
2. **Decision & Control**: TD3 reinforcement learning augmented with a dense reward function and four-stage curriculum learning, achieving navigation success rate of 63.0% in dense dynamic scenarios (+96.9% over baseline TD3).

The system outputs semantic obstacle information from perception directly into the RL environment, reducing search complexity and improving policy robustness.

![System Architecture](assets/framework.png)  
*Overall system architecture and layered design.*

## 🎯 Motivation

Original DR-SPAAM-based TD3 navigation suffered from:

- **Q-value overestimation & policy instability**: Single-Q networks and naive reward design led to erratic behaviors and low convergence.  
- **Sparse and imbalanced rewards**: Resulted in reward hacking (e.g., robot freezing or circling).  
- **Lack of progressive training**: Training directly in dense dynamic scenes led to extremely low success rates and high compute cost.

Our goal is to overcome these limitations by integrating high-quality perception, reward reshaping, and curriculum-based staged training.

## 🛠 Method

### 1. Perception Stage (YOLOv8-enhanced DR-SPAAM)

- **Pseudo-label Generation**: Replace Mask R-CNN with YOLOv8 for 2D pedestrian detection on JRDB sequences.  
- **Cross-modal Projection**: Map detected bounding boxes from camera to LiDAR scan plane.  
- **LiDAR Model Training**: DR-SPAAM dual-branch architecture supervised by improved pseudo-labels.

### 2. Decision Stage (TD3 + Curriculum Learning)

- **State Space (32-D)**:  
  - 20-D LiDAR obstacle distances  
  - 8-D dynamic pedestrian features (2 nearest pedestrians × 4-D each)  
  - 4-D robot and goal information (distance, heading, velocity, angular velocity)  

- **Reward Function**: Multi-component dense reward combining:  
  - Goal achievement  
  - Collision penalty  
  - Velocity maintenance  
  - Smoothness  
  - Pedestrian avoidance  

- **TD3 Algorithm Enhancements**:  
  - Double-Q networks and delayed policy update  
  - Target action smoothing  
  - Exploration noise linear decay

- **Curriculum Learning**: Four-stage training (Stage 1–4) with weight inheritance to progressively adapt to increasing dynamic complexity and pedestrian density.

### 3. Control

- TD3 Actor outputs 2D continuous actions mapped to linear and angular velocities for `/cmd_vel` topic to control Pioneer 3DX robot in ROS/Gazebo.

## 📊 Results

| Stage | Success Rate | Collision Rate | Time Efficiency (m/step) |
|-------|-------------|----------------|-------------------------|
| Stage 1 | 60% | 0% | 0.027 |
| Stage 2 | 82% | 15% | 0.022 |
| Stage 3 | 41% | 28% | 0.016 |
| Stage 4 | 63% | 17% | 0.026 |

- AP@0.5m for LiDAR pedestrian detection improved from 0.417 → 0.566.  
- Dense reward function prevents reward hacking and encourages smooth, collision-free navigation.  
- Full system improves robot autonomy in realistic dynamic scenarios with multiple pedestrians.

![Performance Chart](assets/performance_chart.png)  
*Navigation success, collision rate, and efficiency over curriculum stages.*

## ⚙️ Installation

## 🚀 Usage

## Keywords

Robot Perception · Pedestrian Detection · 2D LiDAR · Dynamic Navigation · Reinforcement Learning · TD3 · DR-SPAAM · Curriculum Learning · ROS · Gazebo
