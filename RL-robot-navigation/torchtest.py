import torch
import torchvision
print(f"Torch: {torch.__version__} (CUDA: {torch.version.cuda})")  # 1.7.1+cu110 (11.0)
print(f"Torchvision: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")  # True
print(f"GPU: {torch.cuda.get_device_name(0)}")  # 你的显卡名