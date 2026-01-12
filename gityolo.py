#!/bin/bash
# ==========================================
# 环境安装脚本
# 用途：安装 YOLOv8 和 YOLO-World 所需的依赖
# ==========================================

echo "=========================================="
echo "🔧 安装 YOLO 训练环境"
echo "=========================================="

# 1. 升级 pip
echo ""
echo "📦 升级 pip..."
pip install --upgrade pip -q

# 2. 安装 ultralytics (包含 YOLOv8 和 YOLO-World)
echo ""
echo "📦 安装 ultralytics..."
pip install ultralytics -q

# 3. 安装其他依赖
echo ""
echo "📦 安装其他依赖..."
pip install opencv-python-headless -q
pip install matplotlib -q
pip install pandas -q
pip install seaborn -q
pip install tqdm -q

# 4. 验证安装
echo ""
echo "=========================================="
echo "🔍 验证安装"
echo "=========================================="

python3 << 'EOF'
import sys
print(f"Python 版本: {sys.version}")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("❌ PyTorch 未安装")

try:
    import ultralytics
    print(f"✅ Ultralytics: {ultralytics.__version__}")
except ImportError:
    print("❌ Ultralytics 未安装")

try:
    from ultralytics import YOLO, YOLOWorld
    print("✅ YOLO 和 YOLO-World 可用")
except ImportError as e:
    print(f"❌ YOLO 导入失败: {e}")

try:
    import cv2
    print(f"✅ OpenCV: {cv2.__version__}")
except ImportError:
    print("❌ OpenCV 未安装")

print("\n========================================")
print("🎉 环境安装完成!")
print("========================================")
print("下一步:")
print("  1. 运行 python train_yolov8.py 训练 YOLOv8")
print("  2. 运行 python train_yoloworld.py 训练 YOLO-World")
print("========================================")