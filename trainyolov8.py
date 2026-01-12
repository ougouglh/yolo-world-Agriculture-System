"""
YOLOv8 训练脚本
用途：作为 baseline 对比实验
"""

from ultralytics import YOLO
import os
import torch

# ==================== 配置区 ====================

# 路径配置
DATASET_YAML = "/root/autodl-tmp/project/dataset/data.yaml"
OUTPUT_DIR = "/root/autodl-tmp/project/runs/yolov8"

# 训练配置
MODEL_SIZE = "yolov8s.pt"  # 可选: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt
EPOCHS = 100
BATCH_SIZE = 16  # 根据显存调整，16GB显存用16，8GB用8
IMG_SIZE = 640
DEVICE = 0  # GPU编号，多卡可用 "0,1"


# ==================== 检查环境 ====================

def check_environment():
    print("=" * 60)
    print("🔍 环境检查")
    print("=" * 60)

    # 检查 PyTorch
    print(f"   PyTorch 版本: {torch.__version__}")
    print(f"   CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA 版本: {torch.version.cuda}")
        print(f"   GPU 数量: {torch.cuda.device_count()}")
        print(f"   GPU 名称: {torch.cuda.get_device_name(0)}")
        print(f"   GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    # 检查数据集
    if os.path.exists(DATASET_YAML):
        print(f"   ✅ 数据集配置: {DATASET_YAML}")
    else:
        print(f"   ❌ 数据集配置不存在: {DATASET_YAML}")
        return False

    return True


# ==================== 训练函数 ====================

def train_yolov8():
    print("\n" + "=" * 60)
    print("🚀 开始训练 YOLOv8")
    print("=" * 60)

    # 加载预训练模型
    print(f"\n📦 加载预训练模型: {MODEL_SIZE}")
    model = YOLO(MODEL_SIZE)

    # 开始训练
    print(f"\n🏋️ 训练配置:")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Image Size: {IMG_SIZE}")
    print(f"   Device: {DEVICE}")

    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project=OUTPUT_DIR,
        name="train",
        exist_ok=True,

        # 优化器配置
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        # 数据增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,

        # 其他
        patience=20,  # 早停耐心值
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("✅ YOLOv8 训练完成!")
    print("=" * 60)
    print(f"   模型保存位置: {OUTPUT_DIR}/train/weights/best.pt")

    return results


# ==================== 验证函数 ====================

def validate_yolov8():
    print("\n" + "=" * 60)
    print("📊 验证 YOLOv8 模型")
    print("=" * 60)

    best_model_path = f"{OUTPUT_DIR}/train/weights/best.pt"

    if not os.path.exists(best_model_path):
        print(f"   ❌ 模型不存在: {best_model_path}")
        return None

    model = YOLO(best_model_path)

    # 在验证集上评估
    results = model.val(
        data=DATASET_YAML,
        split="val",
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        plots=True,
        verbose=True,
    )

    print("\n📈 验证结果:")
    print(f"   mAP@50: {results.box.map50:.4f}")
    print(f"   mAP@50-95: {results.box.map:.4f}")
    print(f"   Precision: {results.box.mp:.4f}")
    print(f"   Recall: {results.box.mr:.4f}")

    return results


# ==================== 零样本测试（YOLOv8无法做，仅记录） ====================

def test_zeroshot_yolov8():
    """
    YOLOv8 无法进行零样本检测！
    这个函数仅用于记录这一事实，作为对比实验的证据。
    """
    print("\n" + "=" * 60)
    print("🔬 零样本测试 (YOLOv8)")
    print("=" * 60)
    print("   ⚠️ YOLOv8 是闭集检测模型")
    print("   ⚠️ 无法识别训练时未见过的类别")
    print("   ⚠️ 零样本测试集 (5类) 的 mAP = 0")
    print("\n   这正是 YOLO-World 的优势所在！")
    print("=" * 60)


# ==================== 主函数 ====================

def main():
    print("=" * 60)
    print("🌱 PlantVillage 病虫害检测 - YOLOv8 Baseline")
    print("=" * 60)

    # 1. 检查环境
    if not check_environment():
        print("❌ 环境检查失败，请先配置环境")
        return

    # 2. 训练模型
    train_yolov8()

    # 3. 验证模型
    validate_yolov8()

    # 4. 零样本测试说明
    test_zeroshot_yolov8()

    print("\n" + "=" * 60)
    print("🎉 YOLOv8 Baseline 实验完成!")
    print("   下一步: 运行 train_yoloworld.py 进行 YOLO-World 训练")
    print("=" * 60)


if __name__ == "__main__":
    main()