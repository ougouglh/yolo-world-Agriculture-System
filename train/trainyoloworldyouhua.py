"""
YOLO-World 零样本检测优化版
重点：优化文本描述 + 多描述集成 + 参数调优
"""

from ultralytics import YOLOWorld
import os
import json
from pathlib import Path

# ==================== 配置 ====================

ZEROSHOT_IMG_DIR = "/root/autodl-tmp/project/dataset/images/test_zeroshot"
OUTPUT_DIR = "/root/autodl-tmp/project/runs/yoloworld"

# 使用预训练模型 (保留更强的零样本能力)
# MODEL_PATH = "yolov8s-world.pt"  # 原始预训练
MODEL_PATH = f"{OUTPUT_DIR}/train/weights/best.pt"  # 或用微调后的

# ==================== 优化的文本描述 ⭐ ====================

# 每个类别准备多个描述，更详细的视觉特征
ZEROSHOT_PROMPTS = {
    "Apple_Cedar_apple_rust": [
        # 描述1: 强调颜色特征
        "apple leaf with bright orange spots, cedar apple rust disease",
        # 描述2: 强调形状特征
        "apple leaf with circular yellow-orange lesions, rust fungus infection",
        # 描述3: 简洁版
        "orange rust spots on apple leaf",
        # 描述4: 更详细
        "apple leaf showing cedar rust, bright orange-yellow circular spots with red border",
    ],

    "Corn_(maize)_Northern_Leaf_Blight": [
        "corn leaf with long gray-green elliptical lesions, northern leaf blight",
        "maize leaf with cigar-shaped gray spots, fungal disease",
        "corn leaf blight with elongated brown-gray lesions",
        "northern corn leaf blight, long elliptical tan-colored lesions",
    ],

    "Grape_Leaf_blight_(Isariopsis_Leaf_Spot)": [
        "grape leaf with brown irregular spots, leaf blight disease",
        "grapevine leaf with dark brown lesions and yellow halo",
        "grape leaf spot disease, brown patches with dried edges",
        "diseased grape leaf with necrotic brown spots",
    ],

    "Tomato_Tomato_Yellow_Leaf_Curl_Virus": [
        # 这个类别检测率最低，重点优化
        "tomato leaf curling upward, yellowing at edges, virus disease",
        "tomato plant with curled cupped leaves, yellow leaf curl virus",
        "tomato leaf with upward curl and yellow margins",
        "stunted tomato with small curled leaves, viral infection",
        "tomato yellow leaf curl, leaves curled upward with chlorosis",
    ],

    "Tomato_Tomato_mosaic_virus": [
        "tomato leaf with mosaic pattern, light and dark green mottling",
        "tomato mosaic virus, mottled leaves with yellow-green patches",
        "tomato leaf showing mosaic discoloration pattern",
        "virus infected tomato leaf with irregular green-yellow pattern",
    ],
}


# ==================== 测试函数 ====================

def test_with_single_prompt():
    """使用单个最佳描述测试"""
    print("\n" + "=" * 60)
    print("🔬 测试1: 单描述 + 低置信度")
    print("=" * 60)

    model = YOLOWorld(MODEL_PATH)

    # 每个类别选第一个描述
    class_names = list(ZEROSHOT_PROMPTS.keys())
    prompts = [ZEROSHOT_PROMPTS[name][0] for name in class_names]

    print("使用的描述:")
    for name, prompt in zip(class_names, prompts):
        print(f"  {name}: {prompt}")

    model.set_classes(prompts)

    # 统计
    results_stats = run_detection(model, class_names, conf=0.1)  # 降低置信度

    return results_stats


def test_with_ensemble():
    """多描述集成测试"""
    print("\n" + "=" * 60)
    print("🔬 测试2: 多描述集成")
    print("=" * 60)

    model = YOLOWorld(MODEL_PATH)

    class_names = list(ZEROSHOT_PROMPTS.keys())

    # 收集所有描述
    all_prompts = []
    prompt_to_class = {}

    for name in class_names:
        for prompt in ZEROSHOT_PROMPTS[name]:
            all_prompts.append(prompt)
            prompt_to_class[prompt] = name

    print(f"总描述数: {len(all_prompts)}")
    model.set_classes(all_prompts)

    # 检测并合并结果
    results_stats = run_detection_ensemble(model, class_names, prompt_to_class, conf=0.1)

    return results_stats


def test_with_pretrained():
    """使用预训练模型（不微调）"""
    print("\n" + "=" * 60)
    print("🔬 测试3: 原始预训练模型")
    print("=" * 60)

    model = YOLOWorld("yolov8s-world.pt")  # 原始预训练

    class_names = list(ZEROSHOT_PROMPTS.keys())
    prompts = [ZEROSHOT_PROMPTS[name][0] for name in class_names]

    model.set_classes(prompts)

    results_stats = run_detection(model, class_names, conf=0.1)

    return results_stats


def run_detection(model, class_names, conf=0.1):
    """执行检测"""
    test_images = [f for f in os.listdir(ZEROSHOT_IMG_DIR) if f.endswith('.jpg')]

    total_images = 0
    detected_images = 0
    class_detections = {name: 0 for name in class_names}
    class_totals = {name: 0 for name in class_names}

    for img_file in test_images:
        img_path = os.path.join(ZEROSHOT_IMG_DIR, img_file)
        total_images += 1

        # 判断真实类别
        true_class = None
        for name in class_names:
            if name in img_file:
                true_class = name
                class_totals[name] += 1
                break

        # 推理
        results = model.predict(
            img_path,
            conf=conf,  # 使用更低的置信度
            iou=0.45,
            imgsz=640,
            device=0,
            verbose=False,
        )

        if len(results[0].boxes) > 0:
            detected_images += 1
            if true_class:
                class_detections[true_class] += 1

        if total_images % 1000 == 0:
            print(f"   已处理 {total_images}/{len(test_images)}...")

    # 打印结果
    print(f"\n📊 结果 (conf={conf}):")
    print(f"   总检测率: {detected_images}/{total_images} ({detected_images / total_images:.2%})")

    for name in class_names:
        total = class_totals.get(name, 0)
        detected = class_detections.get(name, 0)
        rate = detected / total if total > 0 else 0
        print(f"   {name}: {detected}/{total} ({rate:.2%})")

    return {
        "total_rate": detected_images / total_images,
        "by_class": {name: class_detections[name] / class_totals[name]
                     for name in class_names if class_totals[name] > 0}
    }


def run_detection_ensemble(model, class_names, prompt_to_class, conf=0.1):
    """多描述集成检测"""
    test_images = [f for f in os.listdir(ZEROSHOT_IMG_DIR) if f.endswith('.jpg')]

    total_images = 0
    detected_images = 0
    class_detections = {name: 0 for name in class_names}
    class_totals = {name: 0 for name in class_names}

    for img_file in test_images:
        img_path = os.path.join(ZEROSHOT_IMG_DIR, img_file)
        total_images += 1

        true_class = None
        for name in class_names:
            if name in img_file:
                true_class = name
                class_totals[name] += 1
                break

        results = model.predict(
            img_path,
            conf=conf,
            iou=0.45,
            imgsz=640,
            device=0,
            verbose=False,
        )

        if len(results[0].boxes) > 0:
            detected_images += 1
            if true_class:
                class_detections[true_class] += 1

        if total_images % 1000 == 0:
            print(f"   已处理 {total_images}/{len(test_images)}...")

    print(f"\n📊 集成结果 (conf={conf}):")
    print(f"   总检测率: {detected_images}/{total_images} ({detected_images / total_images:.2%})")

    for name in class_names:
        total = class_totals.get(name, 0)
        detected = class_detections.get(name, 0)
        rate = detected / total if total > 0 else 0
        print(f"   {name}: {detected}/{total} ({rate:.2%})")

    return {"total_rate": detected_images / total_images}


# ==================== 主函数 ====================

def main():
    print("=" * 60)
    print("🚀 YOLO-World 零样本检测优化实验")
    print("=" * 60)

    # 测试1: 优化描述 + 低置信度
    results1 = test_with_single_prompt()

    # 测试2: 多描述集成
    results2 = test_with_ensemble()

    # 测试3: 原始预训练模型
    results3 = test_with_pretrained()

    # 汇总
    print("\n" + "=" * 60)
    print("📈 实验汇总")
    print("=" * 60)
    print(f"原始结果 (conf=0.25):        9.25%")
    print(f"优化描述 (conf=0.1):         {results1['total_rate']:.2%}")
    print(f"多描述集成 (conf=0.1):       {results2['total_rate']:.2%}")
    print(f"预训练模型 (conf=0.1):       {results3['total_rate']:.2%}")


if __name__ == "__main__":
    main()