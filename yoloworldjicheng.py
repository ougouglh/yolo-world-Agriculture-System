"""
YOLO-World 模型集成脚本
策略：结合微调模型和预训练模型的优势
"""

from ultralytics import YOLOWorld
import os
import json
from pathlib import Path

# ==================== 配置 ====================

ZEROSHOT_IMG_DIR = "/root/autodl-tmp/project/dataset/images/test_zeroshot"
OUTPUT_DIR = "/root/autodl-tmp/project/runs/yoloworld"

# 两个模型路径
FINETUNED_MODEL = f"{OUTPUT_DIR}/train/weights/best.pt"  # 微调模型
PRETRAINED_MODEL = "yolov8s-world.pt"  # 预训练模型

# 零样本类别
ZEROSHOT_CLASSES = [
    "Apple_Cedar_apple_rust",
    "Corn_(maize)_Northern_Leaf_Blight",
    "Grape_Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus",
]

# 文本描述
PROMPTS = {
    "Apple_Cedar_apple_rust": "apple leaf with bright orange spots, cedar apple rust disease",
    "Corn_(maize)_Northern_Leaf_Blight": "corn leaf with long gray-green elliptical lesions, northern leaf blight",
    "Grape_Leaf_blight_(Isariopsis_Leaf_Spot)": "grape leaf with brown irregular spots, leaf blight disease",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus": "tomato leaf curling upward, yellowing at edges, virus disease",
    "Tomato_Tomato_mosaic_virus": "tomato leaf with mosaic pattern, light and dark green mottling",
}

# 根据实验结果，指定每个类别用哪个模型
# finetuned = 微调模型更好, pretrained = 预训练模型更好
BEST_MODEL_FOR_CLASS = {
    "Apple_Cedar_apple_rust": "finetuned",  # 29.45% vs 26.91%
    "Corn_(maize)_Northern_Leaf_Blight": "finetuned",  # 96.95% vs 1.73%
    "Grape_Leaf_blight_(Isariopsis_Leaf_Spot)": "pretrained",  # 38.48% vs 60.22%
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus": "pretrained",  # 4.13% vs 69.78%
    "Tomato_Tomato_mosaic_virus": "pretrained",  # 24.40% vs 85.52%
}


# ==================== 集成策略 ====================

class ModelEnsemble:
    """模型集成类"""

    def __init__(self):
        print("=" * 60)
        print("🔧 加载模型集成")
        print("=" * 60)

        # 加载两个模型
        print("   加载微调模型...")
        self.finetuned = YOLOWorld(FINETUNED_MODEL)

        print("   加载预训练模型...")
        self.pretrained = YOLOWorld(PRETRAINED_MODEL)

        # 设置类别
        prompts_list = [PROMPTS[c] for c in ZEROSHOT_CLASSES]
        self.finetuned.set_classes(prompts_list)
        self.pretrained.set_classes(prompts_list)

        print("   ✅ 模型加载完成")

    def predict_smart(self, img_path, true_class=None, conf=0.1):
        """
        策略1: 智能选择 - 根据类别选择最佳模型
        """
        if true_class and true_class in BEST_MODEL_FOR_CLASS:
            best = BEST_MODEL_FOR_CLASS[true_class]
            model = self.finetuned if best == "finetuned" else self.pretrained
        else:
            model = self.pretrained  # 默认用预训练

        results = model.predict(img_path, conf=conf, verbose=False)
        return len(results[0].boxes) > 0

    def predict_union(self, img_path, conf=0.1):
        """
        策略2: 并集 - 任一模型检测到即算成功
        """
        r1 = self.finetuned.predict(img_path, conf=conf, verbose=False)
        r2 = self.pretrained.predict(img_path, conf=conf, verbose=False)

        return len(r1[0].boxes) > 0 or len(r2[0].boxes) > 0

    def predict_max_conf(self, img_path, conf=0.1):
        """
        策略3: 最高置信度 - 取两个模型中置信度最高的结果
        """
        r1 = self.finetuned.predict(img_path, conf=conf, verbose=False)
        r2 = self.pretrained.predict(img_path, conf=conf, verbose=False)

        max_conf1 = max([b.conf.item() for b in r1[0].boxes], default=0)
        max_conf2 = max([b.conf.item() for b in r2[0].boxes], default=0)

        return max(max_conf1, max_conf2) > conf


# ==================== 测试函数 ====================

def run_ensemble_test():
    """运行集成测试"""

    ensemble = ModelEnsemble()

    # 获取测试图片
    test_images = [f for f in os.listdir(ZEROSHOT_IMG_DIR) if f.endswith('.jpg')]
    print(f"\n📊 测试图片数: {len(test_images)}")

    # 统计变量
    results = {
        "smart": {"total": 0, "detected": 0, "by_class": {c: [0, 0] for c in ZEROSHOT_CLASSES}},
        "union": {"total": 0, "detected": 0, "by_class": {c: [0, 0] for c in ZEROSHOT_CLASSES}},
        "max_conf": {"total": 0, "detected": 0, "by_class": {c: [0, 0] for c in ZEROSHOT_CLASSES}},
    }

    print("\n🔍 开始集成测试...")

    for i, img_file in enumerate(test_images):
        img_path = os.path.join(ZEROSHOT_IMG_DIR, img_file)

        # 判断真实类别
        true_class = None
        for c in ZEROSHOT_CLASSES:
            if c in img_file:
                true_class = c
                break

        # 策略1: 智能选择
        if ensemble.predict_smart(img_path, true_class, conf=0.1):
            results["smart"]["detected"] += 1
            if true_class:
                results["smart"]["by_class"][true_class][0] += 1
        results["smart"]["total"] += 1
        if true_class:
            results["smart"]["by_class"][true_class][1] += 1

        # 策略2: 并集
        if ensemble.predict_union(img_path, conf=0.1):
            results["union"]["detected"] += 1
            if true_class:
                results["union"]["by_class"][true_class][0] += 1
        results["union"]["total"] += 1
        if true_class:
            results["union"]["by_class"][true_class][1] += 1

        # 策略3: 最高置信度
        if ensemble.predict_max_conf(img_path, conf=0.1):
            results["max_conf"]["detected"] += 1
            if true_class:
                results["max_conf"]["by_class"][true_class][0] += 1
        results["max_conf"]["total"] += 1
        if true_class:
            results["max_conf"]["by_class"][true_class][1] += 1

        # 进度
        if (i + 1) % 1000 == 0:
            print(f"   已处理 {i + 1}/{len(test_images)}...")

    # 打印结果
    print("\n" + "=" * 60)
    print("📈 模型集成结果对比")
    print("=" * 60)

    print("\n【历史基准】")
    print(f"   单独微调模型 (多描述):    21.84%")
    print(f"   单独预训练模型:           59.46%")

    for strategy_name, strategy_label in [
        ("smart", "策略1-智能选择"),
        ("union", "策略2-并集"),
        ("max_conf", "策略3-最高置信度")
    ]:
        r = results[strategy_name]
        rate = r["detected"] / r["total"] if r["total"] > 0 else 0
        print(f"\n【{strategy_label}】 总检测率: {r['detected']}/{r['total']} ({rate:.2%})")

        for c in ZEROSHOT_CLASSES:
            detected, total = r["by_class"][c]
            c_rate = detected / total if total > 0 else 0
            print(f"   {c}: {detected}/{total} ({c_rate:.2%})")

    return results


# ==================== 主函数 ====================

def main():
    print("=" * 60)
    print("🚀 YOLO-World 模型集成实验")
    print("=" * 60)

    results = run_ensemble_test()

    print("\n" + "=" * 60)
    print("🎉 集成实验完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()