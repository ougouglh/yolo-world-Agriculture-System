"""
PlantVillage 数据处理脚本
功能：筛选5种作物，划分训练/验证/零样本测试集，生成YOLO格式数据
"""

import os
import shutil
import json
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ==================== 配置区 ====================

# 路径配置
SOURCE_DIR = "/root/autodl-tmp/project/PlantVillage-Dataset/raw/color"
OUTPUT_DIR = "/root/autodl-tmp/project/dataset"

# 图片配置
TARGET_SIZE = (640, 640)
TRAIN_RATIO = 0.8  # 训练集比例

# 随机种子（保证可复现）
RANDOM_SEED = 42

# 零样本测试类别（5个）
ZEROSHOT_CLASSES = [
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Apple___Cedar_apple_rust",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
]

# 目标作物前缀
TARGET_CROPS = ["Apple", "Corn", "Grape", "Potato", "Tomato"]

# 类别文本描述（用于YOLO-World）
CLASS_DESCRIPTIONS = {
    # Apple
    "Apple___Apple_scab": "apple leaf with scab disease, dark olive-green spots, velvety texture",
    "Apple___Black_rot": "apple leaf with black rot, dark brown lesions with purple margins",
    "Apple___Cedar_apple_rust": "apple leaf with cedar rust, bright orange-yellow spots, circular lesions",
    "Apple___healthy": "healthy apple leaf, green color, no disease spots",

    # Corn
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "corn leaf with gray leaf spot, rectangular gray-brown lesions",
    "Corn_(maize)___Common_rust_": "corn leaf with common rust, small reddish-brown pustules",
    "Corn_(maize)___Northern_Leaf_Blight": "corn leaf with northern leaf blight, long elliptical gray-green lesions",
    "Corn_(maize)___healthy": "healthy corn leaf, green color, no disease",

    # Grape
    "Grape___Black_rot": "grape leaf with black rot, brown circular spots with dark borders",
    "Grape___Esca_(Black_Measles)": "grape leaf with esca disease, tiger-stripe pattern, interveinal discoloration",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "grape leaf with leaf blight, brown irregular spots",
    "Grape___healthy": "healthy grape leaf, green color, no disease",

    # Potato
    "Potato___Early_blight": "potato leaf with early blight, dark brown spots with concentric rings",
    "Potato___Late_blight": "potato leaf with late blight, water-soaked lesions, white mold",
    "Potato___healthy": "healthy potato leaf, green color, no disease",

    # Tomato
    "Tomato___Bacterial_spot": "tomato leaf with bacterial spot, small dark brown spots with yellow halos",
    "Tomato___Early_blight": "tomato leaf with early blight, brown spots with concentric rings",
    "Tomato___Late_blight": "tomato leaf with late blight, water-soaked gray-green spots",
    "Tomato___Leaf_Mold": "tomato leaf with leaf mold, yellow spots on upper surface, olive-green mold below",
    "Tomato___Septoria_leaf_spot": "tomato leaf with septoria spot, small circular spots with gray centers",
    "Tomato___Spider_mites Two-spotted_spider_mite": "tomato leaf with spider mite damage, stippled yellow appearance",
    "Tomato___Target_Spot": "tomato leaf with target spot, brown lesions with concentric rings",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "tomato leaf with yellow leaf curl virus, curled leaves, yellow margins",
    "Tomato___Tomato_mosaic_virus": "tomato leaf with mosaic virus, mottled light and dark green pattern",
    "Tomato___healthy": "healthy tomato leaf, green color, no disease",
}


# ==================== 工具函数 ====================

def get_target_classes(source_dir):
    """获取目标作物的所有类别"""
    all_dirs = os.listdir(source_dir)
    target_classes = []
    for d in all_dirs:
        for crop in TARGET_CROPS:
            if d.startswith(crop):
                target_classes.append(d)
                break
    return sorted(target_classes)


def create_directory_structure(output_dir):
    """创建输出目录结构"""
    dirs = [
        "images/train", "images/val", "images/test_zeroshot",
        "labels/train", "labels/val", "labels/test_zeroshot"
    ]
    for d in dirs:
        Path(output_dir, d).mkdir(parents=True, exist_ok=True)
    print(f"✅ 创建目录结构: {output_dir}")


def process_image(src_path, dst_path, target_size):
    """处理单张图片：调整尺寸、统一格式"""
    try:
        with Image.open(src_path) as img:
            # 转换为RGB（防止PNG等格式问题）
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # 调整尺寸
            img = img.resize(target_size, Image.LANCZOS)
            # 保存为jpg
            img.save(dst_path, 'JPEG', quality=95)
        return True
    except Exception as e:
        print(f"⚠️ 处理失败 {src_path}: {e}")
        return False


def generate_label(class_id, output_path):
    """生成YOLO格式标签文件（整图标注）"""
    # YOLO格式: class_id cx cy w h (归一化坐标)
    # 整图标注: 0.5 0.5 1.0 1.0
    with open(output_path, 'w') as f:
        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")


def generate_class_name(original_name):
    """生成简化的类别名称"""
    # 例: Tomato___Early_blight -> Tomato_Early_blight
    return original_name.replace("___", "_").replace(" ", "_")


# ==================== 主处理流程 ====================

def main():
    print("=" * 60)
    print("🌱 PlantVillage 数据处理脚本")
    print("=" * 60)

    random.seed(RANDOM_SEED)

    # 1. 获取目标类别
    print("\n📋 步骤1: 获取目标类别")
    all_classes = get_target_classes(SOURCE_DIR)
    print(f"   找到 {len(all_classes)} 个类别")

    # 2. 划分训练类和零样本测试类
    print("\n📋 步骤2: 划分类别")
    train_classes = [c for c in all_classes if c not in ZEROSHOT_CLASSES]
    zeroshot_classes = [c for c in all_classes if c in ZEROSHOT_CLASSES]

    print(f"   训练/验证类别: {len(train_classes)} 个")
    print(f"   零样本测试类别: {len(zeroshot_classes)} 个")

    # 验证零样本类别是否都存在
    for zc in ZEROSHOT_CLASSES:
        if zc not in all_classes:
            print(f"   ⚠️ 警告: 零样本类别 '{zc}' 不存在于数据集中!")

    # 3. 创建目录结构
    print("\n📋 步骤3: 创建目录结构")
    create_directory_structure(OUTPUT_DIR)

    # 4. 构建类别ID映射（训练类）
    class_to_id = {cls: idx for idx, cls in enumerate(train_classes)}
    # 零样本类别ID从训练类之后开始
    for idx, cls in enumerate(zeroshot_classes):
        class_to_id[cls] = len(train_classes) + idx

    print(f"\n📋 类别ID映射:")
    print(f"   训练类 (0-{len(train_classes) - 1}): {len(train_classes)} 个")
    print(f"   零样本类 ({len(train_classes)}-{len(class_to_id) - 1}): {len(zeroshot_classes)} 个")

    # 5. 处理图片
    print("\n📋 步骤4: 处理图片")

    stats = {"train": 0, "val": 0, "test_zeroshot": 0}

    # 处理训练/验证类别
    print("\n   处理训练/验证类别...")
    for cls in tqdm(train_classes, desc="   训练类"):
        cls_dir = os.path.join(SOURCE_DIR, cls)
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # 随机打乱并划分
        random.shuffle(images)
        split_idx = int(len(images) * TRAIN_RATIO)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        class_id = class_to_id[cls]
        simple_name = generate_class_name(cls)

        # 处理训练集
        for idx, img_name in enumerate(train_images):
            src_path = os.path.join(cls_dir, img_name)
            new_name = f"{simple_name}_{idx:04d}.jpg"
            dst_img = os.path.join(OUTPUT_DIR, "images/train", new_name)
            dst_label = os.path.join(OUTPUT_DIR, "labels/train", new_name.replace('.jpg', '.txt'))

            if process_image(src_path, dst_img, TARGET_SIZE):
                generate_label(class_id, dst_label)
                stats["train"] += 1

        # 处理验证集
        for idx, img_name in enumerate(val_images):
            src_path = os.path.join(cls_dir, img_name)
            new_name = f"{simple_name}_{idx:04d}.jpg"
            dst_img = os.path.join(OUTPUT_DIR, "images/val", new_name)
            dst_label = os.path.join(OUTPUT_DIR, "labels/val", new_name.replace('.jpg', '.txt'))

            if process_image(src_path, dst_img, TARGET_SIZE):
                generate_label(class_id, dst_label)
                stats["val"] += 1

    # 处理零样本测试类别
    print("\n   处理零样本测试类别...")
    for cls in tqdm(zeroshot_classes, desc="   零样本类"):
        cls_dir = os.path.join(SOURCE_DIR, cls)
        if not os.path.exists(cls_dir):
            print(f"   ⚠️ 目录不存在: {cls_dir}")
            continue

        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        class_id = class_to_id[cls]
        simple_name = generate_class_name(cls)

        for idx, img_name in enumerate(images):
            src_path = os.path.join(cls_dir, img_name)
            new_name = f"{simple_name}_{idx:04d}.jpg"
            dst_img = os.path.join(OUTPUT_DIR, "images/test_zeroshot", new_name)
            dst_label = os.path.join(OUTPUT_DIR, "labels/test_zeroshot", new_name.replace('.jpg', '.txt'))

            if process_image(src_path, dst_img, TARGET_SIZE):
                generate_label(class_id, dst_label)
                stats["test_zeroshot"] += 1

    # 6. 生成配置文件
    print("\n📋 步骤5: 生成配置文件")

    # data.yaml
    all_class_names = [generate_class_name(c) for c in train_classes + zeroshot_classes]
    yaml_content = f"""# PlantVillage Dataset for YOLO-World
# 自动生成

path: {OUTPUT_DIR}
train: images/train
val: images/val
test: images/test_zeroshot

# 类别数量
nc: {len(train_classes)}  # 训练类别数
nc_zeroshot: {len(zeroshot_classes)}  # 零样本测试类别数

# 训练类别名称 (0-{len(train_classes) - 1})
names:
"""
    for idx, cls in enumerate(train_classes):
        yaml_content += f"  {idx}: {generate_class_name(cls)}\n"

    yaml_content += f"""
# 零样本测试类别 ({len(train_classes)}-{len(class_to_id) - 1})
names_zeroshot:
"""
    for idx, cls in enumerate(zeroshot_classes):
        yaml_content += f"  {len(train_classes) + idx}: {generate_class_name(cls)}\n"

    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"   ✅ 生成 {yaml_path}")

    # class_texts.json
    class_texts = {}
    for cls in train_classes + zeroshot_classes:
        simple_name = generate_class_name(cls)
        if cls in CLASS_DESCRIPTIONS:
            class_texts[simple_name] = CLASS_DESCRIPTIONS[cls]
        else:
            # 自动生成描述
            parts = cls.replace("___", " ").replace("_", " ").lower()
            class_texts[simple_name] = parts

    json_path = os.path.join(OUTPUT_DIR, "class_texts.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(class_texts, f, indent=2, ensure_ascii=False)
    print(f"   ✅ 生成 {json_path}")

    # 7. 生成统计报告
    print("\n" + "=" * 60)
    print("📊 处理完成！统计信息:")
    print("=" * 60)
    print(f"   训练集: {stats['train']} 张")
    print(f"   验证集: {stats['val']} 张")
    print(f"   零样本测试集: {stats['test_zeroshot']} 张")
    print(f"   总计: {sum(stats.values())} 张")
    print(f"\n   训练类别: {len(train_classes)} 个")
    print(f"   零样本类别: {len(zeroshot_classes)} 个")
    print(f"\n📁 输出目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()