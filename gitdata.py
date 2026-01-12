# explore_data.py
import os
from pathlib import Path


def find_plantvillage_data():
    """查找PlantVillage图像数据"""
    print("🔍 探索PlantVillage数据集结构")
    print("=" * 50)

    base_path = "/root/autodl-tmp/project/PlantVillage-Dataset"

    if not os.path.exists(base_path):
        print("❌ 数据集不存在")
        return

    print(f"📁 数据集根目录: {base_path}")

    # 查找可能的图像数据目录
    possible_dirs = []
    for root, dirs, files in os.walk(base_path):
        # 寻找包含图像文件的目录
        img_count = 0
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                img_count += 1

        if img_count > 10:  # 如果有较多图片
            possible_dirs.append((root, img_count))

    if possible_dirs:
        print("\n✅ 找到图像数据目录:")
        for dir_path, count in sorted(possible_dirs, key=lambda x: x[1], reverse=True)[:10]:
            rel_path = os.path.relpath(dir_path, base_path)
            print(f"📂 {rel_path}/ - {count}张图片")

            # 显示前几个文件
            files = [f for f in os.listdir(dir_path)
                     if f.lower().endswith(('.jpg', '.png'))][:3]
            if files:
                print(f"    示例: {files}")
    else:
        print("\n🔎 搜索所有目录...")
        all_dirs = []
        for root, dirs, files in os.walk(base_path):
            all_dirs.append(root)

        print(f"总目录数: {len(all_dirs)}")
        for dir_path in all_dirs[:20]:  # 显示前20个
            rel_path = os.path.relpath(dir_path, base_path)
            print(f"  {rel_path}/")

        # 检查常见目录名
        print("\n🔎 检查常见作物目录...")
        for dir_path in all_dirs:
            dir_name = os.path.basename(dir_path)
            if any(crop in dir_name.lower() for crop in ['tomato', 'apple', 'corn', 'grape', 'potato']):
                print(f"🌱 找到可能的数据目录: {dir_name}")
                files = os.listdir(dir_path)[:3]
                print(f"    文件示例: {files}")


def check_structure():
    """检查已知的目录结构"""
    print("\n🔍 检查已知目录结构")
    print("=" * 50)

    base_path = "/root/autodl-tmp/project/PlantVillage-Dataset"

    # 常见的数据集结构
    common_structures = [
        os.path.join(base_path, "raw", "color"),
        os.path.join(base_path, "Plant_leave_diseases_dataset_without_augmentation"),
        os.path.join(base_path, "plantvillage_dataset"),
        os.path.join(base_path, "data"),
    ]

    for path in common_structures:
        if os.path.exists(path):
            print(f"\n✅ 找到已知结构: {os.path.relpath(path, base_path)}")
            # 列出子目录
            try:
                subdirs = [d for d in os.listdir(path)
                           if os.path.isdir(os.path.join(path, d))]
                print(f"   包含 {len(subdirs)} 个子目录")

                # 显示作物相关的目录
                crop_dirs = []
                for d in subdirs:
                    if any(crop in d.lower() for crop in ['tomato', 'apple', 'corn', 'grape', 'potato']):
                        crop_dirs.append(d)

                if crop_dirs:
                    print(f"   作物目录: {crop_dirs[:10]}")
                    if len(crop_dirs) > 10:
                        print(f"   ... 还有 {len(crop_dirs) - 10} 个")
            except Exception as e:
                print(f"   读取错误: {e}")


def count_total_images():
    """统计总图片数"""
    print("\n📊 统计图片数量")
    print("=" * 50)

    base_path = "/root/autodl-tmp/project/PlantVillage-Dataset"

    total_images = 0
    crop_stats = {}

    for root, dirs, files in os.walk(base_path):
        # 统计图片文件
        img_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if img_files:
            total_images += len(img_files)
            dir_name = os.path.basename(root)

            # 按作物分类
            for crop in ['tomato', 'apple', 'corn', 'grape', 'potato']:
                if crop in dir_name.lower():
                    crop_stats[crop] = crop_stats.get(crop, 0) + len(img_files)
                    break

    print(f"🌐 总图片数: {total_images}")

    if crop_stats:
        print("\n🌱 按作物分类:")
        for crop, count in sorted(crop_stats.items()):
            print(f"  {crop.capitalize()}: {count}张")
    else:
        print("⚠️  未找到常见作物图片")


def main():
    print("PlantVillage数据集结构分析")
    print("=" * 50)

    find_plantvillage_data()
    check_structure()
    count_total_images()

if __name__ == "__main__":
    main()