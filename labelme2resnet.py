import json
import os
import shutil
import argparse
import random
import sys

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='将图像数据集按比例划分为训练集和测试集')
    parser.add_argument('--source_folder', type=str, default='track_image', help='源图像文件夹路径')
    parser.add_argument('--target_folder', type=str, default='line_follow_dataset', help='目标数据集文件夹路径')
    parser.add_argument('--train_ratio', type=float, default=0.9, help='训练集比例 (默认0.9)')
    parser.add_argument('--shuffle', action='store_true', help='是否随机打乱数据')
    parser.add_argument('--verbose', action='store_true', help='是否显示详细处理信息')
    args = parser.parse_args()

    # 源文件夹和目标文件夹路径
    source_folder = args.source_folder
    target_folder = args.target_folder
    train_ratio = args.train_ratio
    
    # 创建目标文件夹结构
    train_folder = os.path.join(target_folder, 'train')
    train_image = os.path.join(train_folder, 'image')
    train_label = os.path.join(train_folder, 'label')
    
    test_folder = os.path.join(target_folder, 'test')
    test_image = os.path.join(test_folder, 'image')
    test_label = os.path.join(test_folder, 'label')
    
    # 创建目录（如果不存在）
    for path in [target_folder, train_folder, train_image, train_label, test_folder, test_image, test_label]:
        if not os.path.exists(path):
            os.makedirs(path)
            if args.verbose:
                print(f"创建目录: {path}")
    
    # 统计源文件夹中的文件
    all_files = os.listdir(source_folder)
    json_files = [f for f in all_files if f.lower().endswith('.json')]
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in all_files if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    print(f"源文件夹: {source_folder}")
    print(f"找到 {len(json_files)} 个JSON文件")
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 匹配JSON和图像文件
    matched_pairs = []
    unmatched_jsons = []
    unmatched_images = []
    
    for json_file in json_files:
        base_name = os.path.splitext(json_file)[0]
        # 查找对应的图像文件
        matching_image = next((img for img in image_files if img.lower().startswith(base_name.lower())), None)
        if matching_image:
            matched_pairs.append((json_file, matching_image))
        else:
            unmatched_jsons.append(json_file)
    
    for image_file in image_files:
        base_name = os.path.splitext(image_file)[0]
        # 查找对应的JSON文件
        matching_json = next((json for json in json_files if json.lower().startswith(base_name.lower())), None)
        if not matching_json:
            unmatched_images.append(image_file)
    
    print(f"找到 {len(matched_pairs)} 对匹配的JSON-图像文件")
    if unmatched_jsons:
        print(f"警告: {len(unmatched_jsons)} 个JSON文件没有匹配的图像文件")
    if unmatched_images:
        print(f"警告: {len(unmatched_images)} 个图像文件没有匹配的JSON文件")
    
    # 随机打乱数据（如果需要）
    if args.shuffle:
        random.shuffle(matched_pairs)
        print("数据已随机打乱")
    
    # 计算训练集和测试集的大小
    total_pairs = len(matched_pairs)
    train_size = int(total_pairs * train_ratio)
    test_size = total_pairs - train_size
    
    print(f"数据集划分: 训练集 {train_size} 对, 测试集 {test_size} 对")
    
    # 复制文件到训练集和测试集
    train_count = 0
    test_count = 0
    error_count = 0
    
    for i, (json_file, image_file) in enumerate(matched_pairs):
        is_train = i < train_size
        source_json_path = os.path.join(source_folder, json_file)
        source_image_path = os.path.join(source_folder, image_file)
        
        # 确定目标路径
        dest_image_folder = train_image if is_train else test_image
        dest_label_folder = train_label if is_train else test_label
        
        # 获取基本文件名（不带扩展名）
        base_name = os.path.splitext(json_file)[0]
        
        try:
            # 读取JSON文件并提取点坐标
            with open(source_json_path, 'r') as f:
                data = json.load(f)
            
            points = []
            shape_count = 0
            
            for shape in data.get('shapes', []):
                if shape.get('shape_type') == 'point':
                    shape_count += 1
                    if shape_count > 1:
                        print(f"警告: {json_file} 包含多个点标记，仅使用第一个")
                    
                    # 提取点坐标
                    point = shape.get('points', [[]])[0]  # 格式: [[x, y]]
                    if len(point) >= 2:
                        points.append((float(point[0]), float(point[1])))
                    
                    # 只处理第一个点标记
                    if shape_count == 1:
                        break
            
            if not points:
                print(f"错误: {json_file} 中未找到点标记，跳过")
                error_count += 1
                continue
            
            # 复制图像文件
            dest_image_path = os.path.join(dest_image_folder, image_file)
            shutil.copy2(source_image_path, dest_image_path)
            
            # 创建标签文件
            dest_label_path = os.path.join(dest_label_folder, f"{base_name}.txt")
            with open(dest_label_path, 'w') as f:
                # 写入点坐标（取整）
                f.write(f"{int(points[0][0])} {int(points[0][1])}")
            
            if args.verbose:
                dataset_type = "训练集" if is_train else "测试集"
                print(f"[{dataset_type}] 复制: {image_file} -> {dest_image_path}")
                print(f"[{dataset_type}] 创建标签: {base_name}.txt")
            
            if is_train:
                train_count += 1
            else:
                test_count += 1
                
        except Exception as e:
            print(f"处理文件 {json_file} 时出错: {str(e)}")
            error_count += 1
    
    # 输出结果统计
    print("\n===== 处理结果 =====")
    print(f"成功处理: {train_count} 对训练数据, {test_count} 对测试数据")
    print(f"处理失败: {error_count} 对")
    print(f"训练集路径: {train_image} 和 {train_label}")
    print(f"测试集路径: {test_image} 和 {test_label}")
    print("=" * 40)
    print(f"数据集准备完成! 总数据: {train_count + test_count}/{total_pairs}")

if __name__ == "__main__":
    main()    
