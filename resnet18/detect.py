import argparse
import torchvision
import torch
import PIL.Image
from PIL import ImageDraw
import torchvision.transforms as transforms
import os
import glob

def detect(weights, source, output):
    # 加载模型
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, 2)
    model.load_state_dict(torch.load(weights))
    model.eval()
    device = torch.device('cpu')
    model = model.to(device)
    
    # 创建输出目录（如果不存在）
    os.makedirs(output, exist_ok=True)
    
    # 判断source是文件还是文件夹
    if os.path.isfile(source):
        process_single_image(model, source, output)
    elif os.path.isdir(source):
        process_directory(model, source, output)
    else:
        print(f"错误：{source} 不是有效的文件或目录")

def process_single_image(model, image_path, output_dir):
    """处理单个图像文件"""
    try:
        # 读取图像
        image_raw = PIL.Image.open(image_path)
        width, height = image_raw.size
        
        # 预处理
        color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)
        image = color_jitter(image_raw)
        image = transforms.functional.resize(image, (224, 224))
        image = transforms.functional.to_tensor(image)
        image = image.numpy().copy()
        image = torch.from_numpy(image)
        image = transforms.functional.normalize(image,
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        image = image.unsqueeze(dim=0)
        
        # 预测
        with torch.no_grad():
            pred = model(image)
            pred = torch.squeeze(pred)
        
        # 转换坐标
        x = int((pred[0] * 112 + 112) * width / 224.0)
        y = int(height - (pred[1] * 112 + 112))
        print(f"图像: {os.path.basename(image_path)}, 预测坐标: ({x}, {y})")
        
        # 绘制标记
        imagedraw = ImageDraw.Draw(image_raw)
        # 使用矩形代替嵌套循环，提高效率
        imagedraw.rectangle([x-5, y-5, x+5, y+5], outline=(255, 0, 0), width=2)
        
        # 保存结果
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        image_raw.save(output_path)
        print(f"已保存结果到: {output_path}")
        
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {str(e)}")

def process_directory(model, input_dir, output_dir):
    """处理目录中的所有图像文件"""
    print(f"开始处理目录: {input_dir}")
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    
    # 获取所有图像文件
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    
    print(f"找到 {len(image_paths)} 张图像")
    
    # 逐个处理图像
    for i, img_path in enumerate(image_paths):
        print(f"正在处理 [{i+1}/{len(image_paths)}]: {os.path.basename(img_path)}")
        process_single_image(model, img_path, output_dir)
    
    print(f"目录处理完成，结果保存在: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='best_line_follower_model_xy.pt', help='模型权重文件路径')
    parser.add_argument('--source', type=str, default='images', help='输入图像或目录')
    parser.add_argument('--output', type=str, default='output', help='输出目录')
    args = parser.parse_args()

    with torch.no_grad():
        detect(args.weights, args.source, args.output)
