# Copyright (c) 2022，Horizon Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets



def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train classifier.')
    parser.add_argument(
        '--dataset', '-d', type=str, required=True, help='Root directory of dataset'
    )
    parser.add_argument(
        '--outdir', '-o', type=str, default='./calibration_data/', help='Output directory'
    )
    args = parser.parse_args()
    return args

def main(args):
  # 数据预处理，与模型训练保持一致
  preprocess = transforms.Compose(
    [transforms.Resize((224,224)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
  # 创建数据集
  dataset = datasets.ImageFolder(args.dataset, transform=preprocess)

  if not os.path.exists(args.outdir):
     os.makedirs(args.outdir)
  # 保存校准数据
  for i in range(len(dataset)):
      img, label = dataset[i]
      img.squeeze().numpy().tofile(args.outdir + '/' + str(i) + "_.rgb")
      

if __name__ == '__main__':
  args = get_args()
  main(args)
