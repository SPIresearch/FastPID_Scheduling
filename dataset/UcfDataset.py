import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import cv2

class UCF101Dataset(Dataset):
    def __init__(self, mode='train', split='01'):
        """
        Args:
            root_path: UCF101数据集根目录，包含frames和flow子目录
            annotation_path: 包含txt文件的标注目录
            mode: 'train' or 'test'
            split: '01', '02', or '03'
        """
        root_path = 'C:/Users/SPI/Desktop/OGM-GE_CVPR2022-main/UCF101'
        annotation_path = 'C:/Users/SPI/Desktop/OGM-GE_CVPR2022-main/UCF101/annotation'
        invalid_list_file = 'C:/Users/SPI/Desktop/OGM-GE_CVPR2022-main/UCF101/invalid_videos_all_splits.txt'

        with open(invalid_list_file, 'r') as f:
            self.invalid_videos = set(line.strip() for line in f)
        self.root_path = Path(root_path)
        self.mode = mode
        
        # 读取类别映射
        class_file = os.path.join(annotation_path, 'classInd.txt')
        self.class_dict = {}
        with open(class_file, 'r') as f:
            for line in f:
                class_id, class_name = line.strip().split()
                self.class_dict[class_name] = int(class_id) - 1  # 转换为0-based索引
        
        # 读取数据列表
        if mode == 'train':
            list_file = os.path.join(annotation_path, f'trainlist{split}.txt')
        else:
            list_file = os.path.join(annotation_path, f'testlist{split}.txt')
            
        self.samples = []
        with open(list_file, 'r') as f:
            for line in f:
                if mode == 'train':
                    video_path, label = line.strip().split()
                else:
                    video_path = line.strip()
                if video_path in self.invalid_videos:
                    continue
                
                action_name = video_path.split('/')[0]
                video_name = video_path.split('/')[1].split('.')[0]
                
                frames_dir = self.root_path / 'frames' / action_name / video_name
                flow_dir = self.root_path / 'flow' / action_name / video_name
                
                if frames_dir.exists() and flow_dir.exists():
                    self.samples.append({
                        'frames_dir': frames_dir,
                        'flow_dir': flow_dir,
                        'label': self.class_dict[action_name]
                    })

        # 定义图像转换
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 读取并处理帧
        frames = []
        for i in range(1, 4):  # 读取3帧
            frame_path = sample['frames_dir'] / f'frame_{i:03d}.jpg'
            frame = Image.open(frame_path).convert('RGB')
            frame = self.transform(frame)
            frames.append(frame)
        
        # 将帧堆叠在一起 (3, 3, 224, 224)
        frames = torch.stack(frames, dim=1)  # (3, 3, 224, 224)
        frames = torch.permute(frames, (1, 0, 2, 3))
        
        # 读取并处理光流
        flows = []
        for i in range(1, 4):  # 读取3个光流
            flow_path = sample['flow_dir'] / f'flow_{i:03d}.npy'
            flow = np.load(flow_path)  # (224, 224, 2)
            flow_resized = np.zeros((224, 224, 2))
            for c in range(2):
                flow_resized[:, :, c] = cv2.resize(flow[:, :, c], (224, 224))
            flow = torch.from_numpy(flow_resized).float()
            flow = flow.permute(2, 0, 1)  # (2, 224, 224)
            flows.append(flow)
        
        # 将光流堆叠在一起 (3, 2, 224, 224)
        flows = torch.stack(flows, dim=0)  # (3, 2, 224, 224)
        flows = torch.permute(flows, (1, 0, 2, 3))

        return idx, flows, frames, sample['label']
        
        # return {
        #     'frames': frames,  # (3, 3, 224, 224)
        #     'flows': flows,    # (3, 2, 224, 224)
        #     'label': sample['label']
        # }

# # 使用示例
# if __name__ == "__main__":
#     root_path = 'C:/Users/SPI/Desktop/OGM-GE_CVPR2022-main/UCF101'
#     annotation_path = 'C:/Users/SPI/Desktop/OGM-GE_CVPR2022-main/UCF101/annotation'
    
#     dataset = UCF101Dataset(
#         # root_path=root_path,
#         # annotation_path=annotation_path,
#         mode='train',
#         split='01'
#     )
    
#     # 测试一个样本
#     flows, frames, label = dataset[0]
#     print("Frames shape:", frames.shape)  # 应该是(3, 3, 224, 224)
#     print("Flows shape:", flows.shape)    # 应该是(3, 2, 224, 224)
#     print("Label:", label)