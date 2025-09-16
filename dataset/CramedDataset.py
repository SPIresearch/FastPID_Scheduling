import copy
import csv
import os
import pickle
import librosa
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pdb
import random
# from diffusers import DDPMScheduler

class CramedDataset(Dataset):

    def __init__(self, args, mode='train'):
        self.args = args
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode

        self.data_root = './data/'
        class_dict = {'NEU':0, 'HAP':1, 'SAD':2, 'FEA':3, 'DIS':4, 'ANG':5}

        self.visual_feature_path = './CREMA-D/'
        self.audio_feature_path = './CREMA-D/AudioWAV'

        self.train_csv = os.path.join(self.data_root, 'CREMAD' + '/train.csv')
        self.my_train_csv = os.path.join(self.data_root, 'CREMAD' + '/my_train.csv')
        self.val_csv = os.path.join(self.data_root, 'CREMAD' + '/val.csv')
        self.test_csv = os.path.join(self.data_root, 'CREMAD' + '/test.csv')

        # self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

        if mode == 'train':
            csv_file = self.train_csv
        elif mode == "my_train":
            csv_file = self.my_train_csv
        elif mode == "val":
            csv_file = self.val_csv
        else:
            csv_file = self.test_csv

        with open(csv_file, encoding='UTF-8-sig') as f2:
            csv_reader = csv.reader(f2)
            for item in csv_reader:
                audio_path = os.path.join(self.audio_feature_path, item[0] + '.wav')
                visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS'.format(1), item[0])

                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.image.append(visual_path)
                    self.audio.append(audio_path)
                    self.label.append(class_dict[item[1]])
                else:
                    continue


    def __len__(self):
        return len(self.image)

    # def audio_augment(self, samples):
    #     # 时间拉伸
    #     stretch_factor = random.uniform(0.8, 1.2)
    #     samples = librosa.effects.time_stretch(samples, rate=stretch_factor)

    #     # 音高移位
    #     n_steps = random.randint(-4, 4)
    #     samples = librosa.effects.pitch_shift(samples, sr=22050, n_steps=n_steps)

    #     # 添加高斯噪声
    #     noise_factor = random.uniform(0, 0.05)
    #     noise = np.random.randn(len(samples))
    #     samples = samples + noise_factor * noise

    #     # 随机裁剪
    #     if len(samples) > 22050 * 3:
    #         start = random.randint(0, len(samples) - 22050 * 3)
    #         samples = samples[start:start + 22050 * 3]
    #     else:
    #         samples = np.pad(samples, (0, max(0, 22050 * 3 - len(samples))))

    #     return samples


    def __getitem__(self, idx):

        # audio
        samples, rate = librosa.load(self.audio[idx], sr=22050)

        # if self.mode == 'train':
        #     resamples = self.audio_augment(samples)
        # else:
        resamples = np.tile(samples, 3)[:22050 * 3]

        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.

        spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)

        spectrogram = torch.from_numpy(spectrogram).float()
        spectrogram = transforms.Resize((224, 224))(spectrogram.unsqueeze(0))
        spectrogram = spectrogram.squeeze(0)
        # 在spectrogram = torch.from_numpy(spectrogram).float()之后添加
        mean = spectrogram.mean()
        std = spectrogram.std()
        spectrogram = (spectrogram - mean) / (std + 1e-7)

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        image_samples = os.listdir(self.image[idx])
        select_index = np.random.choice(len(image_samples), size=1, replace=False)
        select_index.sort()
        images = torch.zeros((1, 3, 224, 224))
        for i in range(1):
            img = Image.open(os.path.join(self.image[idx], image_samples[i])).convert('RGB')
            img = transform(img)
            images[i] = img

        images = torch.permute(images, (1,0,2,3))

        # 对音频（spectrogram）加噪
        # timesteps = torch.randint(0, 999, (1,)).long()
        # noise_audio = torch.randn_like(spectrogram)  # 生成与频谱图相同形状的噪声
        # noisy_spectrogram = self.noise_scheduler.add_noise(spectrogram, noise_audio, timesteps)

        # label
        label = self.label[idx]

        return idx, spectrogram, images, label # (224, 224), (3, 224, 224)