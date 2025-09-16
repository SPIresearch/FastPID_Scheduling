import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import resnet18
from .fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion
from transformers import BertModel

import torch
import torch.nn as nn


import torch
import torch.nn as nn
import math

class MLPHead(nn.Module):
    """
    一个简单的多层感知机(MLP)投影头，用于增强特征变换。
    结构: Linear -> Activation -> Dropout -> Linear
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU() # GELU是Transformer中常用的激活函数
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.fc3 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class Transformer_Encoder(nn.Module):
    """Transformer编码器"""
    
    def __init__(self, input_dim, dim, n_head, n_layers):
        """初始化Transformer编码器
        
        Args:
            input_dim (int): 输入特征维度
            dim (int): 嵌入维度/隐藏层大小
            n_head (int): 注意力头数量
            n_layers (int): Transformer层数
        """
        super().__init__()
        self.embed_dim = dim
        self.conv = nn.Conv1d(input_dim, self.embed_dim,
                              kernel_size=1, padding=0, bias=False)
        layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=n_head)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x):
        """应用Transformer到输入
        
        Args:
            x (torch.Tensor): 输入张量 [batch_size, seq_len, input_dim]
            
        Returns:
            torch.Tensor: 输出张量 [batch_size, embed_dim]
        """
        if type(x) is list:
            x = x[0]
        x = self.conv(x.permute([0, 2, 1]))  # [batch_size, embed_dim, seq_len]
        x = x.permute([2, 0, 1])  # [seq_len, batch_size, embed_dim]
        x = self.transformer(x)[-1]  # 取最后一个token的表示
        return x


class TextImageClassifier(nn.Module):
    """文本-图像多模态分类器"""
    
    def __init__(self, args):
        super(TextImageClassifier, self).__init__()
        
        # 配置参数
        fusion = args.fusion_method
        self.use_extra_linear = args.extra_linear
        self.dataset = args.dataset
        self.batch_size = args.batch_size

        self.text_projection_head = MLPHead(
            input_dim=512, 
            hidden_dim=512 * 2, 
            output_dim=64
        )
        
        # 设置输入维度
        if self.use_extra_linear:
            input_dim = 128
            self.extra_linear_text = nn.Linear(64, 64)  # 文本特征已经是64维
            self.extra_linear_visual = nn.Linear(512, 64)  # 图像特征是512维
        else:
            input_dim = 512 + 64  # 图像特征512维 + 文本特征64维
        
        # 设置类别数
        if args.dataset == 'FOOD101':
            n_classes = 101
        else:
            raise NotImplementedError('本分类器仅支持FOOD101数据集')
        self.n_class = n_classes
        
        # 设置融合模块
        if fusion == 'sum':
            self.fusion_module = SumFusion(input_dim=input_dim//2, output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(input_dim=input_dim, output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(input_dim=input_dim, output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(input_dim=input_dim, output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))
        
        # 设置文本处理网络
        self.bert = BertModel.from_pretrained('./utils/bert')
        # 冻结BERT参数以加快训练（可选）
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        
        # 添加Transformer编码器处理BERT输出
        self.text_transformer = Transformer_Encoder(
            input_dim=768,  # BERT输出维度
            dim=512,        # 目标维度
            n_head=8,
            n_layers=4
        )
        
        # 设置图像处理网络
        self.visual_net = resnet18(modality='visual', pretrain=True)
        
        # 设置分类头
        if self.use_extra_linear:
            self.head_text = nn.Linear(64, n_classes)
            self.head_visual = nn.Linear(64, n_classes)
        else:
            self.head_text = nn.Linear(64, n_classes)
            self.head_visual = nn.Linear(512, n_classes)
    
    def get_embeddings(self, text_input_ids, attention_mask, visual):
        """
        获取文本和图像的嵌入表示
        
        Args:
            text_input_ids (torch.Tensor): 文本输入ID [batch_size, seq_len]
            attention_mask (torch.Tensor): 注意力掩码 [batch_size, seq_len]
            visual (torch.Tensor): 图像输入 [batch_size, 3, H, W]
            
        Returns:
            tuple: (文本特征, 图像特征)
        """
        # 处理文本
        bert_outputs = self.bert(input_ids=text_input_ids, attention_mask=attention_mask)
        text_features = bert_outputs.last_hidden_state  # [batch_size, seq_len, 768]
        text_features = self.text_transformer(text_features)  # [batch_size, 64]
        
        # 处理图像
        visual_features = self.visual_net(visual)  # [batch_size, 512, H', W']
        visual_features = F.adaptive_avg_pool2d(visual_features, 1)  # [batch_size, 512, 1, 1]
        visual_features = torch.flatten(visual_features, 1)  # [batch_size, 512]
        
        # 应用额外的线性层（如果启用）
        if self.use_extra_linear:
            text_features = self.text_projection_head(text_features)
            visual_features = self.extra_linear_visual(visual_features)
        
        return text_features, visual_features
    
    def forward(self, text_input_ids, attention_mask, visual):
        """
        前向传播
        
        Args:
            text_input_ids (torch.Tensor): 文本输入ID [batch_size, seq_len]
            attention_mask (torch.Tensor): 注意力掩码 [batch_size, seq_len]
            visual (torch.Tensor): 图像输入 [batch_size, 3, H, W]
            
        Returns:
            tuple: (文本分类结果, 图像分类结果, 融合分类结果)
        """
        # 获取特征
        text_features, visual_features = self.get_embeddings(text_input_ids, attention_mask, visual)
        
        # 特征融合
        text_features, visual_features, out = self.fusion_module(text_features, visual_features)
        
        # 分类预测
        out_text = self.head_text(text_features)
        out_visual = self.head_visual(visual_features)
        
        return out_text, out_visual, out

class SingleModalityClassifier(nn.Module):
    def __init__(self, args):
        super(SingleModalityClassifier, self).__init__()
        self.modality = args.modality
        self.use_extra_linear = args.extra_linear

        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 400
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if self.modality == 'audio':
            self.backbone = resnet18(modality='audio')
        elif self.modality == 'visual':
            self.backbone = resnet18(modality='visual')
        else:
            raise ValueError(f"Invalid modality: {self.modality}")

        if self.use_extra_linear:
            self.extra_linear = nn.Linear(512, 64)
            self.fc = nn.Linear(64, n_classes)
        else:
            self.fc = nn.Linear(512, n_classes)

    def get_embeddings(self, x):
        features = self.backbone(x)

        if self.modality == 'audio':
            features = F.adaptive_avg_pool2d(features, 1)
        elif self.modality == 'visual':
            (B, C, H, W) = features.size()
            features = features.view(B, -1, C, H, W)
            features = features.permute(0, 2, 1, 3, 4)
            features = F.adaptive_avg_pool3d(features, 1)

        features = torch.flatten(features, 1)

        if self.use_extra_linear:
            features = self.extra_linear(features)
        return features

    def forward(self, x):
        # 原始的前向传播
        features = self.get_embeddings(x)
        x = self.fc(features)  # 最后的分类层
        return x

class AVClassifier(nn.Module):
    def __init__(self, args):
        super(AVClassifier, self).__init__()

        fusion = args.fusion_method
        self.use_extra_linear = args.extra_linear
        self.dataset = args.dataset
        self.batch_size = args.batch_size

        if self.use_extra_linear:
            input_dim = 128
            self.extra_linear_audio = nn.Linear(512, 64)
            self.extra_linear_visual = nn.Linear(512, 64)
        else:
            input_dim = 1024

        # 设置类别数
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 400
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))
        self.n_class = n_classes

        # 设置融合模块
        if fusion == 'sum':
            self.fusion_module = SumFusion(input_dim=input_dim//2, output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(input_dim=input_dim, output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(input_dim=input_dim, output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(input_dim=input_dim, output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        # 设置网络
        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')
        self.modality = 'audio'

        # 设置分类头
        if self.use_extra_linear:
            self.head_audio = nn.Linear(64, n_classes)
            self.head_video = nn.Linear(64, n_classes)
        else:
            self.head_audio = nn.Linear(512, n_classes)
            self.head_video = nn.Linear(512, n_classes)

    def get_embeddings(self, x1, visual):
        """
        x1: 音频输入
        visual: 视觉输入
        """
        a = self.audio_net(x1)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        if self.use_extra_linear:
            a = self.extra_linear_audio(a)
            v = self.extra_linear_visual(v)

        return a, v

    def forward(self, x1, v):
        # 获取特征
        x1, v = self.get_embeddings(x1, v)
        
        # 特征融合
        x1, v, out = self.fusion_module(x1, v)
        
        # 分类预测
        out_audio = self.head_audio(x1)
        out_video = self.head_video(v)
        return out_audio, out_video, out

class FVClassifier(nn.Module):
    def __init__(self, args):
        super(FVClassifier, self).__init__()

        fusion = args.fusion_method
        self.use_extra_linear = args.extra_linear
        self.dataset = args.dataset
        self.batch_size = args.batch_size

        if self.use_extra_linear:
            input_dim = 128
            self.extra_linear_flow = nn.Linear(512, 64)
            self.extra_linear_visual = nn.Linear(512, 64)
        else:
            input_dim = 1024

        n_classes = 101
        self.n_class = n_classes

        # 设置融合模块
        if fusion == 'sum':
            self.fusion_module = SumFusion(input_dim=input_dim//2, output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(input_dim=input_dim, output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(input_dim=input_dim, output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(input_dim=input_dim, output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        # 设置网络
        self.visual_net = resnet18(modality='visual')
        self.flow_net = resnet18(modality='flow')

        # 设置分类头
        if self.use_extra_linear:
            self.head_flow= nn.Linear(64, n_classes)
            self.head_video = nn.Linear(64, n_classes)
        else:
            self.head_flow = nn.Linear(512, n_classes)
            self.head_video = nn.Linear(512, n_classes)

    def get_embeddings(self, x1, visual):
        """
        x1: 音频输入
        visual: 视觉输入
        """
        B = x1.size()[0]
        f = self.flow_net(x1)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        

        f = f.view(B, -1, C, H, W)
        f = f.permute(0, 2, 1, 3, 4)

        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        f = F.adaptive_avg_pool3d(f, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        f = torch.flatten(f, 1)
        v = torch.flatten(v, 1)

        if self.use_extra_linear:
            f = self.extra_linear_flow(f)
            v = self.extra_linear_visual(v)

        return f, v

    def forward(self, x1, v):
        # 获取特征
        x1, v = self.get_embeddings(x1, v)
        
        # 特征融合
        x1, v, out = self.fusion_module(x1, v)
        
        # 分类预测
        out_flow = self.head_flow(x1)
        out_video = self.head_video(v)
        
        return out_flow, out_video, out


class UnimodalEvaluator(nn.Module):
    def __init__(self, encoder, modality, n_classes):
        super(UnimodalEvaluator, self).__init__()
        self.modality = modality

        # 冻结编码器参数
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # 新的分类头
        self.classifier = nn.Linear(512, n_classes)

    def forward(self, x):
        with torch.no_grad():
            if self.modality == 'audio':
                feat = self.encoder(x)
                feat = F.adaptive_avg_pool2d(feat, 1)
            else:  # visual
                feat = self.encoder(x)
                B, C, H, W = feat.size()
                feat = feat.view(B, -1, C, H, W)
                feat = feat.permute(0, 2, 1, 3, 4)
                feat = F.adaptive_avg_pool3d(feat, 1)

            feat = torch.flatten(feat, 1)

        out = self.classifier(feat)
        return out

class convnet(nn.Module):
    def __init__(self, num_classes=10, modal='gray'):
        super(convnet, self).__init__()

        self.modal = modal

        if modal == 'grey':
            in_channel = 1
        elif modal == 'colored':
            in_channel = 3
        else:
            raise ValueError('non exist modal')
        self.bn0 = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(64, 512)

    def forward(self, x):
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.relu(x)  # 28x28
        x = self.maxpool(x)  # 14x14

        x = self.conv2(x)
        x = self.relu(x)  # 14x14
        x = self.conv3(x)
        x = self.relu(x)  # 7x7
        x = self.conv4(x)
        x = self.relu(x)  # 7x7

        feat = x
        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)
        feat = self.fc(feat)

        return feat

class CGClassifier(nn.Module):
    def __init__(self, args):
        super(CGClassifier, self).__init__()

        fusion = args.fusion_method
        self.use_extra_linear = args.extra_linear
        
        # 添加额外的线性层选项
        if self.use_extra_linear:
            input_dim = 128  # 64 + 64
            self.extra_linear_gray = nn.Linear(512, 64)
            self.extra_linear_colored = nn.Linear(512, 64)
        else:
            input_dim = 1024  # 512 + 512

        n_classes = 10

        # 更新fusion模块的输入维度
        if fusion == 'sum':
            self.fusion_module = SumFusion(input_dim=input_dim//2, output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(input_dim=input_dim, output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(input_dim=input_dim, output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(input_dim=input_dim, output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.gray_net = convnet(modal='grey')
        self.colored_net = convnet(modal='colored')

        # 添加单模态分类头
        self.head_gray = nn.Linear(64, n_classes)
        self.head_colored = nn.Linear(64, n_classes)

    def get_embeddings(self, gray, colored):
        """获取两个模态的嵌入向量"""
        g = self.gray_net(gray)
        c = self.colored_net(colored)

        g = torch.flatten(g, 1)
        c = torch.flatten(c, 1)

        if self.use_extra_linear:
            g = self.extra_linear_gray(g)
            c = self.extra_linear_colored(c)

        return g, c

    def forward(self, gray, colored):
        # 获取嵌入
        g, c = self.get_embeddings(gray, colored)
        
        # 通过fusion模块
        g, c, out = self.fusion_module(g, c)

        # 单模态输出
        out_gray = self.head_gray(g)
        out_colored = self.head_colored(c)

        return out_gray, out_colored, out
      

class SingleModalityClassifier_Mnist(nn.Module):
    def __init__(self, args):
        super(SingleModalityClassifier_Mnist, self).__init__()
        n_classes = 10
        self.backbone = convnet(modal=args.modality)
        self.fc = nn.Linear(512, n_classes)
    def forward(self, img):
        f = self.backbone(img)

        f = torch.flatten(f, 1)
        out = self.fc(f)

        return out

