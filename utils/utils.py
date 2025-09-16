import torch
import torch.nn as nn
import numpy as np
import random
from models.basic_model import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
import copy

def evaluate_unimodal(trained_model, train_loader, val_loader, args, device):
    audio_net = copy.deepcopy(trained_model.audio_net)
    visual_net = copy.deepcopy(trained_model.visual_net)
    # 音频模态评估
    audio_evaluator = UnimodalEvaluator(
        audio_net,
        'audio',
        trained_model.n_class
    )

    # 视觉模态评估
    visual_evaluator = UnimodalEvaluator(
        visual_net,
        'visual',
        trained_model.n_class
    )

    audio_evaluator.to(device)
    visual_evaluator.to(device)

    # 优化器
    optimizer_a = torch.optim.Adam(audio_evaluator.classifier.parameters(), lr=args.eval_lr)
    optimizer_v = torch.optim.Adam(visual_evaluator.classifier.parameters(), lr=args.eval_lr)
    criterion = nn.CrossEntropyLoss()

    # 训练新的分类头
    for epoch in range(args.eval_epochs):
        audio_evaluator.train()
        visual_evaluator.train()

        for batch_idx, (spec, image, label) in enumerate(train_loader):
            spec, image, label = spec.to(device), image.to(device), label.to(device)

            # 训练音频分类头
            optimizer_a.zero_grad()
            out_a = audio_evaluator(spec.unsqueeze(1).float())
            loss_a = criterion(out_a, label)
            loss_a.backward()
            optimizer_a.step()

            # 训练视觉分类头
            optimizer_v.zero_grad()
            out_v = visual_evaluator(image.float())
            loss_v = criterion(out_v, label)
            loss_v.backward()
            optimizer_v.step()

        # 验证
        audio_evaluator.eval()
        visual_evaluator.eval()
        correct_a = 0
        correct_v = 0
        total = 0

        with torch.no_grad():
            for spec, image, label in val_loader:
                spec, image, label = spec.to(device), image.to(device), label.to(device)

                out_a = audio_evaluator(spec.unsqueeze(1).float())
                out_v = visual_evaluator(image.float())

                _, pred_a = out_a.max(1)
                _, pred_v = out_v.max(1)

                correct_a += pred_a.eq(label).sum().item()
                correct_v += pred_v.eq(label).sum().item()
                total += label.size(0)

        audio_acc = 100. * correct_a / total
        visual_acc = 100. * correct_v / total


    return audio_acc, visual_acc