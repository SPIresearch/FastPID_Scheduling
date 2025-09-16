import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pdb
from datetime import datetime
import os

# ==============================================================================
# SECTION 1: NEW IMPORTS FOR THE IMPROVED PID CALCULATION
# ==============================================================================
# These imports are required for the new PyTorch-based PID solver and its helpers.
import torch.nn.functional as F
from scipy.special import rel_entr
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
# ==============================================================================

# from cluster_pid import * # This import is no longer needed.

from dataset.CramedDataset import CramedDataset
from dataset.VGGSoundDataset import VGGSound
from dataset.AVEDataset import AVEDataset
from dataset.CGMNIST import CGMNISTDataset
from dataset.Food101 import FoodTextImageDataset

from models.basic_model import AVClassifier, TextImageClassifier, CGClassifier
from utils.utils import setup_seed, weight_init

num_workers = 8

# ==============================================================================
# SECTION 2: PYTORCH-BASED PID SOLVER AND HELPERS
# This section contains the new, improved PID calculation logic based on the
# reference code provided. It replaces the original CVXPY implementation.
# ==============================================================================

NATS_TO_BITS = 1 / np.log(2)

class PIDNets_Adam_Optimizer(nn.Module):
    """
    A PyTorch-based PID solver using the Adam optimizer.
    This class finds a distribution Q that minimizes the conditional entropy H(Y|X1,X2)
    while satisfying marginal constraints, allowing for the calculation of PID components.
    The 'analytical' initialization provides a good starting point, leading to faster convergence.
    """
    def __init__(self, max_iter=2000, lr=0.1, tol=1e-5, eps=1e-12, init_method: str = 'analytical', max_sinkhorn_iter=10):
        super().__init__()
        self.max_iter, self.lr, self.tol = max_iter, lr, tol
        self.init_method, self.max_sinkhorn_iter = init_method, max_sinkhorn_iter
        self.eps = eps
        self.NATS_TO_BITS = torch.tensor(NATS_TO_BITS, dtype=torch.float32)

    def _solve_q_pytorch(self, P: torch.Tensor) -> torch.Tensor:
        """Computes the analytical solution for Q to be used for warm-starting."""
        P_x1y = P.sum(dim=1, keepdim=True); P_x2y = P.sum(dim=0, keepdim=True)
        P_y = P.sum(dim=(0, 1), keepdim=True)
        Q = (P_x1y * P_x2y) / (P_y + self.eps)
        return Q / (Q.sum() + self.eps)

    def _mi_pytorch(self, P_joint: torch.Tensor) -> torch.Tensor:
        """Calculates Mutual Information in PyTorch."""
        P_joint = P_joint / (P_joint.sum() + self.eps)
        margin_a = P_joint.sum(dim=1, keepdim=True); margin_b = P_joint.sum(dim=0, keepdim=True)
        log_P_joint = (P_joint + self.eps).log(); log_margin_a = (margin_a + self.eps).log()
        log_margin_b = (margin_b + self.eps).log()
        mi_terms = P_joint * (log_P_joint - log_margin_a - log_margin_b)
        return mi_terms.sum() * self.NATS_TO_BITS.to(P_joint.device)

    def _coi_pytorch(self, P: torch.Tensor) -> torch.Tensor:
        """Calculates Co-Information (Redundancy) in PyTorch."""
        mi_y_x1 = self._mi_pytorch(P.sum(dim=1)); mi_y_x2 = self._mi_pytorch(P.sum(dim=0))
        P_y_x1x2 = P.permute(2, 0, 1).reshape(P.shape[2], -1)
        mi_y_x1x2 = self._mi_pytorch(P_y_x1x2)
        return mi_y_x1 + mi_y_x2 - mi_y_x1x2

    def _cmi_pytorch(self, P: torch.Tensor, cond_on_dim: int) -> torch.Tensor:
        """Calculates Conditional Mutual Information (Uniqueness) in PyTorch."""
        cmi = torch.tensor(0.0, device=P.device, dtype=torch.float32)
        p_norm = P / (P.sum() + self.eps)
        if cond_on_dim == 1: # I(Y; X1 | X2) -> U1
            p_x2 = p_norm.sum(dim=(0, 2))
            for x2_idx in range(P.shape[1]):
                if p_x2[x2_idx] > self.eps: cmi += p_x2[x2_idx] * self._mi_pytorch(p_norm[:, x2_idx, :].T)
        elif cond_on_dim == 0: # I(Y; X2 | X1) -> U2
            p_x1 = p_norm.sum(dim=(1, 2))
            for x1_idx in range(P.shape[0]):
                if p_x1[x1_idx] > self.eps: cmi += p_x1[x1_idx] * self._mi_pytorch(p_norm[x1_idx, :, :].T)
        return cmi

    def _conditional_entropy(self, Q: torch.Tensor) -> torch.Tensor:
        """Calculates the conditional entropy H(Y|X1,X2), the loss function to be minimized."""
        Q_x1x2 = Q.sum(dim=2, keepdim=True); Q_y_given_x1x2 = Q / (Q_x1x2 + self.eps)
        log_Q_y_given_x1x2 = (Q_y_given_x1x2 + self.eps).log()
        return -torch.sum(Q * log_Q_y_given_x1x2)

    def _sinkhorn_projection(self, Q, P_x1y, P_x2y):
        """Projects a distribution Q onto the marginal constraints defined by P."""
        Q_proj = Q.clone()
        for _ in range(self.max_sinkhorn_iter):
            Q_proj = Q_proj * (P_x2y / (Q_proj.sum(dim=0) + self.eps)).unsqueeze(0)
            Q_proj = Q_proj * (P_x1y / (Q_proj.sum(dim=1) + self.eps)).unsqueeze(1)
        return Q_proj / (Q_proj.sum() + self.eps)

    def forward(self, P):
        """The main optimization loop."""
        P = P / (P.sum() + self.eps); P_x1y = P.sum(dim=1); P_x2y = P.sum(dim=0)
        if self.init_method == 'analytical': q_init = self._solve_q_pytorch(P)
        else: q_init = torch.ones_like(P) / P.numel()
        logits = (q_init.clamp(min=self.eps)).log().clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([logits], lr=self.lr)
        q_old = q_init; actual_iters = 0
        for i in range(self.max_iter):
            actual_iters = i + 1; optimizer.zero_grad()
            q_curr = F.softmax(logits.flatten(), dim=0).reshape_as(P)
            q_proj = self._sinkhorn_projection(q_curr, P_x1y, P_x2y)
            with torch.no_grad():
                change = torch.max(torch.abs(q_proj - q_old))
                if change < self.tol and i > 1: break
                q_old = q_proj.clone()
            loss = -self._conditional_entropy(q_proj)
            loss.backward(); optimizer.step()
        with torch.no_grad():
            q_star_unproj = F.softmax(logits.flatten(), dim=0).reshape_as(P)
            q_star = self._sinkhorn_projection(q_star_unproj, P_x1y, P_x2y)

        # Calculate all PID measures with harmonized output keys
        mi_p = self._mi_pytorch(P.permute(2,0,1).reshape(P.shape[2],-1))
        mi_q = self._mi_pytorch(q_star.permute(2,0,1).reshape(q_star.shape[2],-1))
        results = {'R': self._coi_pytorch(q_star), 'U1': self._cmi_pytorch(q_star, 1),
                   'U2': self._cmi_pytorch(q_star, 0), 'S': mi_p - mi_q}
        return {k: v.item() for k, v in results.items()}, actual_iters

def clustering(X, pca=True, n_clusters=20, n_components=5):
    """
    Discretizes continuous embedding vectors into cluster labels using PCA and KMeans.
    """
    X = np.nan_to_num(X)
    if len(X.shape) > 2: X = X.reshape(X.shape[0], -1)
    if pca:
        X = normalize(X)
        X = PCA(n_components=n_components, random_state=42).fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    return kmeans.labels_

def convert_data_to_distribution(x1_d, x2_d, y_d, n_clusters_x, n_clusters_y):
    """
    Converts discrete cluster labels and class labels into a joint probability
    distribution P(X1, X2, Y).
    """
    joint_dist = np.zeros((n_clusters_x, n_clusters_x, n_clusters_y))
    np.add.at(joint_dist, (x1_d, x2_d, y_d), 1)
    return joint_dist / joint_dist.sum()

# ==============================================================================

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='AVE', type=str,
                        help='UCF101, FOOD101, CREMAD, AVE, CGMNIST')
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film'])
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--use_video_frames', default=3, type=int)
    parser.add_argument('--random_seed', default=42, type=int)
    parser.add_argument('--audio_path', default='./CREMA-D/AudioWAV', type=str)
    parser.add_argument('--visual_path', default='./CREMA-D/', type=str)
    parser.add_argument('--train', default="True", help='turn on train mode')

    parser.add_argument('--batch_size', default=64, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')
    parser.add_argument('--probing_freq', type=int, default=5, help='Frequency of probing (epochs)')
    parser.add_argument('--extra_linear', default=True, choices=[True, False])

    parser.add_argument('--uniqueness_threshold', type=float, default=3)
    # parser.add_argument('--synergy_threshold', type=float, default=0.5)
    parser.add_argument('--max_unimodal_epochs', type=int, default=200)
    parser.add_argument('--max_joint_epochs', type=int, default=100)
    parser.add_argument('--joint_lr', default=0.0005, type=float, help='initial learning rate')
    parser.add_argument('--synergy_drop_threshold', type=float, default=0.8, help='Threshold for synergy drop to trigger early joint training')

    parser.add_argument('--gpu', default='1', type=str, help='GPU')

    return parser.parse_args()

args = get_arguments()
current_time = datetime.now().strftime('%d_%H:%M')
log_path = f'./logs/{args.dataset}'
os.makedirs(log_path, exist_ok=True)
log_filename = f'/probing_log_{current_time}.txt'
log_file = open(log_path+log_filename, 'a')

def log_print(message):
    print(message)
    log_file.write(message + '\n')


def online_probing(args, model, train_dataset, device, pid_solver, n_components=5, n_clusters=20):
    """
    执行在线probing (使用PyTorch-based PID求解器).
    This function replaces the original implementation. It extracts embeddings,
    discretizes them, builds a joint distribution, and then calls the fast
    PyTorch-based solver to get PID measures.
    """
    model.eval()
    all_audio_embeddings = []
    all_visual_embeddings = []
    all_labels = []

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=num_workers, pin_memory=True)

    with torch.no_grad():
        i = 0
        for batch in train_dataloader:
            if args.dataset == 'FOOD101':
                text_input_ids = batch['text'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                visual = batch['visual'].to(device)
                label = batch['label'].cpu().numpy()

                # 获取embeddings
                text_emb, visual_emb = model.get_embeddings(text_input_ids, attention_mask, visual)

                all_audio_embeddings.append(text_emb.cpu().numpy())  # 文本作为"audio"替代
                all_visual_embeddings.append(visual_emb.cpu().numpy())
                all_labels.append(label)

            else:
                # 原有代码处理其他数据集
                _, spec, image, label = batch

                if args.dataset != 'CGMNIST' and args.dataset != 'UCF101':
                    spec = spec.unsqueeze(1).float().to(device)
                    image = image.float().to(device)
                else:
                    spec = spec.to(device)
                    image = image.to(device)

                # 获取embeddings
                audio_emb, visual_emb = model.get_embeddings(spec, image)

                all_audio_embeddings.append(audio_emb.cpu().numpy())
                all_visual_embeddings.append(visual_emb.cpu().numpy())
                all_labels.append(label.cpu().numpy())

            i+=1
            # Limit probing to a fixed number of batches to ensure speed
            if i >= 20:
                break

    # If no data was loaded for any reason, return a default dict
    if not all_labels:
        print("Warning: No data collected for online probing.")
        return {"unique1": 1.0, "unique2": 1.0, "synergy": -1.0, "redundancy": -1.0}

    # 合并所有结果
    audio_embeddings = np.concatenate(all_audio_embeddings, axis=0)
    visual_embeddings = np.concatenate(all_visual_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # 1. Discretize features via PCA and KMeans
    cluster_audio = clustering(audio_embeddings, pca=True, n_components=n_components, n_clusters=n_clusters)
    cluster_visual = clustering(visual_embeddings, pca=True, n_components=n_components, n_clusters=n_clusters)

    # 2. Create the joint probability distribution P(X1, X2, Y)
    n_y_cats = len(np.unique(labels))
    P_np = convert_data_to_distribution(cluster_audio, cluster_visual, labels, n_clusters, n_y_cats)

    # If distribution is empty, return default
    if P_np.sum() < 1e-9:
        print("Warning: Failed to create a valid probability distribution in online_probing.")
        return {"unique1": 1.0, "unique2": 1.0, "synergy": -1.0, "redundancy": -1.0}

    # 3. Convert to PyTorch tensor and move to the correct device
    P_torch = torch.from_numpy(P_np).float().to(device)

    final_measures = {}
    try:
        # 4. Calculate PID using the PyTorch solver
        measures, _ = pid_solver(P_torch)
        # 5. Map new keys ('U1', 'U2', 'S', 'R') to old keys ('unique1', 'unique2', 'synergy')
        # for compatibility with the existing training logic.
        final_measures = {
            'unique1': measures.get('U1', 1.0),
            'unique2': measures.get('U2', 1.0),
            'synergy': measures.get('S', -1.0),
            'redundancy': measures.get('R', -1.0)
        }
    except Exception as e:
        print(f"Error during PID calculation in online_probing: {e}")
        # Return a default value that prevents division by zero and doesn't trigger thresholds
        final_measures = {"unique1": 1.0, "unique2": 1.0, "synergy": -1.0, "redundancy": -1.0}

    return final_measures


def train_epoch(args, epoch, model, device, dataloader, dataset, optimizer, scheduler, current_stage, measures, pid_solver, is_first_epoch=False):
    criterion = nn.CrossEntropyLoss()
    model.train()

    print("Start training ... ")

    for batch_idx, batch in enumerate(dataloader):
        # 根据数据集类型处理输入
        if args.dataset == 'FOOD101':
            text_input_ids = batch['text'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            visual = batch['visual'].to(device)
            label = batch['label'].to(device)

            optimizer.zero_grad()
            out_a, out_v, out = model(text_input_ids, attention_mask, visual)
        else:
            # 原有代码处理其他数据集
            _, spec, image, label = batch
            label = label.to(device)
            optimizer.zero_grad()

            if args.dataset != 'CGMNIST' and args.dataset != 'UCF101':
                out_a, out_v, out = model(spec.unsqueeze(1).float().to(device), image.float().to(device))
            else:
                out_a, out_v, out = model(spec.to(device), image.to(device))


        # 根据不同mode计算loss
        if current_stage == 'modal_1':
            loss = criterion(out_a, label)
        elif current_stage == 'modal_2':
            loss = criterion(out_v, label)
        elif current_stage == 'unimodal':
            loss_a = criterion(out_a, label)
            loss_v = criterion(out_v, label)
            loss = loss_a + loss_v
        elif current_stage == 'joint':
            loss = criterion(out, label) + criterion(out_a, label)*0.6 + criterion(out_v, label)*0.6

        loss.backward()
        optimizer.step()

    if current_stage=="joint":
        scheduler.step()

    # 如果是probing的epoch
    if (epoch+1) % args.probing_freq == 0:
        print(f"Performing probing at epoch {epoch}...")
        # Pass the pid_solver to the probing function
        measures = online_probing(args, model, dataset, device, pid_solver)
        log_print(f"PID measures at epoch {epoch}:"+str(measures))
    return measures



def valid(args, model, device, dataloader, current_stage):
    softmax = nn.Softmax(dim=1)
    model.eval()

    if args.dataset == 'UCF101':
        n_classes = 101
    elif args.dataset == 'KineticSound':
        n_classes = 31
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'CGMNIST':
        n_classes = 10
    elif args.dataset == 'AVE':
        n_classes = 28
    elif args.dataset == 'FOOD101':
        n_classes = 101
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    with torch.no_grad():
        # 每个类别的样本数和正确预测数
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        for batch in dataloader:
            if args.dataset == 'FOOD101':
                text_input_ids = batch['text'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                visual = batch['visual'].to(device)
                label = batch['label'].to(device)

                out_a, out_v, out = model(text_input_ids, attention_mask, visual)
            else:
                # 原有代码处理其他数据集
                _, spec, image, label = batch
                spec = spec.to(device)
                image = image.to(device)
                label = label.to(device)

                if args.dataset != 'CGMNIST' and args.dataset != 'UCF101':
                    out_a, out_v, out = model(spec.unsqueeze(1).float(), image.float())
                else:
                    out_a, out_v, out = model(spec, image)


            # 应用softmax
            prediction = softmax(out)
            pred_v = softmax(out_v)
            pred_a = softmax(out_a)

            # 逐样本统计
            if args.dataset == 'FOOD101':
                number = text_input_ids.shape[0]
            else:
                number = image.shape[0]

            for i in range(number):
                # 更新样本计数
                num[label[i]] += 1.0

                # 获取预测结果
                ma = np.argmax(prediction[i].cpu().data.numpy())
                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())

                # 根据mode更新正确预测计数
                if current_stage in ['modal_1', 'unimodal']:
                    if np.asarray(label[i].cpu()) == a:
                        acc_a[label[i]] += 1.0

                if current_stage in ['modal_2', 'unimodal']:
                    if np.asarray(label[i].cpu()) == v:
                        acc_v[label[i]] += 1.0

                if current_stage == 'joint':
                    if np.asarray(label[i].cpu()) == ma:
                        acc[label[i]] += 1.0
                    if np.asarray(label[i].cpu()) == v:
                        acc_v[label[i]] += 1.0
                    if np.asarray(label[i].cpu()) == a:
                        acc_a[label[i]] += 1.0

        # 根据不同模式返回相应的准确率
        if current_stage == 'modal_1':
            return None, sum(acc_a) / sum(num), None
        elif current_stage == 'modal_2':
            return None, None, sum(acc_v) / sum(num)
        elif current_stage == 'unimodal':
            return None, sum(acc_a) / sum(num), sum(acc_v) / sum(num)
        elif current_stage == 'joint':
            return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)


def main():
    log_print(str(args))

    setup_seed(args.random_seed)

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    # Instantiate the new PyTorch-based PID solver once and pass it to the training loop.
    # Using 'analytical' init is recommended for speed and stability.
    pid_solver = PIDNets_Adam_Optimizer(init_method='analytical', tol=1e-5, lr=0.1, max_iter=2000).to(device)

    if args.dataset == 'CGMNIST':
        model = CGClassifier(args)
    elif args.dataset == 'FOOD101':
        model = TextImageClassifier(args)
    else:
        model = AVClassifier(args)

    model.apply(weight_init)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    if args.dataset == 'UCF101':
        from dataset.UcfDataset import UCF101Dataset
        train_dataset = UCF101Dataset( mode='train', split="01")
        test_dataset = UCF101Dataset( mode='test', split="01")
    elif args.dataset == 'CREMAD':
        train_dataset = CramedDataset(args, mode='train')
        test_dataset = CramedDataset(args, mode='test')
    elif args.dataset == 'CGMNIST':
        train_dataset = CGMNISTDataset(args, mode='train')
        test_dataset = CGMNISTDataset(args, mode='test')
        val_dataset = CGMNISTDataset(args, mode='test')
    elif args.dataset == 'AVE':
        train_dataset = AVEDataset(args, mode='train')
        test_dataset = AVEDataset(args, mode='test')
    elif args.dataset == 'FOOD101':
        train_dataset = FoodTextImageDataset(
            root_dir="/media/dell/a7cc988c-d258-4230-a87c-983893c749a61/Tang_Jiaqi/Food101",
            mode='train',
        )
        test_dataset = FoodTextImageDataset(
            root_dir="/media/dell/a7cc988c-d258-4230-a87c-983893c749a61/Tang_Jiaqi/Food101",
            mode='test',
        )
    else:
        raise NotImplementedError('Incorrect dataset name {}!'.format(args.dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=num_workers, pin_memory=True)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=num_workers, pin_memory=True)


    # --- 训练逻辑开始 ---
    current_stage = 'unimodal'
    best_synergy = -1
    last_synergy = -1

    measures = {}
    reverse = 0
    synergy_collapse = False  # <--- 新增：Synergy崩溃标志

    # 初始化以防止Key-Error
    measures["unique1"] = 1.0
    measures["unique2"] = 1.0
    measures["synergy"] = -1.0

    for epoch in range(args.max_unimodal_epochs):
        log_print('Uni Epoch: {}: '.format(epoch))
        measures = train_epoch(args, epoch, model, device, train_dataloader, train_dataset, optimizer, scheduler,
                            current_stage, measures, pid_solver)

        # 验证阶段
        acc, acc_a, acc_v = valid(args, model, device, test_dataloader, current_stage)
        log_print(f'Current mode: {current_stage}')
        if acc is not None:
            log_print(f'Joint Accuracy: {acc:.4f}')
        if acc_a is not None:
            log_print(f'modality_1 Accuracy: {acc_a:.4f}')
        if acc_v is not None:
            log_print(f'modality_2 Accuracy: {acc_v:.4f}')

        # 检查独特性和协同性的核心逻辑
        if measures["unique1"] != 1.0 and measures["unique2"] > 0 and current_stage != "joint":
            # 1. 更新Synergy相关的状态
            if measures['synergy'] > best_synergy:
                best_synergy = measures['synergy']
            last_synergy = measures['synergy']

            # 2. 检查Uniqueness Ratio以决定是否切换训练模态
            uniqueness_ratio = measures['unique1'] / measures['unique2']
            if uniqueness_ratio > args.uniqueness_threshold and current_stage != "modal_2":
                current_stage = "modal_2"
                print("Stopping audio training due to high uniqueness ratio")
                reverse += 1
            elif 1/uniqueness_ratio > args.uniqueness_threshold and current_stage != "modal_1":
                current_stage = "modal_1"
                print("Stopping visual training due to high uniqueness ratio")
                reverse += 1
            else:
                pass

            # 3. <--- 新增：检查Synergy是否崩溃 ---
            # 确保best_synergy已经是一个有意义的正数
            if best_synergy > 0 and last_synergy < best_synergy * args.synergy_drop_threshold:
                synergy_collapse = True
                log_print(f"Synergy has dropped significantly (Current: {last_synergy:.4f}, Peak: {best_synergy:.4f}). Preparing for joint training.")

        # <--- 修改：更新最终的跳出条件 ---
        # 检查是否应该结束单模态训练
        if current_stage == "joint" or reverse == 2 or synergy_collapse:
            log_print("Ending unimodal phase. Reason:")
            if reverse == 2:
                log_print(" -> Modality training switch completed twice (balanced).")
            if synergy_collapse:
                log_print(" -> Synergy has collapsed, indicating overfitting.")
            
            current_stage = "joint"  # 确保退出后的状态是'joint'
            break

    best_acc = 0.0
    current_stage = "joint"
    optimizer_2 = optim.SGD(model.parameters(), lr=args.joint_lr, momentum=0.9, weight_decay=1e-4)

    for epoch in range(args.max_joint_epochs):
        log_print('Joint Epoch: {}: '.format(epoch))
        # Pass the pid_solver instance to the training epoch function
        measures = train_epoch(args, epoch, model, device, train_dataloader, train_dataset, optimizer_2, scheduler,
                               current_stage, measures, pid_solver)
        acc, acc_a, acc_v = valid(args, model, device, test_dataloader, current_stage)

        # 更新best accuracy
        if acc > best_acc:
            best_acc = acc
            log_print(f'New best accuracy: {best_acc:.4f}')

        log_print(f'Current mode: {current_stage}')
        log_print(f'Joint Accuracy: {acc:.4f}, best Accuracy:{best_acc:.4f}')
        if acc_a is not None:
            log_print(f'Audio Accuracy: {acc_a:.4f}')
        if acc_v is not None:
            log_print(f'Visual Accuracy: {acc_v:.4f}')

if __name__ == "__main__":
    main()