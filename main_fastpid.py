import argparse
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pdb
from datetime import datetime
import os


import torch.nn.functional as F
from scipy.special import rel_entr
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA


from dataset.CramedDataset import CramedDataset

from models.basic_model import AVClassifier, TextImageClassifier, CGClassifier
from utils.utils import setup_seed, weight_init

num_workers = 8


NATS_TO_BITS = 1 / np.log(2)

class PIDNets_Adam_Optimizer(nn.Module):

    def __init__(self, max_iter=2000, lr=0.1, tol=1e-5, eps=1e-12, init_method: str = 'analytical', max_sinkhorn_iter=10):
        super().__init__()
        self.max_iter, self.lr, self.tol = max_iter, lr, tol
        self.init_method, self.max_sinkhorn_iter = init_method, max_sinkhorn_iter
        self.eps = eps
        self.NATS_TO_BITS = torch.tensor(NATS_TO_BITS, dtype=torch.float32)

    def _solve_q_pytorch(self, P: torch.Tensor) -> torch.Tensor:

        P_x1y = P.sum(dim=1, keepdim=True); P_x2y = P.sum(dim=0, keepdim=True)
        P_y = P.sum(dim=(0, 1), keepdim=True)
        Q = (P_x1y * P_x2y) / (P_y + self.eps)
        return Q / (Q.sum() + self.eps)

    def _mi_pytorch(self, P_joint: torch.Tensor) -> torch.Tensor:

        P_joint = P_joint / (P_joint.sum() + self.eps)
        margin_a = P_joint.sum(dim=1, keepdim=True); margin_b = P_joint.sum(dim=0, keepdim=True)
        log_P_joint = (P_joint + self.eps).log(); log_margin_a = (margin_a + self.eps).log()
        log_margin_b = (margin_b + self.eps).log()
        mi_terms = P_joint * (log_P_joint - log_margin_a - log_margin_b)
        return mi_terms.sum() * self.NATS_TO_BITS.to(P_joint.device)

    def _coi_pytorch(self, P: torch.Tensor) -> torch.Tensor:

        mi_y_x1 = self._mi_pytorch(P.sum(dim=1)); mi_y_x2 = self._mi_pytorch(P.sum(dim=0))
        P_y_x1x2 = P.permute(2, 0, 1).reshape(P.shape[2], -1)
        mi_y_x1x2 = self._mi_pytorch(P_y_x1x2)
        return mi_y_x1 + mi_y_x2 - mi_y_x1x2

    def _cmi_pytorch(self, P: torch.Tensor, cond_on_dim: int) -> torch.Tensor:

        cmi = torch.tensor(0.0, device=P.device, dtype=torch.float32)
        p_norm = P / (P.sum() + self.eps)
        if cond_on_dim == 1:
            p_x2 = p_norm.sum(dim=(0, 2))
            for x2_idx in range(P.shape[1]):
                if p_x2[x2_idx] > self.eps: cmi += p_x2[x2_idx] * self._mi_pytorch(p_norm[:, x2_idx, :].T)
        elif cond_on_dim == 0:
            p_x1 = p_norm.sum(dim=(1, 2))
            for x1_idx in range(P.shape[0]):
                if p_x1[x1_idx] > self.eps: cmi += p_x1[x1_idx] * self._mi_pytorch(p_norm[x1_idx, :, :].T)
        return cmi

    def _conditional_entropy(self, Q: torch.Tensor) -> torch.Tensor:

        Q_x1x2 = Q.sum(dim=2, keepdim=True); Q_y_given_x1x2 = Q / (Q_x1x2 + self.eps)
        log_Q_y_given_x1x2 = (Q_y_given_x1x2 + self.eps).log()
        return -torch.sum(Q * log_Q_y_given_x1x2)

    def _sinkhorn_projection(self, Q, P_x1y, P_x2y):

        Q_proj = Q.clone()
        for _ in range(self.max_sinkhorn_iter):
            Q_proj = Q_proj * (P_x2y / (Q_proj.sum(dim=0) + self.eps)).unsqueeze(0)
            Q_proj = Q_proj * (P_x1y / (Q_proj.sum(dim=1) + self.eps)).unsqueeze(1)
        return Q_proj / (Q_proj.sum() + self.eps)

    def forward(self, P):

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


        mi_p = self._mi_pytorch(P.permute(2,0,1).reshape(P.shape[2],-1))
        mi_q = self._mi_pytorch(q_star.permute(2,0,1).reshape(q_star.shape[2],-1))
        results = {'R': self._coi_pytorch(q_star), 'U1': self._cmi_pytorch(q_star, 1),
                   'U2': self._cmi_pytorch(q_star, 0), 'S': mi_p - mi_q}
        return {k: v.item() for k, v in results.items()}, actual_iters

def clustering(X, pca=True, n_clusters=20, n_components=5):

    X = np.nan_to_num(X)
    if len(X.shape) > 2: X = X.reshape(X.shape[0], -1)
    if pca:
        X = normalize(X)
        X = PCA(n_components=n_components, random_state=42).fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    return kmeans.labels_

def convert_data_to_distribution(x1_d, x2_d, y_d, n_clusters_x, n_clusters_y):

    joint_dist = np.zeros((n_clusters_x, n_clusters_x, n_clusters_y))
    np.add.at(joint_dist, (x1_d, x2_d, y_d), 1)
    return joint_dist / joint_dist.sum()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CREMAD', type=str,
                        help='UCF101, FOOD101, CREMAD, AVE, CGMNIST')
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film'])
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--use_video_frames', default=3, type=int)
    parser.add_argument('--random_seed', default=42, type=int)
    parser.add_argument('--val_fraction', default=0.1, type=float)
    parser.add_argument('--val_seed', default=42, type=int)
    parser.add_argument('--checkpoint_dir', default=None, type=str)
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

    parser.add_argument('--uniqueness_threshold', type=float, default=1.5)
    parser.add_argument('--epsilon', type=float, default=1e-8)

    parser.add_argument('--max_unimodal_epochs', type=int, default=200)
    parser.add_argument('--max_joint_epochs', type=int, default=50)
    parser.add_argument('--joint_lr', default=0.0001, type=float, help='initial learning rate')
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


                text_emb, visual_emb = model.get_embeddings(text_input_ids, attention_mask, visual)

                all_audio_embeddings.append(text_emb.cpu().numpy())
                all_visual_embeddings.append(visual_emb.cpu().numpy())
                all_labels.append(label)

            else:

                spec, image, label = batch[-3:]

                if args.dataset != 'CGMNIST' and args.dataset != 'UCF101':
                    spec = spec.unsqueeze(1).float().to(device)
                    image = image.float().to(device)
                else:
                    spec = spec.to(device)
                    image = image.to(device)


                audio_emb, visual_emb = model.get_embeddings(spec, image)

                all_audio_embeddings.append(audio_emb.cpu().numpy())
                all_visual_embeddings.append(visual_emb.cpu().numpy())
                all_labels.append(label.cpu().numpy())

            i+=1

            if i >= 20:
                break


    if not all_labels:
        print("Warning: No data collected for online probing.")
        return {"unique1": 1.0, "unique2": 1.0, "synergy": -1.0, "redundancy": -1.0}


    audio_embeddings = np.concatenate(all_audio_embeddings, axis=0)
    visual_embeddings = np.concatenate(all_visual_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)


    cluster_audio = clustering(audio_embeddings, pca=True, n_components=n_components, n_clusters=n_clusters)
    cluster_visual = clustering(visual_embeddings, pca=True, n_components=n_components, n_clusters=n_clusters)


    n_y_cats = len(np.unique(labels))
    P_np = convert_data_to_distribution(cluster_audio, cluster_visual, labels, n_clusters, n_y_cats)


    if P_np.sum() < 1e-9:
        print("Warning: Failed to create a valid probability distribution in online_probing.")
        return {"unique1": 1.0, "unique2": 1.0, "synergy": -1.0, "redundancy": -1.0}


    P_torch = torch.from_numpy(P_np).float().to(device)

    final_measures = {}
    try:

        measures, _ = pid_solver(P_torch)


        final_measures = {
            'unique1': measures.get('U1', 1.0),
            'unique2': measures.get('U2', 1.0),
            'synergy': measures.get('S', -1.0),
            'redundancy': measures.get('R', -1.0)
        }
    except Exception as e:
        print(f"Error during PID calculation in online_probing: {e}")

        final_measures = {"unique1": 1.0, "unique2": 1.0, "synergy": -1.0, "redundancy": -1.0}

    return final_measures


def train_epoch(args, epoch, model, device, dataloader, dataset, optimizer, scheduler, current_stage, measures, pid_solver, is_first_epoch=False):
    criterion = nn.CrossEntropyLoss()
    model.train()

    print("Start training ... ")

    for batch_idx, batch in enumerate(dataloader):

        if args.dataset == 'FOOD101':
            text_input_ids = batch['text'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            visual = batch['visual'].to(device)
            label = batch['label'].to(device)

            optimizer.zero_grad()
            out_a, out_v, out = model(text_input_ids, attention_mask, visual)
        else:

            spec, image, label = batch[-3:]
            label = label.to(device)
            optimizer.zero_grad()

            if args.dataset != 'CGMNIST' and args.dataset != 'UCF101':
                out_a, out_v, out = model(spec.unsqueeze(1).float().to(device), image.float().to(device))
            else:
                out_a, out_v, out = model(spec.to(device), image.to(device))


        if current_stage == 'modal_1':
            loss = criterion(out_a, label)
        elif current_stage == 'modal_2':
            loss = criterion(out_v, label)
        elif current_stage == 'unimodal':
            loss_a = criterion(out_a, label)
            loss_v = criterion(out_v, label)
            loss = loss_a + loss_v
        elif current_stage == 'joint':
            loss = criterion(out, label) + criterion(out_a, label) + criterion(out_v, label)

        loss.backward()
        optimizer.step()

    scheduler.step()


    if (epoch+1) % args.probing_freq == 0:
        print(f"Performing probing at epoch {epoch}...")

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

                spec, image, label = batch[-3:]
                spec = spec.to(device)
                image = image.to(device)
                label = label.to(device)

                if args.dataset != 'CGMNIST' and args.dataset != 'UCF101':
                    out_a, out_v, out = model(spec.unsqueeze(1).float(), image.float())
                else:
                    out_a, out_v, out = model(spec, image)


            prediction = softmax(out)
            pred_v = softmax(out_v)
            pred_a = softmax(out_a)


            if args.dataset == 'FOOD101':
                number = text_input_ids.shape[0]
            else:
                number = image.shape[0]

            for i in range(number):

                num[label[i]] += 1.0


                ma = np.argmax(prediction[i].cpu().data.numpy())
                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())


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


        if current_stage == 'modal_1':
            return None, sum(acc_a) / sum(num), None
        elif current_stage == 'modal_2':
            return None, None, sum(acc_v) / sum(num)
        elif current_stage == 'unimodal':
            return None, sum(acc_a) / sum(num), sum(acc_v) / sum(num)
        elif current_stage == 'joint':
            return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)


def build_dataset(args, mode):
    if args.dataset == 'UCF101':
        from dataset.UcfDataset import UCF101Dataset
        return UCF101Dataset(mode=mode, split="01")
    if args.dataset == 'CREMAD':
        return CramedDataset(args, mode=mode)
    if args.dataset == 'CGMNIST':
        from dataset.CGMNIST import CGMNISTDataset
        return CGMNISTDataset(args, mode=mode)
    if args.dataset == 'AVE':
        from dataset.AVEDataset import AVEDataset
        return AVEDataset(args, mode=mode)
    if args.dataset == 'FOOD101':
        from dataset.Food101 import FoodTextImageDataset
        return FoodTextImageDataset(
            root_dir="/media/dell/a7cc988c-d258-4230-a87c-983893c749a61/Tang_Jiaqi/Food101",
            mode=mode,
        )
    raise NotImplementedError('Incorrect dataset name {}!'.format(args.dataset))


def split_train_validation(dataset, fraction, seed):

    if not 0 < fraction < 1:
        raise ValueError('val_fraction must be between 0 and 1')
    val_size = max(1, int(len(dataset) * fraction))
    if val_size >= len(dataset):
        raise ValueError('At least one training and one validation sample are required')
    indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(seed)).tolist()
    val_indices = sorted(indices[:val_size])
    train_indices = sorted(indices[val_size:])
    validation = copy.copy(dataset)
    validation.mode = 'val'


    if hasattr(validation, 'transform'):
        from torchvision import transforms
        validation.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return Subset(dataset, train_indices), Subset(validation, val_indices)


def main():
    log_print(str(args))

    setup_seed(args.random_seed)

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')


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

    if args.max_unimodal_epochs < 1 or args.max_joint_epochs < 1:
        raise ValueError('Both training stages require at least one epoch')
    if not np.isfinite(args.epsilon) or args.epsilon <= 0:
        raise ValueError('epsilon must be finite and positive')
    train_dataset, val_dataset = split_train_validation(
        build_dataset(args, 'train'), args.val_fraction, args.val_seed)
    checkpoint_dir = args.checkpoint_dir or os.path.join(
        log_path, 'checkpoints_{}_{}'.format(current_time, os.getpid()))
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, 'train_val_split.json'), 'w') as handle:
        json.dump({'val_seed': args.val_seed, 'val_fraction': args.val_fraction,
                   'train_indices': train_dataset.indices,
                   'val_indices': val_dataset.indices}, handle, indent=2)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=num_workers, pin_memory=True)

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=num_workers, pin_memory=True)


    current_stage = 'unimodal'
    best_synergy = -1
    last_synergy = -1

    measures = {}
    synergy_collapse = False


    measures["unique1"] = 1.0
    measures["unique2"] = 1.0
    measures["synergy"] = -1.0

    for epoch in range(args.max_unimodal_epochs):
        log_print('Uni Epoch: {}: '.format(epoch))
        measures = train_epoch(args, epoch, model, device, train_dataloader, train_dataset, optimizer, scheduler,
                            current_stage, measures, pid_solver)


        acc, acc_a, acc_v = valid(args, model, device, val_dataloader, 'unimodal')
        log_print(f'Current mode: {current_stage}')
        if acc is not None:
            log_print(f'Joint Accuracy: {acc:.4f}')
        if acc_a is not None:
            log_print(f'modality_1 Validation Accuracy: {acc_a:.4f}')
        if acc_v is not None:
            log_print(f'modality_2 Validation Accuracy: {acc_v:.4f}')


        if (epoch + 1) % args.probing_freq == 0 and current_stage != "joint":

            if measures['synergy'] > best_synergy:
                best_synergy = measures['synergy']
            last_synergy = measures['synergy']


            uniqueness_ratio = measures['unique1'] / (measures['unique2'] + args.epsilon)
            reverse_uniqueness_ratio = measures['unique2'] / (measures['unique1'] + args.epsilon)
            if uniqueness_ratio > args.uniqueness_threshold and current_stage != "modal_2":
                current_stage = "modal_2"
                print("Stopping audio training due to high uniqueness ratio")
            elif reverse_uniqueness_ratio > args.uniqueness_threshold and current_stage != "modal_1":
                current_stage = "modal_1"
                print("Stopping visual training due to high uniqueness ratio")
            else:
                pass


            if best_synergy > 0 and last_synergy < best_synergy * args.synergy_drop_threshold:
                synergy_collapse = True
                log_print(f"Synergy has dropped significantly (Current: {last_synergy:.4f}, Peak: {best_synergy:.4f}). Preparing for joint training.")


        if synergy_collapse:
            log_print("Ending unimodal phase: synergy decline.")
            current_stage = "joint"
            break


    stage1_checkpoint = os.path.join(checkpoint_dir, 'stage1_final.pt')
    torch.save({
        'model_state_dict': model.state_dict(), 'epoch': epoch + 1,
        'stop_reason': 'synergy_decline' if synergy_collapse else 'stage1_epoch_budget',
        'val_seed': args.val_seed,
    }, stage1_checkpoint)

    best_acc = -1.0
    stage2_checkpoint = os.path.join(checkpoint_dir, 'stage2_best_validation.pt')
    current_stage = "joint"
    optimizer_2 = optim.SGD(model.parameters(), lr=args.joint_lr, momentum=0.9, weight_decay=1e-4)
    scheduler_2 = optim.lr_scheduler.StepLR(optimizer_2, args.lr_decay_step, args.lr_decay_ratio)

    for epoch in range(args.max_joint_epochs):
        log_print('Joint Epoch: {}: '.format(epoch))

        measures = train_epoch(args, epoch, model, device, train_dataloader, train_dataset, optimizer_2, scheduler_2,
                               current_stage, measures, pid_solver)
        acc, acc_a, acc_v = valid(args, model, device, val_dataloader, current_stage)


        if acc > best_acc:
            best_acc = acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1, 'validation_accuracy': acc,
                'selection_metric': 'joint_validation_accuracy',
                'val_seed': args.val_seed, 'stage1_checkpoint': stage1_checkpoint,
            }, stage2_checkpoint)
            log_print(f'New best validation accuracy: {best_acc:.4f}')

        log_print(f'Current mode: {current_stage}')
        log_print(f'Joint Validation Accuracy: {acc:.4f}, best Validation Accuracy:{best_acc:.4f}')
        if acc_a is not None:
            log_print(f'Audio Validation Accuracy: {acc_a:.4f}')
        if acc_v is not None:
            log_print(f'Visual Validation Accuracy: {acc_v:.4f}')


    checkpoint = torch.load(stage2_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_dataset = build_dataset(args, 'test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=num_workers, pin_memory=True)
    test_acc, test_acc_a, test_acc_v = valid(args, model, device, test_dataloader, 'joint')
    log_print('Final test accuracy at validation-selected epoch {}: {:.4f}'.format(
        checkpoint['epoch'], test_acc))
    with open(os.path.join(checkpoint_dir, 'test_result.json'), 'w') as handle:
        json.dump({'selected_epoch': checkpoint['epoch'],
                   'validation_accuracy': checkpoint['validation_accuracy'],
                   'test_accuracy': test_acc, 'test_modality1_accuracy': test_acc_a,
                   'test_modality2_accuracy': test_acc_v,
                   'checkpoint': stage2_checkpoint}, handle, indent=2)

if __name__ == "__main__":
    main()
