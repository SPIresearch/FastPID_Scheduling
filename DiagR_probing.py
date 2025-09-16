import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
from cluster_pid import *
from dataset.CramedDataset import CramedDataset
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

from models.basic_model import AVClassifier, CGClassifier
from utils.utils import setup_seed, weight_init


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CREMAD', type=str,
                        help='VGGSound, KineticSound, CREMAD, AVE, CGMNIST')
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film'])
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--use_video_frames', default=3, type=int)
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--audio_path', default='./CREMA-D/AudioWAV', type=str)
    parser.add_argument('--visual_path', default='./CREMA-D/', type=str)
    parser.add_argument('--train', default="True", help='turn on train mode')

    parser.add_argument('--batch_size', default=64, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')
    parser.add_argument('--probing_freq', type=int, default=5, help='Frequency of probing (epochs)')
    parser.add_argument('--extra_linear', default=True, choices=[True, False])

    parser.add_argument('--uniqueness_threshold', type=float, default=3)
    parser.add_argument('--best_synergy_threshold', type=float, default=0.5)
    parser.add_argument('--synergy_threshold', type=float, default=0.5)
    parser.add_argument('--max_unimodal_epochs', type=int, default=150)
    parser.add_argument('--max_joint_epochs', type=int, default=100)
    parser.add_argument('--joint_lr', default=0.0001, type=float, help='initial learning rate')


    parser.add_argument('--reinit_epoch', default=20, type=int)
    parser.add_argument('--reinit_num', default=3, type=int)

    parser.add_argument('--gpu', default='0', type=str, help='GPU ids')

    return parser.parse_args()


args = get_arguments()
current_time = datetime.now().strftime('%d_%H:%M')
log_filename = f'./logs/{args.dataset}/diag{current_time}.txt'
log_file = open("./"+log_filename, 'a')

def log_print(message):
    print(message)
    log_file.write(message + '\n')


def purity_score(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that 
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

def reinit_score(args, train_audio,train_visual,train_label,val_audio,val_visual,val_label):
    all_feature=[train_audio,val_audio,train_visual,val_visual]
    stages=['train_audio','val_audio','train_visual','val_visual']
    all_purity=[]

    for idx,fea in enumerate(all_feature):
        print('Computing t-SNE embedding')
        result = fea
        # 归一化处理
        scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        result = scaler.fit_transform(result)
        y_pred = KMeans(n_clusters=args.n_classes, random_state=0).fit_predict(result)

        if(stages[idx][:5]=='train'):
            purity=purity_score(np.array(train_label),y_pred)
        else:
            purity=purity_score(np.array(val_label),y_pred)
        all_purity.append(purity)


        print('%s purity= %.4f' % (stages[idx],purity))
        print('%%%%%%%%%%%%%%%%%%%%%%%%') 
    
    purity_gap_audio=np.abs(all_purity[0]-all_purity[1])
    purity_gap_visual=np.abs(all_purity[2]-all_purity[3])


    weight_audio=torch.tanh(torch.tensor(args.move_lambda*purity_gap_audio))
    weight_visual=torch.tanh(torch.tensor(args.move_lambda*purity_gap_visual))

    print('weight audio')
    print(weight_audio)
    print('weight visual')
    print(weight_visual)


    return weight_audio,weight_visual

def reinit(args, model,checkpoint,weight_audio,weight_visual):


    print("Start reinit ... ")


    record_names_audio = []
    record_names_visual = []
    for name, param in model.named_parameters():
        if 'audio_net' in name:
            if('conv' in name):
                record_names_audio.append((name, param))
        elif 'visual_net' in name:
            if('conv' in name):
                record_names_visual.append((name, param))


    for name, param in model.named_parameters():
        if 'audio_net' in name:
            init_weight=checkpoint[name]
            current_weight=param.data
            new_weight=weight_audio*init_weight+(1-weight_audio).cuda()*current_weight
            param.data=new_weight
        elif 'visual_net' in name:
            init_weight=checkpoint[name]
            current_weight=param.data
            new_weight=weight_visual*init_weight+(1-weight_visual).cuda()*current_weight
            param.data=new_weight

    
    return model

def get_feature(args, epoch, model, device, dataloader):
    model.eval()
    all_audio=[]
    all_visual=[]
    all_label=[]


    with torch.no_grad():
        for _, spec, images, label in dataloader:

            images = images.to(device)
            spec = spec.to(device)
            label = label.to(device)
            _,_,_,a,v = model(spec.float(), images.float())
            all_audio.append(a.data.cpu())
            all_visual.append(v.data.cpu())
            all_label.append(label.data.cpu())
    
    all_audio=torch.cat(all_audio)
    all_visual=torch.cat(all_visual)
    all_label=torch.cat(all_label)



    return all_audio,all_visual,all_label


def online_probing(args, model, train_dataset, device, n_components=5, n_clusters=20):
    """执行在线probing"""
    model.eval()
    all_audio_embeddings = []
    all_visual_embeddings = []
    all_labels = []

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=0, pin_memory=True)

    with torch.no_grad():
        i = 0
        for _, spec, image, label in train_dataloader:
            if args.dataset != 'CGMNIST':
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
            if i >= 10:
                break

    # 合并所有结果
    audio_embeddings = np.concatenate(all_audio_embeddings, axis=0)
    visual_embeddings = np.concatenate(all_visual_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)


    # 聚类
    kmeans_audio, _ = clustering(audio_embeddings, pca=True,
                                 n_components=n_components, n_clusters=n_clusters)
    kmeans_visual, _ = clustering(visual_embeddings, pca=True,
                                  n_components=n_components, n_clusters=n_clusters)

    # 计算PID
    P, _ = convert_data_to_distribution(
        kmeans_audio.reshape(-1, 1),
        kmeans_visual.reshape(-1, 1),
        labels
    )
    measures = get_measure(P)

    return measures


def train_epoch(args, epoch, model, device, dataloader, dataset, optimizer, scheduler, mode, measures):
    criterion = nn.CrossEntropyLoss()
    model.train()

    _loss = 0
    _loss_a = 0
    _loss_v = 0
    print("Start training ... ")

    for _, spec, image, label in dataloader:

        label = label.to(device)
        optimizer.zero_grad()

        if args.dataset != 'CGMNIST':
            out_a, out_v, out = model(spec.unsqueeze(1).float().to(device), image.float().to(device))
        else:
            out_a, out_v, out = model(spec.to(device), image.to(device))

            # 根据不同mode计算loss
        if mode == 'audio':
            loss = criterion(out_a, label)
        elif mode == 'visual':
            loss = criterion(out_v, label)*5
        elif mode == 'unimodal':
            loss_a = criterion(out_a, label)
            loss_v = criterion(out_v, label)
            loss = loss_a + loss_v
        elif mode == 'joint':
            loss = criterion(out, label)
        else:
            raise ValueError(f"Unsupported training mode: {mode}")

        # print(loss.item())

        loss.backward()
        optimizer.step()
    
    if mode=="joint":
        scheduler.step()

    # 如果是probing的epoch
    if epoch > 3 and epoch % args.probing_freq == 0:
        print(f"Performing probing at epoch {epoch}...")
        measures = online_probing(args, model, dataset, device)
        log_print(f"PID measures at epoch {epoch}:"+str(measures))
    return measures



def valid(args, model, device, dataloader, mode):
    softmax = nn.Softmax(dim=1)
    model.eval()

    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'KineticSound':
        n_classes = 31
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'CGMNIST':
        n_classes = 10
    elif args.dataset == 'AVE':
        n_classes = 28
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    with torch.no_grad():
        # 每个类别的样本数和正确预测数
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        for _, spec, image, label in dataloader:
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            if args.dataset != 'CGMNIST':
                out_a, out_v, out = model(spec.unsqueeze(1).float(), image.float())
            else:
                out_a, out_v, out = model(spec.to(device), image.to(device))

    
            # 应用softmax
            prediction = softmax(out)
            pred_v = softmax(out_v)
            pred_a = softmax(out_a)

            # 逐样本统计
            for i in range(image.shape[0]):
                # 更新样本计数
                num[label[i]] += 1.0

                # 获取预测结果
                ma = np.argmax(prediction[i].cpu().data.numpy())
                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())

                # 根据mode更新正确预测计数
                if mode in ['audio', 'unimodal']:
                    if np.asarray(label[i].cpu()) == a:
                        acc_a[label[i]] += 1.0

                if mode in ['visual', 'unimodal']:
                    if np.asarray(label[i].cpu()) == v:
                        acc_v[label[i]] += 1.0

                if mode == 'joint':
                    if np.asarray(label[i].cpu()) == ma:
                        acc[label[i]] += 1.0
                    if np.asarray(label[i].cpu()) == v:
                        acc_v[label[i]] += 1.0
                    if np.asarray(label[i].cpu()) == a:
                        acc_a[label[i]] += 1.0

        # 根据不同模式返回相应的准确率
        if mode == 'audio':
            return None, sum(acc_a) / sum(num), None
        elif mode == 'visual':
            return None, None, sum(acc_v) / sum(num)
        elif mode == 'unimodal':
            return None, sum(acc_a) / sum(num), sum(acc_v) / sum(num)
        elif mode == 'joint':
            return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)


def main():
    log_print(str(args))

    setup_seed(args.random_seed)

    device = torch.device('cuda:{}'.format(args.gpu))


    if args.dataset == 'CGMNIST':
        model = CGClassifier(args)
    else:
        model = AVClassifier(args)

    model.apply(weight_init)
    model.to(device)

    torch.save(model.state_dict(), 'init_para.pkl')
    PATH='init_para.pkl'
    checkpoint = torch.load(PATH)

    # model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    train_dataset = CramedDataset(args, mode='my_train')
    test_dataset = CramedDataset(args, mode='test')
    val_dataset=CramedDataset(args, mode='val')


    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=8, pin_memory=True)
    
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=8, pin_memory=False)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=8, pin_memory=True)

    current_stage = 'unimodal'  # 初始阶段
    best_synergy = -1
    last_synergy = -1
    flag_reinit = 0

    measures = {}
    measures["unique1"] = 1
    measures["unique2"] = 1


    for epoch in range(args.max_unimodal_epochs):
        log_print('Uni Epoch: {}: '.format(epoch))
        measures = train_epoch(args, epoch, model, device, train_dataloader, train_dataset, optimizer, scheduler, current_stage, measures)

        # 验证阶段
        acc, acc_a, acc_v = valid(args, model, device, test_dataloader, current_stage)
        log_print(f'Current mode: {current_stage}')
        if acc is not None:
            log_print(f'Joint Accuracy: {acc:.4f}')
        if acc_a is not None:
            log_print(f'Audio Accuracy: {acc_a:.4f}')
        if acc_v is not None:
            log_print(f'Visual Accuracy: {acc_v:.4f}')

        if measures["unique1"] != 1 and current_stage != "joint":
            
            uniqueness_ratio = measures['unique1'] / measures['unique2']
            if uniqueness_ratio > args.uniqueness_threshold:
                current_stage = "visual"
                print("Stopping audio training due to high uniqueness ratio")
            elif 1 / uniqueness_ratio > args.uniqueness_threshold:
                current_stage = "audio"
                print("Stopping visual training due to high uniqueness ratio")

            # # 检查是否应该开始联合训练
            # if last_synergy !=-1 and measures['synergy'] < last_synergy * args.synergy_threshold:
            #     print("switch the unimodals...")
            #     if current_stage == 'visual':
            #         current_stage = 'audio'
            #     else:
            #         current_stage = 'visual'

            if last_synergy!=-1 and measures['synergy'] < best_synergy * args.best_synergy_threshold:
                print("Synergy decreasing, switching to joint training...")
                current_stage = 'joint'

            if measures['synergy'] > best_synergy:
                best_synergy = measures['synergy']
                
            last_synergy = measures['synergy']

        if current_stage == "joint":
            break

    best_acc = 0.0
    current_stage = "joint"
    optimizer_2 = optim.SGD(model.parameters(), lr=args.joint_lr, momentum=0.9, weight_decay=1e-4)

    for epoch in range(args.max_joint_epochs):
        log_print('Joint Epoch: {}: '.format(epoch))
        measures = train_epoch(args, epoch, model, device, train_dataloader, train_dataset, optimizer_2, scheduler,
                               current_stage, measures)
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

        if((epoch % args.reinit_epoch == 0)&(epoch>0)):
            flag_reinit+=1
            if(flag_reinit<=args.reinit_num):
                print('reinit %d' % flag_reinit)
                print("Start getting training feature ... ")
                train_audio,train_visual,train_label=get_feature(args, epoch, model, device, train_dataloader)
                print("Start getting evluating feature ... ")
                val_audio,val_visual,val_label=get_feature(args, epoch, model, device, val_dataloader)
                weight_audio,weight_visual= reinit_score(args, train_audio,train_visual,train_label,val_audio,val_visual,val_label)
                model=reinit(args, model,checkpoint,weight_audio,weight_visual)

if __name__ == "__main__":
    main()
