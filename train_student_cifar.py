import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os
import shutil
import argparse
import numpy as np


import models
import torchvision
import torchvision.transforms as transforms
from utils import cal_param_size, cal_multi_adds


from bisect import bisect_right
import time
import math
from loss import MetaSGD, GKDLoss


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--data', default='/data/winycg/dataset/', type=str, help='Dataset directory')
parser.add_argument('--dataset', default='cifar100', type=str, help='Dataset name')
parser.add_argument('--arch', default='wrn_16_2_aux', type=str, help='student network architecture')
parser.add_argument('--tarch', default='wrn_40_2_aux', type=str, help='teacher network architecture')
parser.add_argument('--tcheckpoint', default='wrn_40_2_aux.pth.tar', type=str, help='pre-trained weights of teacher')
parser.add_argument('--init-lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay')

parser.add_argument('--times', type=int, default=2, help='times of meta optimization')
parser.add_argument('--meta-freq', type=int, default=50, help='frequency of meta optimization')
parser.add_argument('--lr-type', default='multistep', type=str, help='learning rate strategy')
parser.add_argument('--milestones', default=[150,180,210], type=list, help='milestones for lr-multistep')
parser.add_argument('--sgdr-t', default=300, type=int, dest='sgdr_t',help='SGDR T_0')
parser.add_argument('--warmup-epoch', default=0, type=int, help='warmup epoch')
parser.add_argument('--epochs', type=int, default=240, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--num-workers', type=int, default=8, help='the number of workers')
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--kd_T', type=float, default=3, help='temperature for KD distillation')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')
parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='directory fot storing checkpoints')

# global hyperparameter set
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

log_txt = str(os.path.basename(__file__).split('.')[0]) + '_'+\
          'tarch' + '_' +  args.tarch + '_'+\
          'arch' + '_' +  args.arch + '_'+\
          'dataset' + '_' +  args.dataset + '_'+\
          'seed'+ str(args.manual_seed) +'.txt'

log_dir = str(os.path.basename(__file__).split('.')[0]) + '_'+\
          'tarch' + '_' +  args.tarch + '_'+\
          'arch'+ '_' + args.arch + '_'+\
          'dataset' + '_' +  args.dataset + '_'+\
          'seed'+ str(args.manual_seed)


args.checkpoint_dir = os.path.join(args.checkpoint_dir, log_dir)
if not os.path.isdir(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)
log_txt = os.path.join(args.checkpoint_dir, log_txt)
if args.resume is False and args.evaluate is False:
    with open(log_txt, 'a+') as f:
        f.write("==========\nArgs:{}\n==========".format(args) + '\n')

np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
torch.set_printoptions(precision=4)


num_classes = 100
trainset = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                [0.2675, 0.2565, 0.2761])
                                        ]))

testset = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                [0.2675, 0.2565, 0.2761]),
                                        ]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                    pin_memory=(torch.cuda.is_available()))

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                    pin_memory=(torch.cuda.is_available()))

print('==> Building model..')
net = getattr(models, args.tarch)(num_classes=num_classes)
net.eval()
resolution = (1, 3, 32, 32)
print('Teacher Arch: %s, Params: %.2fM, Multi-adds: %.2fG'
        % (args.tarch, cal_param_size(net)/1e6, cal_multi_adds(net, resolution)/1e9))
del(net)
net = getattr(models, args.arch)(num_classes=num_classes)
net.eval()
resolution = (1, 3, 32, 32)
print('Student Arch: %s, Params: %.2fM, Multi-adds: %.2fG'
        % (args.arch, cal_param_size(net)/1e6, cal_multi_adds(net, resolution)/1e9))
del(net)


print('load pre-trained teacher weights from: {}'.format(args.tcheckpoint))     
checkpoint = torch.load(args.tcheckpoint, map_location=torch.device('cpu'))

model = getattr(models, args.arch)
net = model(num_classes=num_classes).cuda()
net =  torch.nn.DataParallel(net)

tmodel = getattr(models, args.tarch)
tnet = tmodel(num_classes=num_classes).cuda()
tnet.load_state_dict(checkpoint['net'])
tnet.eval()
tnet =  torch.nn.DataParallel(tnet)

_, ss_logits = net(torch.randn(2, 3, 32, 32))
num_auxiliary_branches = len(ss_logits)
cudnn.benchmark = True


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
        return loss


def correct_num(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:, :k].float().sum()
        res.append(correct_k)
    return res


def adjust_lr(optimizer, epoch, args, step=0, all_iters_per_epoch=0):
    cur_lr = 0.
    if epoch < args.warmup_epoch:
        cur_lr = args.init_lr * float(1 + step + epoch*all_iters_per_epoch)/(args.warmup_epoch *all_iters_per_epoch)
    else:
        epoch = epoch - args.warmup_epoch
        cur_lr = args.init_lr * 0.1 ** bisect_right(args.milestones, epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return cur_lr


def inner_objective(data):
    inputs, targets = data[0], data[1]
    
    s_logits, s_feats = net(inputs)
    with torch.no_grad():
        t_logits, t_feats = tnet(inputs)
    
    t_feats = [t_gcn[k](t_feats[k]) for k in range(len(t_feats))]
    s_feats = [s_gcn[k](s_feats[k]) for k in range(len(s_feats))]

    weights = LossWeightNetwork(s_feats, t_feats)
    feature_kd_loss, relation_kd_loss = criterion_rkd(t_feats, s_feats, weights)
               
    return feature_kd_loss + relation_kd_loss
    
def outer_objective(data):
    inputs, target = data[0], data[1]
    s_logits, s_feats = net(inputs)
    with torch.no_grad():
        t_logits, t_feats = tnet(inputs)
    
    loss_cls = torch.tensor(0.).cuda()
    loss_div = torch.tensor(0.).cuda()
    for i in range(len(s_logits)):
        loss_cls = loss_cls + criterion_cls(s_logits[i], target)
        loss_div = loss_div + criterion_div(s_logits[i], t_logits[i].detach())

    loss_cls = loss_cls / len(s_logits)
    loss_div = loss_div / len(s_logits)
    return loss_cls + loss_div


def train(epoch, criterion_list, optimizer, meta_optimizer, weight_optimizer):
    train_loss = 0.
    train_loss_cls = 0.
    train_loss_div = 0.
    train_loss_feature_kd = 0.
    train_loss_relation_kd = 0.

    class_top1_num = [0] * num_auxiliary_branches
    class_top5_num = [0] * num_auxiliary_branches
    top1_num = 0
    top5_num = 0
    total = 0

    if epoch >= args.warmup_epoch:
        lr = adjust_lr(optimizer, epoch, args)

    start_time = time.time()
    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_rkd = criterion_list[2]
    t_gcn = criterion_list[3]
    s_gcn = criterion_list[4]
    teacher_gcn_classifier = criterion_list[5]

    net.train()
    t_gcn.train()
    s_gcn.train()
    teacher_gcn_classifier.train()

    for batch_idx, (inputs, target) in enumerate(trainloader):
        batch_start_time = time.time()
        inputs = inputs.float().cuda()
        target = target.cuda()

        if epoch < args.warmup_epoch:
            lr = adjust_lr(optimizer, epoch, args, batch_idx, len(trainloader))

        optimizer.zero_grad()
        s_logits, s_feats = net(inputs)
        with torch.no_grad():
            t_logits, t_feats = tnet(inputs)


        t_feats = [t_gcn[k](t_feats[k]) for k in range(len(t_feats))]
        s_feats = [s_gcn[k](s_feats[k]) for k in range(len(s_feats))]

        with torch.no_grad():
            weights = LossWeightNetwork(s_feats, t_feats)

        loss_cls = torch.tensor(0.).cuda()
        loss_div = torch.tensor(0.).cuda()

        t_loss_cls = torch.tensor(0.).cuda()
        for k in range(len(t_feats)): 
            t_loss_cls += criterion_cls(teacher_gcn_classifier(t_feats[k]), target)

        for i in range(len(s_logits)):
            loss_cls = loss_cls + criterion_cls(s_logits[i], target)
            loss_div = loss_div + criterion_div(s_logits[i], t_logits[i].detach())
        loss_cls = loss_cls / len(s_logits)
        loss_div = loss_div / len(s_logits)
        feature_kd_loss, relation_kd_loss = criterion_rkd(t_feats, s_feats, weights)

        loss = loss_cls + loss_div + t_loss_cls + feature_kd_loss + relation_kd_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item() / len(trainloader)
        train_loss_cls += loss_cls.item() / len(trainloader)
        train_loss_div += loss_div.item() / len(trainloader)
        train_loss_feature_kd += feature_kd_loss.item() / len(trainloader)
        train_loss_relation_kd += relation_kd_loss.item() / len(trainloader)

        for i in range(len(s_logits)):
            top1, top5 = correct_num(s_logits[i], target, topk=(1, 5))
            class_top1_num[i] += top1
            class_top5_num[i] += top5

        total += target.size(0)

        print('Epoch:{}, batch_idx:{}/{}, lr:{:.5f}, Duration:{:.2f}, train_loss:{:.5f}'
              ',train_loss_cls:{:.5f},train_loss_div:{:.5f},train_loss_feature_kd:{:.5f},train_loss_relation_kd:{:.5f},Top-1 Acc:{:.4f}'.format(
            epoch, batch_idx, len(trainloader), lr, time.time()-batch_start_time, 
            train_loss, train_loss_cls, train_loss_div, 
            train_loss_feature_kd, train_loss_relation_kd,
            (class_top1_num[-1]/(total)).item()))

        
        if batch_idx % args.meta_freq == 0:
            print('Perform meta-optimization...')
            data = (inputs, target)
            for _ in range(args.times):
                print('Meta-optimization iteration '+str(_))
                meta_optimizer.zero_grad()
                meta_optimizer.step(inner_objective, data)

            meta_optimizer.zero_grad()
            meta_optimizer.step(outer_objective, data)

            meta_optimizer.zero_grad()
            weight_optimizer.zero_grad()
            outer_objective(data).backward()
            meta_optimizer.meta_backward()
            weight_optimizer.step()
            print('Finish meta-optimization.')


    class_acc1 = [round((class_top1_num[i]/(total)).item(), 4) for i in range(num_auxiliary_branches)]
    class_acc5 = [round((class_top5_num[i]/(total)).item(), 4) for i in range(num_auxiliary_branches)]
    
    print('Train epoch:{}\nTrain Top-1 class_accuracy: {}\n'.format(epoch, str(class_acc1)))

    with open(log_txt, 'a+') as f:
        f.write('Epoch:{}\t lr:{:.5f}\t duration:{:.3f}'
                '\ntrain_loss:{:.5f}\t train_loss_cls:{:.5f}\t train_loss_div:{:.5f}\t train_loss_feature_kd:{:.5f}\t train_loss_relation_kd:{:.5f}'
                '\nTrain Top-1 class_accuracy: {}\n'
                .format(epoch, lr, time.time() - start_time,
                        train_loss, train_loss_cls, train_loss_div,
                        train_loss_feature_kd, train_loss_relation_kd,
                        str(class_acc1)))



def test(epoch, criterion_cls, net):
    global best_acc
    test_loss_cls = 0.

    class_top1_num = [0] * num_auxiliary_branches
    class_top5_num = [0] * num_auxiliary_branches
    top1_num = 0
    top5_num = 0
    total = 0
    
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(testloader):
            batch_start_time = time.time()
            inputs, target = inputs.cuda(), target.cuda()
            
            s_logits, s_feats = net(inputs)
            loss_cls = torch.tensor(0.).cuda()
            loss_cls = loss_cls + criterion_cls(s_logits[-1], target)

            test_loss_cls += loss_cls.item()/ len(testloader)

            for i in range(len(s_logits)):
                top1, top5 = correct_num(s_logits[i], target, topk=(1, 5))
                class_top1_num[i] += top1
                class_top5_num[i] += top5
            total += target.size(0)
            

            print('Epoch:{}, batch_idx:{}/{}, Duration:{:.2f}, Top-1 Acc:{:.4f}'.format(
                epoch, batch_idx, len(testloader), time.time()-batch_start_time, (class_top1_num[-1]/(total))))

        class_acc1 = [round((class_top1_num[i]/(total)).item(), 4) for i in range(num_auxiliary_branches)]
        class_acc5 = [round((class_top5_num[i]/(total)).item(), 4) for i in range(num_auxiliary_branches)]
        with open(log_txt, 'a+') as f:
            f.write('test epoch:{}\t test_loss_cls:{:.5f}\nTop-1 class_accuracy: {}\n'
                    .format(epoch, test_loss_cls, str(class_acc1)))
        print('test epoch:{}\nTest Top-1 class_accuracy: {}'.format(epoch, str(class_acc1)))

    return class_acc1[-1]


if __name__ == '__main__':
    best_acc = 0.  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.kd_T)
    criterion_rkd = GKDLoss()

    data = torch.randn(1, 3, 32, 32).cuda()
    net.eval()
    s_logits, s_embedding = net(data)
    t_logits, t_embedding = tnet(data)
    args.number_stage = len(s_logits)
    
    args.s_rep_dim = []
    args.t_rep_dim = []
    s_gcn = nn.ModuleList([])
    t_gcn = nn.ModuleList([])
    for i in range(args.number_stage):
        args.s_rep_dim.append(s_embedding[i].size(1))
        s_gcn.append(getattr(models, 'GCN')(s_embedding[i].size(1), t_embedding[i].size(1)))
    for i in range(args.number_stage):
        args.t_rep_dim.append(t_embedding[i].size(1))
        t_gcn.append(getattr(models, 'GCN')(t_embedding[i].size(1), t_embedding[i].size(1)))
    teacher_gcn_classifier = nn.Linear(t_embedding[i].size(1), num_classes).cuda()


    if args.evaluate: 
        print('load pre-trained weights from: {}'.format(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar')))     
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
                                map_location=torch.device('cpu'))
        net.module.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        test(start_epoch, criterion_cls, net)
    else:
        print('Evaluate Teacher:')
        #acc = test(0, criterion_cls, tnet)
        #print('Teacher Acc:', acc)

        LossWeightNetwork = getattr(models, 'LossWeightNetwork')(args.t_rep_dim, args.t_rep_dim).cuda()
        target_params = list(net.parameters()) + list(t_gcn.parameters()) + list(s_gcn.parameters())
        meta_optimizer = MetaSGD(target_params,
                                    [net, t_gcn, s_gcn],
                                    lr=args.init_lr,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay, 
                                    rollback=True, cpu=args.times>2)
            
        weight_optimizer = optim.SGD(LossWeightNetwork.parameters(),
                                    lr=args.init_lr,
                                    momentum=0.9, 
                                    weight_decay=args.weight_decay, nesterov=True)

        trainable_list = nn.ModuleList([])
        trainable_list.append(net)
        trainable_list.append(t_gcn)
        trainable_list.append(s_gcn)
        trainable_list.append(teacher_gcn_classifier)
        optimizer = optim.SGD(trainable_list.parameters(),
                              lr=0.1, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

        criterion_list = nn.ModuleList([])
        criterion_list.append(criterion_cls)
        criterion_list.append(criterion_div)  
        criterion_list.append(criterion_rkd)
        criterion_list.append(t_gcn)
        criterion_list.append(s_gcn)
        criterion_list.append(teacher_gcn_classifier)
        criterion_list.cuda()


        if args.resume:
            print('load pre-trained weights from: {}'.format(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar')))
            checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
                                    map_location=torch.device('cpu'))
            net.module.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch'] + 1

        for epoch in range(start_epoch, args.epochs):
            train(epoch, criterion_list, optimizer, meta_optimizer, weight_optimizer)
            acc = test(epoch, criterion_cls, net)

            state = {
                'net': net.module.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'))

            is_best = False
            if best_acc < acc:
                best_acc = acc
                is_best = True

            if is_best:
                shutil.copyfile(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
                                os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'))

        print('Evaluate the best model:')
        print('load pre-trained weights from: {}'.format(os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar')))
        args.evaluate = True
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'),
                                map_location=torch.device('cpu'))
        net.module.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        top1_acc = test(start_epoch, criterion_cls, net)
