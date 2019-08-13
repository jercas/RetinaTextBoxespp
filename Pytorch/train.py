from __future__ import print_function

import time
import os
import argparse
import numpy as np
import cv2
from subprocess import Popen, PIPE

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset

from tensorboardX import SummaryWriter
from augmentations import Augmentation_traininig

from loss import FocalLoss, OHEM_loss
from retinanet import RetinaNet
from datagen import ListDataset
from encoder import DataEncoder

from torch.autograd import Variable

# Indicate visible gpu device
device_ids = [3,4,5,6]
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, device_ids))

# Multi-semantic transform
def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

# Dynamic adjust lr
def adjust_learning_rate(cur_lr, optimizer, gamma, step):
    lr = cur_lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# usage:
# CUDA_VISIBLE_DEVICES=6,7 python train.py --root=./DB/ --dataset=PLATE --batch_size=16 --multi_scale=True --logdir=logs/multi_step1/ --save_folder=models/multi_step1/ --num_workers=6
parser = argparse.ArgumentParser(description='PyTorch RetinaTextBoxes++ Training')
parser.add_argument('--root', default='./DB/',
                                                        type=str, help='root of the dataset dir')
parser.add_argument('--lr', default=1e-3,
                                                        type=float, help='learning rate')
parser.add_argument('--input_size', default=768,
                                                        type=int, help='Input size for training')
parser.add_argument('--batch_size', default=8,
                                                        type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=8,
                                                        type=int, help='Number of workers used in data loading')
parser.add_argument('--resume', default=None,
                                                        type=str,  help='resume from checkpoint')
parser.add_argument('--dataset', default='ICDAR2015',
                                                        type=str, help='select training dataset')
parser.add_argument('--multi_scale', default=False,
                                                        type=str2bool, help='Use multi-scale training')
parser.add_argument('--focal_loss', default=True,
                                                        type=str2bool, help='Use Focal loss or OHEM loss')
parser.add_argument('--logdir', default='./logs/',
                                                        type=str, help='Tensorboard log dir')
parser.add_argument('--max_iter', default=1200000,
                                                        type=int, help='Number of training iterations')
parser.add_argument('--gamma', default=0.5,
                                                        type=float, help='Gamma update for SGD')
parser.add_argument('--save_interval', default=500,
                                                        type=int, help='Frequency for saving checkpoint models')
parser.add_argument('--save_folder', default='./models/',
                                                        type=str, help='Location to save checkpoint models')
parser.add_argument('--evaluation', default=True,
                                                        type=str2bool, help='Evaulation during training')
parser.add_argument('--eval_step', default=1000,
                                                        type=int, help='Evauation step')
parser.add_argument('--eval_device', default=2,
                                                        type=int, help='GPU device for evaluation')
parser.add_argument('--cls_thresh', default=0.5,
                                                        type=int, help='classification thresh')
parser.add_argument('--nms_thresh', default=0.2,
                                                        type=int, help='nms thresh')
args = parser.parse_args()

# confirm GPU & Focal loss use
assert torch.cuda.is_available(), 'Error: CUDA not found!'
assert args.focal_loss, "OHEM + ce_loss is not working... :("

# confirm existence of the folder for saving model and log if there are not exist, if not create them.
if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if not os.path.exists(args.logdir):
    os.mkdir(args.logdir)

# Data load
print('==> Preparing data..')
trainset = ListDataset(root=args.root, dataset=args.dataset, train=True,
                       transform=Augmentation_traininig, input_size=args.input_size, multi_scale=args.multi_scale)

trainloader = DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, collate_fn=trainset.collate_fn, num_workers=args.num_workers)

# Set model (focal_loss vs OHEM_CE loss)
# Backbone - se-resnet50
print('==>loss initializing...\n')
if args.focal_loss:
    imagenet_pretrain = 'weights/retinanet_se50.pth'
    criterion = FocalLoss()
    num_classes = 1
else:
    imagenet_pretrain = 'weights/retinanet_se50_OHEM.pth'
    criterion = OHEM_loss()
    num_classes = 2

# Training Detail option
stepvalues = (10000, 20000, 30000, 40000, 50000) if args.dataset in ["SynthText"] else (2000, 4000, 6000, 8000, 10000)
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch
iteration = 0
cur_lr = args.lr
mean=(0.485,0.456,0.406)
var=(0.229,0.224,0.225)
step_index = 0
pEval = None

# Model
print('==>network establishing...\n')
net = RetinaNet(num_classes)
net.load_state_dict(torch.load(imagenet_pretrain))

# Resume training if there are any break off
if args.resume:
    print('==> Resuming from checkpoint..', args.resume)
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']
    iteration = checkpoint['iteration']
    cur_lr = checkpoint['lr']
    step_index = checkpoint['step_index']
    print(" net: {0}\n start_epoch: {1}\n iteration: {2}\n current_lr: {3}\n step_index: {4}\n".format(
        start_epoch, iteration, cur_lr,step_index))
    #optimizer.load_state_dict(state["optimizer"])

print('==>training detail...\n')
print("multi_scale : ", args.multi_scale)
print("input_size : ", args.input_size)
print("stepvalues : ", stepvalues)
print("start_epoch : ", start_epoch)
print("iteration : ", iteration)
print("cur_lr : ", cur_lr)
print("step_index : ", step_index)
print("gpu available : ", torch.cuda.is_available())
print("num_gpus : ", torch.cuda.device_count())

# Set data parallel training
net = torch.nn.DataParallel(net, device_ids=[0,1,2,3])
net.cuda()

# Training
print("==>training start...")
net.train()
# Freeze BN layer for pre-trained backbone
net.module.freeze_bn()
# Set optimizer -- SGD or Adam
optimizer = optim.SGD(net.parameters(), lr=cur_lr, momentum=0.9, weight_decay=1e-4) #optim.Adam(net.parameters(), lr=cur_lr)
# Encode anchor to each feature maps
encoder = DataEncoder(cls_thresh=0.5, nms_thresh=0.2)
# Tensorboard visualize recorder
writer = SummaryWriter(logdir=args.logdir)

t0 = time.time()
for epoch in range(start_epoch, 10000):
    if iteration > args.max_iter:
        break

    for inputs, loc_targets, cls_targets in trainloader:
        # prepare data and cls & loc label
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())
        optimizer.zero_grad()
        # predict result
        loc_preds, cls_preds = net(inputs)
        # get the loss between prediction and ground truth
        loc_loss, cls_loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        # total loss
        loss = loc_loss + cls_loss
        # bp
        loss.backward()
        # optimizing - stochastic gradient descendent
        optimizer.step()

        # Recording intermediate log
        if iteration % 20 == 0:
            t1 = time.time()
            print('iter ' + repr(iteration) + ' (epoch ' + repr(epoch) + ') || loss: %.4f || l loc_loss: %.4f || l cls_loss: %.4f (Time : %.1f)'\
                 % (loss.sum().item(), loc_loss.sum().item(), cls_loss.sum().item(), (t1 - t0)))
            t0 = time.time()
            # record log and visualization by tensorboard
            writer.add_scalar('loc_loss', loc_loss.sum().item(), iteration)
            writer.add_scalar('cls_loss', cls_loss.sum().item(), iteration)
            writer.add_scalar('loss', loss.sum().item(), iteration)

            # show inference image in tensorboard
            infer_img = np.transpose(inputs[0].cpu().numpy(), (1,2,0))
            infer_img *= var
            infer_img += mean
            infer_img *= 255.
            infer_img = np.clip(infer_img, 0, 255)
            infer_img = infer_img.astype(np.uint8)
            h, w, _ = infer_img.shape

            boxes, labels, scores = encoder.decode(loc_preds[0], cls_preds[0], (w,h))
            boxes = boxes.reshape(-1, 4, 2).astype(np.int32)

            if boxes.shape[0] is not 0:
                infer_img = cv2.polylines(infer_img, boxes, True, (0,255,0), 4)

            writer.add_image('image', img_tensor=infer_img, global_step=iteration, dataformats='HWC')
            writer.add_scalar('input_size', h, iteration)
            writer.add_scalar('learning_rate', cur_lr, iteration)
            t0 = time.time()

        # Saving intermediate model
        if iteration % args.save_interval == 0 and iteration > 0:
            print('Saving state, iter : ', iteration)
            state = {
                'net': net.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                'iteration' : iteration,
                'epoch': epoch,
                'lr' : cur_lr,
                'step_index' : step_index
            }
            model_file = args.save_folder + 'ckpt_' + repr(iteration) + '.pth'
            torch.save(state, model_file)

        if iteration in stepvalues:
            step_index += 1
            cur_lr = adjust_learning_rate(cur_lr, optimizer, args.gamma, step_index)

        if iteration > args.max_iter:
            break

        # Evaluation while training
        if args.evaluation and iteration % args.eval_step == 0:
            try:
                if pEval is None:
                    print("Evaluation started at iteration {0} on {1}...".format(iteration, args.dataset))
                    eval_cmd = "CUDA_VISIBLE_DEVICES=" + str(args.eval_device) + \
                                    " python eval.py" + \
                                    " --tune_from=" + args.save_folder + 'ckpt_' + repr(iteration) + '.pth' + \
                                    " --input_size=1024" + \
                                    " --output_zip=result_temp1"

                    pEval = Popen(eval_cmd, shell=True, stdout=PIPE, stderr=PIPE)

                elif pEval.poll() is not None:
                    (scoreString, stdErrData) = pEval.communicate()

                    hmean = float(str(scoreString).strip().split(":")[3].split(",")[0].split("}")[0].strip())

                    writer.add_scalar('test_hmean', hmean, iteration)
                    
                    print("test_hmean for {}-th iter : {:.4f}".format(iteration, hmean))

                    if pEval is not None:
                        pEval.kill()
                    pEval = None

            except Exception as e:
                print("exception happened in evaluation ", e)
                if pEval is not None:
                    pEval.kill()
                pEval = None

        iteration += 1