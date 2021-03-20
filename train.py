import torch
import torch.nn as nn
import numpy as np
from models import ModelBuilder
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from dataset import TrainDataset
import os
import nibabel as nib
import argparse
from utils import AverageMeter
from distutils.version import LooseVersion
from loss_new import DICELossMultiClass_relax_beta
import math
import time

def train(train_loader, model,  criterion, optimizer, epoch, args):
    losses = AverageMeter()
    class1 = AverageMeter()
    class2 = AverageMeter()
    class3 = AverageMeter()
    model.train()
    for iteration, sample in enumerate(train_loader):
        image = sample['images'].float()
        sym = sample['sym'].float()
        target = sample['labels'].long()
        if args.relaxation:
            relax_mask = sample['relax'].long()
        image = Variable(image).cuda()
        label = Variable(target)
        # The dimension of out should be in the dimension of B,C,W,H,D
        # transform the prediction and label
        out = model(image, sym)

        out = out.permute(0,2,3,4,1).contiguous().view(-1, args.num_classes)
        label = label.contiguous().view(-1)

        loss, class_loss = criterion(out, label, relax_mask, epoch,  100, args.class_focal)

        losses.update(loss.data.item(),image.size(0))
        class1.update(class_loss[0], image.size(0))
        class2.update(class_loss[1], image.size(0))
        class3.update(class_loss[2], image.size(0))
        del image
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # adjust learning rate
        cur_iter = iteration + (epoch - 1) * args.epoch_iters
        adjust_learning_rate(optimizer, cur_iter, args)
        if iteration % 50 ==0:
            print('   * i {} |  lr: {:.6f} | Training Loss: {losses.avg:.3f}'.format(iteration, args.running_lr, losses=losses))

    print('   * EPOCH {epoch} | Training Loss: {losses.avg:.3f}'.format(epoch=epoch, losses=losses))
    return class1.avg, class2.avg, class3.avg

def critical_point(beta):
    if beta ==1:
        x = np.nan
    else:
        x = beta**(beta/(1-beta))
    return x

def increase_grad(dice, beta, delta_beta):
    x = critical_point(beta)
    if beta > 1:
        if dice < x: beta += delta_beta
        elif dice >= x: beta -= delta_beta
    elif beta < 1:
        if dice < x: beta -= delta_beta
        elif dice >= x: beta += delta_beta
    else:
        xa= critical_point(beta+delta_beta)
        xb= critical_point(beta-delta_beta)
        if dice < xa:
            beta += delta_beta
        elif dice > xb:
            beta -= delta_beta
    return beta

def decrease_grad(dice, beta, delta_beta):
    x = critical_point(beta)
    if beta >1 :
        if dice < x: beta -= delta_beta
        elif dice >= x: beta += delta_beta
    elif beta <1:
        if dice < x: beta += delta_beta
        elif dice >= x: beta -= delta_beta
    else:
        xa= critical_point(beta+delta_beta)
        xb= critical_point(beta-delta_beta)
        if dice < xa:
            beta -= delta_beta
        else:
            beta += delta_beta
    return beta


def ranking(dice_avg, beta, delta_dice, delta_beta, max_beta, min_beta):
    indmax = np.argmax(dice_avg)
    indmin = np.argmin(dice_avg)
    if dice_avg[indmax] - dice_avg[indmin] > delta_dice :
        indmed = [0,1,2]
        for i in range(3):
            if i != indmax and i != indmin:
                indmed = i
                break

        if dice_avg[indmax] - dice_avg[indmed] < delta_dice and dice_avg[indmed] - dice_avg[indmin] > delta_dice:
            beta[indmin] = increase_grad(dice_avg[indmin], beta[indmin], delta_beta)
        if dice_avg[indmax] - dice_avg[indmed] > delta_dice and dice_avg[indmed] - dice_avg[indmin] < delta_dice:
            beta[indmax] = decrease_grad(dice_avg[indmax], beta[indmax], delta_beta)
        if dice_avg[indmax] - dice_avg[indmed] > delta_dice and dice_avg[indmed] - dice_avg[indmin] > delta_dice:
            beta[indmin] = increase_grad(dice_avg[indmin], beta[indmin], delta_beta)

    for i in range(3):
        if dice_avg[i]>critical_point(beta[i]) and beta[i] >=1:
            beta [i] -= delta_beta

    if max(beta) > max_beta or min(beta) < min_beta :
        for i in range (3):
            beta[i] = max(min_beta,beta[i])
            beta[i] = min(beta[i], max_beta)

    return beta

def save_checkpoint(state, epoch, args):
    filename = args.ckpt + '/' + str(epoch) + '_checkpoint.pth.tar'
    print(filename)
    torch.save(state, filename)

def adjust_learning_rate(optimizer, cur_iter, args):
    scale_running_lr = ((1. - float(cur_iter) / args.max_iters) ** args.lr_pow)
    args.running_lr = args.lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.running_lr

def main(args):
    # import network architecture
    builder = ModelBuilder()
    model = builder.build_net(
            arch=args.id,
            num_input=args.num_input,
            num_classes=args.num_classes,
            num_branches=args.num_branches,
            padding_list=args.padding_list,
            dilation_list=args.dilation_list)
    model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpus))).cuda()
    cudnn.benchmark = True

    # collect the number of parameters in the network
    print("------------------------------------------")
    print("Network Architecture of Model %s:" % (args.id))
    num_para = 0
    for name, param in model.named_parameters():
        num_mul = 1
        for x in param.size():
            num_mul *= x
        num_para += num_mul

    print(model)
    print("Number of trainable parameters %d in Model %s" % (num_para, args.id))
    print("------------------------------------------")

    # set the optimizer and loss
    optimizer = optim.Adam(model.parameters(), args.lr, eps=args.eps, weight_decay=args.weight_decay)
    criterion= DICELossMultiClass_relax_beta(weight= args.class_weight_dice,focal=args.class_focal, relaxation = args.relaxation)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['opt_dict'])
            print("=> Loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> No checkpoint found at '{}'".format(args.resume))

    # loading data
    tf = TrainDataset(train_dir, args)
    train_loader = DataLoader(tf, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, pin_memory=True)

    print("Start training ...")
    dice = np.zeros((3,args.epoch_length))

    for epoch in range(args.start_epoch + 1, args.num_epochs + 1):

        start_time = time.time()
        dice1, dice2, dice3 = train(train_loader, model,  criterion, optimizer, epoch, args)

        # save models
        if epoch > args.particular_epoch:
            if epoch % args.save_epochs_steps == 0:
                save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}, epoch, args)

        # update focal parameter beta
        ind = epoch % args.epoch_length
        dice[0,ind] = dice1
        dice[1,ind] = dice2
        dice[2,ind] = dice3

        if ind ==0:
            dice_avg = np.average(dice,1)
            print ('dice and beta: ', dice_avg, args.class_focal)
            args.class_focal = ranking(dice_avg, args.class_focal, args.delta_dice, args.delta_beta, args.max_beta, args.min_beta)
            print ('updated beta: ', args.class_focal)

        elapsed_time = time.time() - start_time
        print('     epoch time ' +str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + ', remaining ' +str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time*(args.num_epochs - epoch)))))
    print("Training Done")

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.3.0'), \
        'PyTorch>=0.3.0 is required'

    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='relax_focus',
                        help='a name for identitying the model. Choose from the following options: AFN1-6, Basic, ASPP_c, ASPP_s.')
    parser.add_argument('--padding_list', default=[0,4,6,8], nargs='+', type=int,#[0,4,8,12]
                        help='list of the paddings in the parallel convolutions')
    parser.add_argument('--dilation_list', default=[2,6,10,14], nargs='+', type=int,#default=[2,6,10,14]
                        help='list of the dilation rates in the parallel convolutions')
    parser.add_argument('--num_branches', default=4, type=int,
                        help='the number of parallel convolutions in autofocus layer')

    # Path related arguments
    parser.add_argument('--train_path', default='/home/wynonna/Documents/Research/BRATS2019/datalist/trainfold1.txt',
                        help='text file of the name of training data')
    parser.add_argument('--root_path', default='./',
                        help='root directory of data')
    parser.add_argument('--ckpt', default='/home/wynonna/Documents/Research/BRATS2019/saved_models',
                        help='folder to output checkpoints')

    # Data related arguments
    parser.add_argument('--crop_size', default=[64,64,64], nargs='+', type=int, #[78,78,78] [64,64,64]
                        help='crop size of the input image (int or list)')
    parser.add_argument('--center_size', default=[64,64,64], nargs='+', type=int, #[64,64,64]
                        help='the corresponding output size of the input image (int or list)')
    parser.add_argument('--num_classes', default=4, type=int,
                        help='number of classes')
    parser.add_argument('--dice', default=True, type=bool,
                        help='dice loss involved')
    parser.add_argument('--num_input', default=5, type=int,
                        help='number of input image for each patient plus the mask')
    parser.add_argument('--num_workers', default=5, type=int,
                        help='number of data loading workers')
    parser.add_argument('--random_flip', default=True, type=bool,
                        help='if horizontally flip images when training')
    parser.add_argument('--normalization', default=True, type=bool,
                        help='normalizae the data before training')
    parser.add_argument('--shuffle', default=True, type=bool,
                        help='if shuffle the data during training')
    parser.add_argument('--mask', default=True, type=bool,
                        help='if have the mask')
    parser.add_argument('--class_weight_ce', default=[0.1, 1.0,1.45,4.68], nargs='+', type=float,
                        help='')
    parser.add_argument('--class_weight_dice', default=[1.0,1.45,2.68], nargs='+', type=float,
                        help='')
    parser.add_argument('--class_focal', default=[2.0,2.0,2.0], nargs='+', type=float,
                        help='')
    parser.add_argument('--dice_ce', default=[0.5,1.0], nargs='+', type=float,
                        help='')
    parser.add_argument('--epoch_length', default=5, type=int,
                        help='for adjusting focal parameter beta')
    parser.add_argument('--delta_dice', default=0.1, type=float,
                        help='threshold ')
    parser.add_argument('--delta_beta', default=0.2, type=float,
                        help='threshold ')
    parser.add_argument('--max_beta', default=4.0, type=float,
                        help=' ')
    parser.add_argument('--min_beta', default=0.5, type=float,
                        help=' ')
    parser.add_argument('--relaxation', default=4, type=int,
                        help='number of data loading workers')
    parser.add_argument('--epoch_step', default=100, type=int,
                        help='number of data loading workers')
    # optimization related arguments
    parser.add_argument('--num_gpus', default=1, type=int, help='number of GPUs to use')
    parser.add_argument('--batch_size', default=3, type=int,
                        help='training batch size')
    parser.add_argument('--num_epochs', default=100, type=int,
                        help='epochs for training')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--lr', default=4e-4, type=float,
                        help='start learning rate')
    parser.add_argument('--lr_pow', default=0.90, type=float,
                        help='power in poly to drop learning rate')
    parser.add_argument('--optim', default='RMSprop', help='optimizer')
    parser.add_argument('--alpha', default='0.9', type=float, help='alpha in RMSprop')
    parser.add_argument('--eps', default=10**(-4), type=float, help='eps in RMSprop')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weights regularizer')
    parser.add_argument('--momentum', default=0.6, type=float, help='momentum for RMSprop')
    parser.add_argument('--save_epochs_steps', default=5, type=int,
                        help='frequency to save models after a particular number of epochs')
    parser.add_argument('--particular_epoch', default=1, type=int,
                        help='after this number, we will save models more frequently')
    parser.add_argument('--resume', default='',
                        help='the checkpoint that resumes from')
    parser.add_argument('--num_round', default=100, type=int)

    args = parser.parse_args()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    train_file = open(args.train_path, 'r')
    train_dir = train_file.readlines()

    args.ckpt = os.path.join(args.ckpt, args.id, str(args.num_round))
    print('Models are saved at %s' % (args.ckpt))

    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

    if args.start_epoch > 1:
        args.resume = args.ckpt + '/' + str(args.start_epoch) + '_checkpoint.pth.tar'

    args.running_lr = args.lr
    args.epoch_iters = math.ceil(int(len(train_dir)/args.num_input)/args.batch_size)
    args.max_iters = args.epoch_iters * args.num_epochs

    assert len(args.padding_list) == args.num_branches, \
        '# parallel convolutions should be the same as the length of padding list'

    assert len(args.dilation_list) == args.num_branches, \
        '# parallel convolutions should be the same as # dilation rates'

    assert isinstance(args.crop_size, (int, list))
    if isinstance(args.crop_size, int):
        args.crop_size = [args.crop_size, args.crop_size, args.crop_size]

    assert isinstance(args.center_size, (int, list))
    if isinstance(args.center_size, int):
        args.center_size = [args.center_size, args.center_size, args.center_size]

    main(args)
