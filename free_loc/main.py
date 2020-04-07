import argparse
import os
import shutil
import time
import sys
sys.path.insert(0, 'faster_rcnn')
import sklearn
import sklearn.metrics

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision

from datasets.factory import get_imdb
from custom import *
from PIL import Image
np.set_printoptions(threshold=sys.maxsize)


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='localizer_alexnet')
parser.add_argument(
    '-j',
    '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    '--epochs',
    default=30,
    type=int,
    metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=256,
    type=int,
    metavar='N',
    help='mini-batch size (default: 256)')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.1,
    type=float,
    metavar='LR',
    help='initial learning rate')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--weight-decay',
    '--wd',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)')
parser.add_argument(
    '--print-freq',
    '-p',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--eval-freq',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '-e',
    '--evaluate',
    dest='evaluate',
    action='store_true',
    help='evaluate model on validation set')
parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_true',
    help='use pre-trained model')
parser.add_argument(
    '--world-size',
    default=1,
    type=int,
    help='number of distributed processes')
parser.add_argument(
    '--dist-url',
    default='tcp://224.66.41.62:23456',
    type=str,
    help='url used to set up distributed training')
parser.add_argument(
    '--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--vis', action='store_true')

best_prec1 = 0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global args, best_prec1
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'localizer_alexnet':
        model = localizer_alexnet(pretrained=args.pretrained)
    elif args.arch == 'localizer_alexnet_robust':
        model = localizer_alexnet_robust(pretrained=args.pretrained)

    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    # TODO:
    # define loss function (criterion) and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    # TODO: Write code for IMDBDataset in custom.py
    trainval_imdb = get_imdb('voc_2007_trainval')
    test_imdb = get_imdb('voc_2007_test')

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = IMDBDataset(
        trainval_imdb,
        transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        IMDBDataset(
            test_imdb,
            transforms.Compose([
                # transforms.Resize((384, 384)),
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # TODO: Create loggers for visdom and tboard
    # TODO: You can pass the logger objects to train(), make appropriate
    # modifications to train()

    import visdom
    vis = visdom.Visdom(server='http://localhost', port='8097')

    from tensorboardX import SummaryWriter
    logger = SummaryWriter('./runs/q3')




    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, logger, vis)

        # evaluate on validation set
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            m1, m2 = validate(val_loader, model, criterion, epoch, logger, vis)
            score = m1 * m2
            # remember best prec@1 and save checkpoint
            is_best = score > best_prec1
            best_prec1 = max(score, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)




#TODO: You can add input arguments if you wish
def train(train_loader, model, criterion, optimizer, epoch, logger, vis):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.type(torch.FloatTensor).cuda(async=True)
        input_var = input
        target_var = target

        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output
        # TODO: Compute loss using ``criterion``
        optimizer.zero_grad()
        model_output = model(input_var)
        imoutput = nn.AvgPool2d(kernel_size=29, stride=1)(model_output).squeeze()
        loss = criterion(imoutput, target)


        # measure metrics and record loss
        m1 = metric1(imoutput.data, target)
        m2 = metric2(imoutput.data, target)
        losses.update(loss.data[0], input.size(0))
        avg_m1.update(m1[0], input.size(0))
        avg_m2.update(m2[0], input.size(0))

        # TODO:
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))


            #TODO: Visualize things as mentioned in handout
            #TODO: Visualize at appropriate intervals
            logger.add_scalar('train/loss',
                              loss.item() / args.batch_size,
                              epoch * len(train_loader) + i)
            logger.add_scalar('train/metric1',
                              avg_m1.avg,
                              epoch * len(train_loader) + i)
            logger.add_scalar('train/metric2',
                              avg_m2.avg,
                              epoch * len(train_loader) + i)


        # End of train()

class GradCam:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.model.eval()

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        model_output = self.model(x)
        model_output.register_hook(self.save_gradient)
        return model_output

def cam(model, input, target, i):
    gradCam = GradCam(model)

    model.eval()
    input = input[i,...].unsqueeze_(0)
    input.requires_grad_(True)
    target = target[i,...]
    model_output = gradCam.forward_pass(input)
    imoutput = nn.AvgPool2d(kernel_size=29, stride=1)(model_output).view(1,20)
    imoutput = nn.Sigmoid()(imoutput)

    heatmaps = []
    gt_label_index = [k for k in range(target.shape[0]) if target[k]==1]

    for gt_class in gt_label_index:
        one_hot_output = torch.zeros((1, imoutput.shape[-1]), dtype=torch.float, device=torch.device('cuda'))
        one_hot_output[0, gt_class] = 1
        model.features.zero_grad()
        model.classifier.zero_grad()

        imoutput.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = gradCam.gradients.data.cpu().numpy()[0]

        new_target = model_output.data.cpu().numpy()[0]
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        cam = np.ones(new_target.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * new_target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input.shape[2],
                                                    input.shape[3]), Image.ANTIALIAS))
        cam = cam[np.newaxis,...]
        heatmaps.append(cam)
    return input[0], heatmaps


def validate(val_loader, model, criterion, epoch, logger, vis):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    change = True
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.type(torch.FloatTensor).cuda(async=True)
        input_var = input
        target_var = target

        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output
        # TODO: Compute loss using ``criterion``
        model_output = model(input_var)
        imoutput = nn.AvgPool2d(kernel_size=29, stride=1)(model_output).squeeze()



        '''
            heat map visualization
        '''

        if epoch in [0,14,28] and change:
            count = 2
            change = False
            img1, heat_map1 = cam(model, input, target, count*2+3)
            heat_map1 = np.stack(heat_map1, axis=0)
            heat_map1 = torch.from_numpy(heat_map1)
            img2, heat_map2 = cam(model, input, target, count*2+1)
            heat_map2 = np.stack(heat_map2, axis=0)
            heat_map2 = torch.from_numpy(heat_map2)

            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            if epoch==0:
                for t, m, s in zip(img1, mean, std):
                    t.mul_(s).add_(m)

                for t, m, s in zip(img2, mean, std):
                    t.mul_(s).add_(m)
                logger.add_image('image1/origin', img1, epoch * len(val_loader)+count)
                logger.add_image('image2/origin', img2, epoch * len(val_loader)+count)
                img1 = img1.detach().cpu().numpy()
                img2 = img2.detach().cpu().numpy()
                vis.image(img1, opts=dict(title="image1/origin"))
                vis.image(img2, opts=dict(title="image2/origin"))
            logger.add_image('image1/heatmap_{}'.format(epoch),
                             torchvision.utils.make_grid(heat_map1, nrow=heat_map1.shape[0], padding=10),
                             epoch * len(val_loader) + i)

            logger.add_image('image2/heatmap_{}'.format(epoch),
                             torchvision.utils.make_grid(heat_map2, nrow=heat_map2.shape[0], padding=10),
                             epoch * len(val_loader) + i)

            heat_map1 = heat_map1.numpy()
            heat_map2 = heat_map2.numpy()
            vis.images(heat_map1, nrow=heat_map1.shape[0], opts=dict(title='image1/heatmap_{}'.format(epoch)))
            vis.images(heat_map2, nrow=heat_map2.shape[0], opts=dict(title='image2/heatmap_{}'.format(epoch)))

            if epoch==28:
                count = 9
                img1, heat_map1 = cam(model, input, target, count)
                heat_map1 = np.stack(heat_map1, axis=0)
                heat_map1 = torch.from_numpy(heat_map1)
                img2, heat_map2 = cam(model, input, target, count+1)
                heat_map2 = np.stack(heat_map2, axis=0)
                heat_map2 = torch.from_numpy(heat_map2)
                img3, heat_map3 = cam(model, input, target, count+ 3)
                heat_map3 = np.stack(heat_map3, axis=0)
                heat_map3 = torch.from_numpy(heat_map3)

                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]

                for t, m, s in zip(img1, mean, std):
                    t.mul_(s).add_(m)

                for t, m, s in zip(img2, mean, std):
                    t.mul_(s).add_(m)

                for t, m, s in zip(img3, mean, std):
                    t.mul_(s).add_(m)


                logger.add_image('final_image1/origin', img1, epoch * len(val_loader)+count)
                logger.add_image('final_image2/origin', img2, epoch * len(val_loader)+count)
                logger.add_image('final_image3/origin', img3, epoch * len(val_loader)+count)
                logger.add_image('final_image1/final_heatmap',
                                 torchvision.utils.make_grid(heat_map1, nrow=heat_map1.shape[0], padding=10),
                                 epoch * len(val_loader) + i)
                logger.add_image('final_image2/final_heatmap',
                                 torchvision.utils.make_grid(heat_map2, nrow=heat_map2.shape[0], padding=10),
                                 epoch * len(val_loader) + i)
                logger.add_image('final_image3/final_heatmap',
                                 torchvision.utils.make_grid(heat_map3, nrow=heat_map3.shape[0], padding=10),
                                 epoch * len(val_loader) + i)

                img1 = img1.detach().cpu().numpy()
                img2 = img2.detach().cpu().numpy()
                img3 = img3.detach().cpu().numpy()
                heat_map1 = heat_map1.numpy()
                heat_map2 = heat_map2.numpy()
                heat_map3 = heat_map3.numpy()

                vis.image(img1, opts=dict(title="final_image1/origin"))
                vis.image(img2, opts=dict(title="final_image2/origin"))
                vis.image(img3, opts=dict(title="final_image3/origin"))
                vis.images(heat_map1, nrow=heat_map1.shape[0], opts=dict(title="final_image1/final_heatmap"))
                vis.images(heat_map2, nrow=heat_map2.shape[0], opts=dict(title="final_image2/final_heatmap"))
                vis.images(heat_map3, nrow=heat_map3.shape[0], opts=dict(title="final_image3/final_heatmap"))


        loss = criterion(imoutput, target)

        # measure metrics and record loss
        m1 = metric1(imoutput.data, target)
        m2 = metric2(imoutput.data, target)
        losses.update(loss.item(), input.size(0))
        avg_m1.update(m1[0], input.size(0))
        avg_m2.update(m2[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      i,
                      len(val_loader),
                      batch_time=batch_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

        #TODO: Visualize things as mentioned in handout
        #TODO: Visualize at appropriate intervals



    print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'.format(
        avg_m1=avg_m1, avg_m2=avg_m2))
    # if epoch%2==0:
    logger.add_scalar('val/metric1',
                      avg_m1.avg,
                      epoch * len(val_loader))
    logger.add_scalar('val/metric2',
                      avg_m2.avg,
                      epoch * len(val_loader))

    return avg_m1.avg, avg_m2.avg


# TODO: You can make changes to this function if you wish (not necessary)
def save_checkpoint(state, is_best, filename='./model/checkpoint3.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def metric1(output, target):
    # TODO: Ignore for now - proceed till instructed
    output = nn.Sigmoid()(output)
    target = target.cpu().numpy()
    output = output.cpu().numpy()
    nclasses = target.shape[1]
    AP = []
    for cid in range(nclasses):
        gt_cls = target[:, cid].astype('float32')
        pred_cls = output[:, cid].astype('float32')
        pred_cls -= 1e-5 * gt_cls
        ap = sklearn.metrics.average_precision_score(
            gt_cls, pred_cls)
        AP.append(ap)

    mAP = np.mean(AP)
    return [mAP]


def metric2(output, target):
    #TODO: Ignore for now - proceed till instructed
    target = target.cpu().numpy().astype('float32')
    output = output.cpu().numpy().astype('float32')

    nclasses = target.shape[1]
    AP = []
    for cid in range(nclasses):
        gt_cls = target[:, cid]
        pred_cls = output[:, cid]
        if all([gt_cls[i]==0 for i in range(gt_cls.shape[0])]):
            continue
        pred_cls -= 1e-5 * gt_cls
        ap = sklearn.metrics.roc_auc_score(
            gt_cls, pred_cls)
        AP.append(ap)

    auc = np.mean(AP)
    return [auc]


if __name__ == '__main__':
    main()
