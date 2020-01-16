import argparse
import os
import random
import shutil
import time
import warnings
import sys
import csv
import distutils
from contextlib import redirect_stdout
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torchsummary import summary
import optim
import copy

from convert_to_shift import convert_to_shift, count_layer_type, round_shift_weights
import customized_models

default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--model', default='', type=str, metavar='MODEL_PATH',
                    help='path to model file to load both its architecture and weights (default: none)')
parser.add_argument('--weights', default='', type=str, metavar='WEIGHTS_PATH',
                    help='path to file to load its weights (default: none)')
parser.add_argument('-s', '--shift-depth', type=int, default=0,
                    help='how many layers to convert to shift')
parser.add_argument('-st', '--shift-type', default='Q', choices=['Q', 'PS'],
                    help='type of DeepShift method for training and representing weights (default: Q)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-opt', '--optimizer', metavar='OPT', default="SGD", 
                    help='optimizer algorithm')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-bm', '--batch-multiplier', default=1, type=int,
                    help='how many batches to repeat before updating parameter. '
                         'effective batch size is batch-size * batch-multuplier')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-schedule', dest='lr_schedule', default=False, type=lambda x:bool(distutils.util.strtobool(x)), 
                    help='using learning rate schedule')
parser.add_argument('--lr-sign', default=None, type=float,
                    help='separate initial learning rate for sign params')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--opt-ckpt', default='', type=str, metavar='OPT_PATH',
                    help='path to checkpoint file to load optimizer state (default: none)')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='CHECKPOINT_PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='only evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default=True, type=lambda x:bool(distutils.util.strtobool(x)), 
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--save-model', default=True, type=lambda x:bool(distutils.util.strtobool(x)), 
                    help='For Saving the current Model (default: True)')
parser.add_argument('--print-weights', default=True, type=lambda x:bool(distutils.util.strtobool(x)), 
                    help='For printing the weights of Model (default: True)')
parser.add_argument('--desc', type=str, default=None,
                    help='description to append to model directory name')
parser.add_argument('--use-kernel', type=lambda x:bool(distutils.util.strtobool(x)), default=False,
                    help='whether using custom shift kernel')

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.model:
        if args.arch or args.pretrained:
            print("WARNING: Ignoring arguments \"arch\" and \"pretrained\" when creating model...")
        model = None
        saved_checkpoint = torch.load(args.model)
        if isinstance(saved_checkpoint, nn.Module):
            model = saved_checkpoint
        elif "model" in saved_checkpoint:   
            model = saved_checkpoint["model"]
        else:
            raise Exception("Unable to load model from " + args.model)   

        if (args.gpu is not None):
            model.cuda(args.gpu) 
    elif args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.weights:
        saved_weights = torch.load(args.weights)
        if isinstance(saved_weights, nn.Module):
            state_dict = saved_weights.state_dict()
        elif "state_dict" in saved_weights:
            state_dict = saved_weights["state_dict"]
        else:
            state_dict = saved_weights
            
        try:
            model.load_state_dict(state_dict)
        except:
            # create new OrderedDict that does not contain module.
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove module.
                new_state_dict[name] = v
                
            model.load_state_dict(new_state_dict)

    if args.shift_depth > 0:
        model, _ = convert_to_shift(model, args.shift_depth, args.shift_type, convert_weights = args.pretrained or args.weights, freeze_sign = (args.lr_sign == 0), use_kernel = args.use_kernel)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # create optimizer
    model_other_params = []
    model_sign_params = []
    model_shift_params = []

    for name, param in model.named_parameters():
        if(name.endswith(".sign")):
            model_sign_params.append(param)
        elif(name.endswith(".shift")):
            model_shift_params.append(param)
        else:
            model_other_params.append(param)

    params_dict = [
        {"params": model_other_params},
        {"params": model_sign_params, 'lr': args.lr_sign if args.lr_sign is not None else args.lr, 'weight_decay': 0},
        {"params": model_shift_params, 'lr': args.lr, 'weight_decay': 0}
        ]

    # define optimizer
    optimizer = None 
    if(args.optimizer.lower() == "sgd"):
        optimizer = torch.optim.SGD(params_dict, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif(args.optimizer.lower() == "adadelta"):
        optimizer = torch.optim.Adadelta(params_dict, args.lr, weight_decay=args.weight_decay)
    elif(args.optimizer.lower() == "adagrad"):
        optimizer = torch.optim.Adagrad(params_dict, args.lr, weight_decay=args.weight_decay)
    elif(args.optimizer.lower() == "adam"):
        optimizer = torch.optim.Adam(params_dict, args.lr, weight_decay=args.weight_decay)
    elif(args.optimizer.lower() == "rmsprop"):
        optimizer = torch.optim.RMSprop(params_dict, args.lr, weight_decay=args.weight_decay)
    elif(args.optimizer.lower() == "radam"):
        optimizer = optim.RAdam(params_dict, args.lr, weight_decay=args.weight_decay)
    elif(args.optimizer.lower() == "ranger"):
        optimizer = optim.Ranger(params_dict, args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError("Optimizer type: ", args.optimizer, " is not supported or known")

    lr_scheduler = None
    if args.opt_ckpt:
        print("WARNING: Ignoring arguments \"lr\", \"momentum\", \"weight_decay\", and \"lr_schedule\"")

        opt_ckpt = torch.load(args.opt_ckpt)
        if 'optimizer' in opt_ckpt:
            opt_ckpt = opt_ckpt['optimizer']
        optimizer.load_state_dict(opt_ckpt)

        if 'lr_scheduler' in opt_ckpt:
            lr_scheduler = opt_ckpt['lr_scheduler']

    if (args.lr_schedule and lr_scheduler is not None):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[100, 150], last_epoch=args.start_epoch - 1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except:
                # create new OrderedDict that does not contain module.
                state_dict = checkpoint['state_dict']
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
                        if (k.startswith("features")):
                            name = k[0:9] + k[9+7:] # remove "module" after features
                        else:
                            name = k
                    else:
                        name = k[7:] # remove "module" at beginning of name
                    new_state_dict[name] = v
                
                # load params
                model.load_state_dict(new_state_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'lr_scheduler' in checkpoint and checkpoint['lr_scheduler'] is not None:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # if evaluating round weights to ensure that the results are due to powers of 2 weights
    if (args.evaluate):
        model = round_shift_weights(model)

    cudnn.benchmark = True

    model_tmp_copy = copy.deepcopy(model) # we noticed calling summary() on original model degrades it's accuracy. So we will call summary() on a copy of the model
    try:
        summary(model_tmp_copy, input_size=(3, 224, 224))
        print("WARNING: The summary function reports duplicate parameters for multi-GPU case")
    except:
        print("WARNING: Unable to obtain summary of model")

    # name model sub-directory "shift_all" if all layers are converted to shift layers
    conv2d_layers_count = count_layer_type(model, nn.Conv2d)
    linear_layers_count = count_layer_type(model, nn.Linear)
    if (args.shift_type == 'Q'):
        shift_label = "shift_q"
    else:
        shift_label = "shift"

    if (conv2d_layers_count==0 and linear_layers_count==0):
        shift_label += "_all"
    else:
        shift_label += "_%s" % (args.shift_depth)

    if args.desc is not None and len(args.desc) > 0:
        model_name = '%s/%s_%s' % (args.arch, args.desc, shift_label)
    else:
        model_name = '%s/%s' % (args.arch, shift_label)

    if (args.save_model):
        model_dir = os.path.join(os.path.join(os.path.join(os.getcwd(), "models"), "imagenet"), model_name)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        with open(os.path.join(model_dir, 'command_args.txt'), 'w') as command_args_file:
            for arg, value in sorted(vars(args).items()):
                command_args_file.write(arg + ": " + str(value) + "\n")

        with open(os.path.join(model_dir, 'model_summary.txt'), 'w') as summary_file:
            with redirect_stdout(summary_file):
                try:
                    # TODO: make this summary function deal with parameters that are not named "weight" and "bias"
                    summary(model_tmp_copy, input_size=(3, 224, 224))
                    print("WARNING: The summary function reports duplicate parameters for multi-GPU case")
                except:
                    print("WARNING: Unable to obtain summary of model")

    del model_tmp_copy # to save memory

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    start_time = time.time()

    if args.evaluate:
        start_log_time = time.time()
        val_log = validate(val_loader, model, criterion, args)
        val_log = [val_log]

        with open(os.path.join(model_dir, "test_log.csv"), "w") as test_log_file:
            test_log_csv = csv.writer(test_log_file)
            test_log_csv.writerow(['test_loss', 'test_top1_acc', 'test_top5_acc', 'test_time', 'cumulative_time'])
            test_log_csv.writerows(val_log + [(time.time() - start_log_time,)])
    else:
        train_log = []

        with open(os.path.join(model_dir, "train_log.csv"), "w") as train_log_file:
            train_log_csv = csv.writer(train_log_file)
            train_log_csv.writerow(['epoch', 'train_loss', 'train_top1_acc', 'train_top5_acc', 'train_time', 'test_loss', 'test_top1_acc', 'test_top5_acc', 'test_time', 'cumulative_time'])

        start_log_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            adjust_learning_rate(optimizer, epoch, args)

            # train for one epoch
            print("current lr ", [param['lr'] for param in  optimizer.param_groups])
            train_epoch_log = train(train_loader, model, criterion, optimizer, epoch, args)
            if (args.lr_schedule):
                lr_scheduler.step()

            # evaluate on validation set
            val_epoch_log = validate(val_loader, model, criterion, args)
            acc1 = val_epoch_log[2]

            # append to log
            with open(os.path.join(model_dir, "train_log.csv"), "a") as train_log_file:
                train_log_csv = csv.writer(train_log_file)
                train_log_csv.writerow(((epoch,) + train_epoch_log + val_epoch_log + (time.time() - start_log_time,))) 

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if (args.print_weights):
                with open(os.path.join(model_dir, 'weights_log_' + str(epoch) + '.txt'), 'w') as weights_log_file:
                    with redirect_stdout(weights_log_file):
                        # Log model's state_dict
                        print("Model's state_dict:")
                        # TODO: Use checkpoint above
                        for param_tensor in model.state_dict():
                            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
                            print(model.state_dict()[param_tensor])
                            print("")

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                if is_best:
                    try:
                        if (args.save_model):
                            torch.save(model.state_dict(), os.path.join(model_dir, "weights.pth"))
                            torch.save(optimizer.state_dict(), os.path.join(model_dir, "optimizer.pth"))
                            torch.save(model, os.path.join(model_dir, "model.pth"))
                    except: 
                        print("WARNING: Unable to save model.pth")
                
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                    'lr_scheduler' : lr_scheduler,
                }, is_best, model_dir)

    end_time = time.time()
    print("Total Time:", end_time - start_time )

    if (args.print_weights):
        with open(os.path.join(model_dir, 'weights_log.txt'), 'w') as weights_log_file:
            with redirect_stdout(weights_log_file):
                # Log model's state_dict
                print("Model's state_dict:")
                # TODO: Use checkpoint above
                for param_tensor in model.state_dict():
                    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
                    print(model.state_dict()[param_tensor])
                    print("")


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    sub_batch_count = args.batch_multiplier
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)
        if(args.weight_decay > 0):
            loss += shift_l2_norm(optimizer, args.weight_decay)
        loss /= args.batch_multiplier
        loss.backward()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        if sub_batch_count == 0:
            # compute gradient and do SGD step
            optimizer.step()
            optimizer.zero_grad()
            sub_batch_count = args.batch_multiplier

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)

        sub_batch_count -= 1
    
    return (losses.avg, top1.avg.cpu().numpy(), top5.avg.cpu().numpy(), batch_time.avg)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)


        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return (losses.avg, top1.avg.cpu().numpy(), top5.avg.cpu().numpy(), batch_time.avg)

def shift_l2_norm(opt, weight_decay):
    shift_params = opt.param_groups[2]['params']
    l2_norm = 0
    for shift in shift_params:
        l2_norm += torch.sum((2**shift)**2)
    return weight_decay * 0.5 * l2_norm

def save_checkpoint(state, is_best, dir_path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(dir_path, filename))
    if is_best:
        shutil.copyfile(os.path.join(dir_path, filename), os.path.join(dir_path, 'model_best.pth.tar'))

    if (state['epoch']-1)%10 == 0:
        shutil.copyfile(os.path.join(dir_path, filename), os.path.join(dir_path, 'checkpoint_' + str(state['epoch']-1) + '.pth.tar'))    


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
