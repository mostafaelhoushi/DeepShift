from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import optim
from torchvision import datasets, transforms
import csv
import distutils
import os
from contextlib import redirect_stdout
import time
from torchsummary import summary
import mnist
import copy

import shift
from cuda_kernel import convert_to_unoptimized
from convert_to_shift import convert_to_shift, round_shift_weights

class LinearMNIST(nn.Module):
    def __init__(self):
        super(LinearMNIST, self).__init__()
        self.fc1 = nn.Linear(1*28*28, 512)  
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 1 * 28 * 28) 
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class ConvMNIST(nn.Module):
    def __init__(self):
        super(ConvMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, loss_fn, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    return loss.item()

def test(args, model, device, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss, correct

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--type', default='linear',
                        choices=['linear', 'conv'],
                        help='model architecture type: ' +
                        ' | '.join(['linear', 'conv']) +
                        ' (default: linear)')
    parser.add_argument('--model', default='', type=str, metavar='MODEL_PATH',
                        help='path to model file to load both its architecture and weights (default: none)')
    parser.add_argument('--weights', default='', type=str, metavar='WEIGHTS_PATH',
                        help='path to file to load its weights (default: none)')
    parser.add_argument('--shift-depth', type=int, default=0,
                        help='how many layers to convert to shift')
    parser.add_argument('-st', '--shift-type', default='Q', choices=['Q', 'PS'],
                        help='type of DeepShift method for training and representing weights (default: Q)')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('-opt', '--optimizer', metavar='OPT', default="SGD", 
                        help='optimizer algorithm')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                        help='SGD momentum (default: 0.0)')
    parser.add_argument('--resume', default='', type=str, metavar='CHECKPOINT_PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='only evaluate model on validation set')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--pretrained', dest='pretrained', default=False, type=lambda x:bool(distutils.util.strtobool(x)), 
                        help='use pre-trained model of full conv or fc model')
    
    parser.add_argument('--save-model', default=True, type=lambda x:bool(distutils.util.strtobool(x)), 
                        help='For Saving the current Model (default: True)')
    parser.add_argument('--print-weights', default=True, type=lambda x:bool(distutils.util.strtobool(x)), 
                        help='For printing the weights of Model (default: True)')
    parser.add_argument('--desc', type=str, default=None,
                        help='description to append to model directory name')
    parser.add_argument('--use-kernel', type=lambda x:bool(distutils.util.strtobool(x)), default=False,
                        help='whether using custom shift kernel')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,)) # transforms.Normalize((0,), (255,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,)) # transforms.Normalize((0,), (255,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    if args.model:
        if args.type or args.pretrained:
            print("WARNING: Ignoring arguments \"type\" and \"pretrained\" when creating model...")
        model = None
        saved_checkpoint = torch.load(args.model)
        if isinstance(saved_checkpoint, nn.Module):
            model = saved_checkpoint
        elif "model" in saved_checkpoint:
            model = saved_checkpoint["model"]
        else:
            raise Exception("Unable to load model from " + args.model)
    else:
        if args.type == 'linear':
            model = LinearMNIST().to(device)
        elif args.type == 'conv':
            model = ConvMNIST().to(device)

        if args.pretrained:
            model.load_state_dict(torch.load("./models/mnist/simple_" + args.type + "/shift_0/weights.pt"))
            model = model.to(device)

    if args.weights:
        saved_weights = torch.load(args.weights)
        if isinstance(saved_weights, nn.Module):
            state_dict = saved_weights.state_dict()
        elif "state_dict" in saved_weights:
            state_dict = saved_weights["state_dict"]
        else:
            state_dict = saved_weights
            
        model.load_state_dict(state_dict)

    if args.shift_depth > 0:
        model, _ = convert_to_shift(model, args.shift_depth, args.shift_type, convert_all_linear=(args.type != 'linear'), convert_weights=True, use_kernel = args.use_kernel, use_cuda = use_cuda)
        model = model.to(device)
    elif args.use_kernel and args.shift_depth == 0:
        model = convert_to_unoptimized(model)
        model = model.to(device)
    
    loss_fn = F.cross_entropy # F.nll_loss
    # define optimizer
    optimizer = None 
    if(args.optimizer.lower() == "sgd"):
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum)
    elif(args.optimizer.lower() == "adadelta"):
        optimizer = torch.optim.Adadelta(model.parameters(), args.lr)
    elif(args.optimizer.lower() == "adagrad"):
        optimizer = torch.optim.Adagrad(model.parameters(), args.lr)
    elif(args.optimizer.lower() == "adam"):
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
    elif(args.optimizer.lower() == "rmsprop"):
        optimizer = torch.optim.RMSprop(model.parameters(), args.lr)
    elif(args.optimizer.lower() == "radam"):
        optimizer = optim.RAdam(model.parameters(), args.lr)
    elif(args.optimizer.lower() == "ranger"):
        optimizer = optim.Ranger(model.parameters(), args.lr)
    else:
        raise ValueError("Optimizer type: ", args.optimizer, " is not supported or known")

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("=> loaded checkpoint '{}'"
                  .format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.desc is not None and len(args.desc) > 0:
        model_name = 'simple_%s/%s_shift_%s' % (args.type, args.desc, args.shift_depth)
    else:
        model_name = 'simple_%s/shift_%s' % (args.type, args.shift_depth)

    # if evaluating round weights to ensure that the results are due to powers of 2 weights
    if not args.use_kernel:
        if (args.evaluate):
            model = round_shift_weights(model)

    if not args.use_kernel:
        model_tmp_copy = copy.deepcopy(model) # we noticed calling summary() on original model degrades it's accuracy. So we will call summary() on a copy of the model
        try:
            summary(model_tmp_copy, input_size=(1, 28, 28), device=("cuda" if use_cuda else "cpu"))
            print("WARNING: The summary function reports duplicate parameters for multi-GPU case")
        except:
            print("WARNING: Unable to obtain summary of model")

        model_dir = os.path.join(os.path.join(os.path.join(os.getcwd(), "models"), "mnist"), model_name)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        if (args.save_model):
            with open(os.path.join(model_dir, 'command_args.txt'), 'w') as command_args_file:
                for arg, value in sorted(vars(args).items()):
                    command_args_file.write(arg + ": " + str(value) + "\n")

            with open(os.path.join(model_dir, 'model_summary.txt'), 'w') as summary_file:
                with redirect_stdout(summary_file):
                    try:
                        summary(model_tmp_copy, input_size=(1, 28, 28), device=("cuda" if use_cuda else "cpu"))
                        print("WARNING: The summary function reports duplicate parameters for multi-GPU case")
                    except:
                        print("WARNING: Unable to obtain summary of model")

        del model_tmp_copy

    start_time = time.time()
    if args.evaluate:
        test_loss, correct = test(args, model, device, test_loader, loss_fn)
        test_log = [(test_loss, correct/1e4)]

        # with open(os.path.join(model_dir, "test_log.csv"), "w") as test_log_file:
        #     test_log_csv = csv.writer(test_log_file)
        #     test_log_csv.writerow(['test_loss', 'correct'])
        #     test_log_csv.writerows(test_log)
    else:
        train_log = []
        for epoch in range(1, args.epochs + 1):
            train_loss = train(args, model, device, train_loader, loss_fn, optimizer, epoch)
            test_loss, correct = test(args, model, device, test_loader, loss_fn)

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

            train_log.append((epoch, train_loss, test_loss, correct/1e4))

        with open(os.path.join(model_dir, "train_log.csv"), "w") as train_log_file:
            train_log_csv = csv.writer(train_log_file)
            train_log_csv.writerow(['epoch', 'train_loss', 'test_loss', 'test_accuracy'])
            train_log_csv.writerows(train_log)

    # if (args.save_model):
    #     torch.save(model, os.path.join(model_dir, "model.pt"))
    #     torch.save(model.state_dict(), os.path.join(model_dir, "weights.pt"))
    #     torch.save(optimizer.state_dict(), os.path.join(model_dir, "optimizer.pt"))

    end_time = time.time()
    print("Total Time:", end_time - start_time )

    # if (args.print_weights):
    #     with open(os.path.join(model_dir, 'weights_log.txt'), 'w') as weights_log_file:
    #         with redirect_stdout(weights_log_file):
    #             # Log model's state_dict
    #             print("Model's state_dict:")
    #             # TODO: Use checkpoint above
    #             for param_tensor in model.state_dict():
    #                 print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    #                 print(model.state_dict()[param_tensor])
    #                 print("")
        
if __name__ == '__main__':
    main()
