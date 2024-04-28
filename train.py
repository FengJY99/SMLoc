import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import torch
import sys
import time
import os.path as osp
import numpy as np

from tensorboardX import SummaryWriter
from tools.options import Options
from network.atloc import AtLoc, AtLocPlus
from network.convAtloc import ConvAtLoc
from network.convAtlocV2 import ConvAtLocV2
from network.cooAttLoc import CooAttLoc

from network.cviPLoc import CViPLoc
from network.cviPLocV2 import CViPLocV2
from network.posenet import PoseNet
from network.caploc import CAPLoc

from torchvision import transforms, models
from tools.utils import AtLocCriterion, AtLocPlusCriterion, CALocCriterion, AverageMeter, Logger
from data.dataloaders import SevenScenes, RobotCar, MF, CambridgeLandmarks
from torch.utils.data import DataLoader
from torch.autograd import Variable

# Config
opt = Options().parse()
cuda = torch.cuda.is_available()
torch_seed = 0
numpy_seed = opt.seed
torch.manual_seed(torch_seed)
if cuda:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
np.random.seed(numpy_seed)

device = "cuda:" + ",".join(str(i) for i in opt.gpus) if cuda else "cpu"

logfile = osp.join(opt.runs_dir, 'log.txt')

stdout = Logger(logfile)
print('Logging to {:s}'.format(logfile))
sys.stdout = stdout

# Model
feature_extractor_resnet34 = models.resnet34(pretrained=True)
feature_extractor_resnet50 = models.resnet50(replace_stride_with_dilation=[False, True, True], pretrained=True)
feature_extractor_convnext = models.convnext_tiny(pretrained=True)

atloc = AtLoc(feature_extractor_resnet34, droprate=opt.train_dropout, pretrained=True, lstm=opt.lstm)
# convnext + non-local self attention
# convatloc = ConvAtLoc(feature_extractor_convnext, droprate=opt.train_dropout, pretrained=True, lstm=opt.lstm)
# convatlocV2 = ConvAtLocV2(feature_extractor_convnext, droprate=opt.train_dropout, pretrained=True, lstm=opt.lstm)
# convnext + coordinate attention
# cooattloc = CooAttLoc(feature_extractor_convnext.features, droprate=opt.train_dropout, pretrained=True)

posenet = PoseNet(feature_extractor_resnet34, droprate=opt.train_dropout, pretrained=True)

cvipLoc = CViPLoc(feature_extractor=feature_extractor_convnext.features, droprate=opt.train_dropout, pretrained=True)
cvipLocV2 = CViPLocV2(feature_extractor=feature_extractor_convnext.features, droprate=opt.train_dropout, pretrained=True)
capLoc = CAPLoc(feature_extractor=feature_extractor_convnext.features, droprate=opt.train_dropout, pretrained=True)


# Set the loss
if opt.model == 'AtLoc':
    model = atloc
    train_criterion = AtLocCriterion(saq=opt.beta, learn_beta=True)
    val_criterion = AtLocCriterion()
    param_list = [{'params': model.parameters()}]
elif opt.model == 'CAPLoc':
    model = capLoc
    train_criterion = AtLocCriterion(saq=opt.beta, learn_beta=True)
    val_criterion = AtLocCriterion()
    param_list = [{'params': model.parameters()}]
elif opt.model == 'CViPLocV2':
    model = cvipLocV2
    train_criterion = AtLocCriterion(saq=opt.beta, learn_beta=True)
    val_criterion = AtLocCriterion()
    param_list = [{'params': model.parameters()}]
elif opt.model == 'CViPLoc':
    model = cvipLoc
    train_criterion = AtLocCriterion(saq=opt.beta, learn_beta=True)
    val_criterion = AtLocCriterion()
    param_list = [{'params': model.parameters()}]
elif opt.model == 'PoseNet':
    model = posenet
    train_criterion = AtLocCriterion(saq=opt.beta, learn_beta=True)
    val_criterion = AtLocCriterion()
    param_list = [{'params': model.parameters()}]

else:
    raise NotImplementedError


# Optimizer
param_list = [{'params': model.parameters()}]
if hasattr(train_criterion, 'sax') and hasattr(train_criterion, 'saq'):
    print('learn_beta')
    param_list.append({'params': [train_criterion.sax, train_criterion.saq]})
if opt.gamma is not None and hasattr(train_criterion, 'srx') and hasattr(train_criterion, 'srq'):
    print('learn_gamma')
    param_list.append({'params': [train_criterion.srx, train_criterion.srq]})
# optimizer = torch.optim.Adam(param_list, lr=opt.lr, weight_decay=opt.weight_decay)
optimizer = torch.optim.AdamW(param_list, lr=opt.lr, weight_decay=opt.weight_decay)
# optimizer = torch.optim.SGD(param_list, lr=opt.lr, weight_decay=opt.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_scheduler_step_size,
                                            gamma=opt.lr_scheduler_gamma)

stats_file = osp.join(opt.data_dir, opt.dataset, opt.scene, 'stats.txt')
stats = np.loadtxt(stats_file)

# data enhancement
tforms = [transforms.Resize(opt.cropsize)]
tforms.append(transforms.RandomCrop(opt.cropsize))
if opt.color_jitter > 0:
    assert opt.color_jitter <= 1.0
    print('Using ColorJitter data augmentation')
    tforms.append(
        transforms.ColorJitter(brightness=opt.color_jitter, contrast=opt.color_jitter, saturation=opt.color_jitter,
                               hue=0.5))
else:
    print('Not Using ColorJitter')
tforms.append(transforms.ToTensor())
tforms.append(transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1])))
data_transform = transforms.Compose(tforms)
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

# Load the dataset
kwargs = dict(scene=opt.scene, data_path=opt.data_dir, transform=data_transform, target_transform=target_transform,
              seed=opt.seed)
# or opt.model == 'CALocTiny'or opt.model == 'ConvAtLoc' or opt.model == 'CooAttLoc' or opt.model == 'CALoc' or opt.model == 'CALocPlus'
if opt.model == 'AtLoc'or opt.model == 'PoseNet' or opt.model == 'CAPLoc' or opt.model == 'CViPLoc' or opt.model == 'CViPLocV2':
    if opt.dataset == '7Scenes':
        train_set = SevenScenes(train=True, **kwargs)
        val_set = SevenScenes(train=False, **kwargs)
    elif opt.dataset == 'RobotCar':
        train_set = RobotCar(train=True, **kwargs)
        val_set = RobotCar(train=False, **kwargs)
    elif opt.dataset == 'Cambridge':
        train_set = CambridgeLandmarks(train=True, **kwargs)
        val_set = CambridgeLandmarks(train=False, **kwargs)
    else:
        raise NotImplementedError
elif opt.model == 'AtLocPlus':
    kwargs = dict(kwargs, dataset=opt.dataset, skip=opt.skip, steps=opt.steps, variable_skip=opt.variable_skip)
    train_set = MF(train=True, real=opt.real, **kwargs)
    val_set = MF(train=False, real=opt.real, **kwargs)
else:
    raise NotImplementedError
kwargs = {'num_workers': opt.nThreads, 'pin_memory': True} if cuda else {}
train_loader = DataLoader(train_set, batch_size=opt.batchsize, shuffle=True, **kwargs)
val_loader = DataLoader(val_set, batch_size=opt.batchsize, shuffle=False, **kwargs)

# model.to(device)
# train_criterion.to(device)
# val_criterion.to(device)

model = torch.nn.DataParallel(model)
model.to(device)
train_criterion.to(device)
val_criterion.to(device)


total_steps = opt.steps
writer = SummaryWriter(log_dir=opt.runs_dir)
experiment_name = opt.exp_name

# Load the checkpoint if needed
# if opt.checkpoint_path:
#     model.load_state_dict(torch.load(opt.checkpoint_path))
#     print('Load the checkpoint')
#
# # Freeze parts of the model if indicated
# freeze = opt.freeze
# freeze_exclude_phrase = opt.freeze_exclude_phrase
# if isinstance(freeze_exclude_phrase, str):
#     freeze_exclude_phrase = [freeze_exclude_phrase]


for epoch in range(opt.epochs):
    if epoch % opt.val_freq == 0 or epoch == (opt.epochs - 1):
        val_batch_time = AverageMeter()
        val_loss = AverageMeter()
        model.eval()
        end = time.time()
        val_data_time = AverageMeter()

        for batch_idx, (val_data, val_target) in enumerate(val_loader):
            val_data_time.update(time.time() - end)
            val_data_var = Variable(val_data, requires_grad=False)
            val_target_var = Variable(val_target, requires_grad=False)
            val_data_var = val_data_var.to(device)
            val_target_var = val_target_var.to(device)

            with torch.set_grad_enabled(False):
                val_output = model(val_data_var)
                val_loss_tmp = val_criterion(val_output, val_target_var)
                val_loss_tmp = val_loss_tmp.item()

            # Collect for plotting
            # loss_vals.append(val_loss_tmp)

            val_loss.update(val_loss_tmp)
            val_batch_time.update(time.time() - end)

            writer.add_scalar('val_err', val_loss_tmp, total_steps)
            if batch_idx % opt.print_freq == 0:
                print(
                    'Val {:s}: Epoch {:d}\tBatch {:d}/{:d}\tData time {:.4f} ({:.4f})\tBatch time {:.4f} ({:.4f})\tLoss {:f}' \
                    .format(experiment_name, epoch, batch_idx, len(val_loader) - 1, val_data_time.val,
                            val_data_time.avg, val_batch_time.val, val_batch_time.avg, val_loss_tmp))
            end = time.time()

        print('Val {:s}: Epoch {:d}, val_loss {:f}'.format(experiment_name, epoch, val_loss.avg))

        if epoch % opt.save_freq == 0:
            filename = osp.join(opt.models_dir, 'epoch_{:03d}.pth.tar'.format(epoch))
            checkpoint_dict = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                               'optim_state_dict': optimizer.state_dict(),
                               'criterion_state_dict': train_criterion.state_dict()}
            torch.save(checkpoint_dict, filename)
            print('Epoch {:d} checkpoint saved for {:s}'.format(epoch, experiment_name))

    model.train()
    train_data_time = AverageMeter()
    train_batch_time = AverageMeter()
    end = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        train_data_time.update(time.time() - end)

        data_var = Variable(data, requires_grad=True)
        target_var = Variable(target, requires_grad=False)
        data_var = data_var.to(device)
        target_var = target_var.to(device)

        with torch.set_grad_enabled(True):
            output = model(data_var)
            loss_tmp = train_criterion(output, target_var)

        # Collect for plotting
        # loss_trains = loss_trains.append(loss_tmp)

        loss_tmp.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_batch_time.update(time.time() - end)
        writer.add_scalar('train_err', loss_tmp.item(), total_steps)
        if batch_idx % opt.print_freq == 0:
            print(
                'Train {:s}: Epoch {:d}\tBatch {:d}/{:d}\tData time {:.4f} ({:.4f})\tBatch time {:.4f} ({:.4f})\tLoss {:f}' \
                .format(experiment_name, epoch, batch_idx, len(train_loader) - 1, train_data_time.val,
                        train_data_time.avg, train_batch_time.val, train_batch_time.avg, loss_tmp.item()))
        end = time.time()

    # Scheduler update
    scheduler.step()
        # Plot the loss function
        # loss_fig_path = os.path.join(opt.runs_dir, "loss_fig.png")

writer.close()
