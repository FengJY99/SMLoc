import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import torch
import os.path as osp
import numpy as np
import matplotlib
import sys

DISPLAY = 'DISPLAY' in os.environ
if not DISPLAY:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tools.options import Options
from network.atloc import AtLoc, AtLocPlus

from network.cviPLoc import CViPLoc
from network.cviPLocV2 import CViPLocV2
from network.posenet import PoseNet
from network.caploc import CAPLoc

from torchvision import transforms, models
from tools.utils import quaternion_angular_error, qexp, load_state_dict
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

# Model
feature_extractor_resnet34 = models.resnet34(pretrained=False)
feature_extractor_convnext = models.convnext_tiny(pretrained=False)
feature_extractor_resnet50 = models.resnet50(replace_stride_with_dilation=[False, True, True], pretrained=True)

posenet = PoseNet(feature_extractor_resnet34, droprate=opt.test_dropout, pretrained=False)
atloc = AtLoc(feature_extractor_resnet34, droprate=opt.test_dropout, pretrained=False, lstm=opt.lstm)

cvipLoc = CViPLoc(feature_extractor=feature_extractor_convnext.features, droprate=opt.test_dropout, pretrained=False)
cvipLocV2 = CViPLocV2(feature_extractor=feature_extractor_convnext.features, droprate=opt.test_dropout, pretrained=False)
caploc = CAPLoc(feature_extractor=feature_extractor_convnext.features, droprate=opt.test_dropout, pretrained=False)

if opt.model == 'AtLoc':
    model = atloc
elif opt.model == 'CAPLoc':
    model = caploc
elif opt.model == 'PoseNet':
   model = posenet
elif opt.model == 'CViPLocV2':
    model = cvipLocV2
elif opt.model == 'CViPLoc':
   model = cvipLoc
# elif opt.model == 'CALocPlus':
#    model = caLocPlus
# elif opt.model == 'CALoc':
#     model = caLoc
# elif opt.model == 'ConvAtLocV2':
#     model = convatlocV2
# elif opt.model == 'CooAttLoc':
#     model = cooattloc
elif opt.model == 'AtLocPlus':
    model = AtLocPlus(atlocplus=atloc)
else:
    raise NotImplementedError
model.eval()

# loss functions
t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
q_criterion = quaternion_angular_error

stats_file = osp.join(opt.data_dir, opt.dataset, opt.scene, 'stats.txt')
stats = np.loadtxt(stats_file)
# transformer
data_transform = transforms.Compose([
    transforms.Resize(opt.cropsize),
    transforms.CenterCrop(opt.cropsize),
    transforms.ToTensor(),
    transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1]))])
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

# read mean and stdev for un-normalizing predictions
pose_stats_file = osp.join(opt.data_dir, opt.dataset, opt.scene, 'pose_stats.txt')
pose_m, pose_s = np.loadtxt(pose_stats_file)  # mean and stdev

# Load the dataset
kwargs = dict(scene=opt.scene, data_path=opt.data_dir, train=False, transform=data_transform, target_transform=target_transform, seed=opt.seed)
if opt.model == 'AtLoc' or opt.model == 'CAPLoc' or opt.model == 'PoseNet' or opt.model == 'CViPLocV2' or opt.model == 'CViPLoc':
    if opt.dataset == '7Scenes':
        data_set = SevenScenes(**kwargs)
    elif opt.dataset == 'RobotCar':
        data_set = RobotCar(**kwargs)
    elif opt.dataset == 'Cambridge':
        data_set = CambridgeLandmarks(**kwargs)
    else:
        raise NotImplementedError
elif opt.model == 'AtLocPlus':
    kwargs = dict(kwargs, dataset=opt.dataset, skip=opt.skip, steps=opt.steps, variable_skip=opt.variable_skip)
    data_set = MF(real=opt.real, **kwargs)
else:
    raise NotImplementedError
L = len(data_set)
kwargs = {'num_workers': opt.nThreads, 'pin_memory': True} if cuda else {}
loader = DataLoader(data_set, batch_size=1, shuffle=False, **kwargs)

pred_poses = np.zeros((L, 7))  # store all predicted poses
targ_poses = np.zeros((L, 7))  # store all target poses

# load weights
model = torch.nn.DataParallel(model)
model.to(device)
weights_filename = osp.expanduser(opt.weights)
if osp.isfile(weights_filename):
    checkpoint = torch.load(weights_filename, map_location=device)
    load_state_dict(model, checkpoint['model_state_dict'])
    print('Loaded weights from {:s}'.format(weights_filename))
else:
    print('Could not load weights from {:s}'.format(weights_filename))
    sys.exit(-1)

# inference loop
for idx, (data, target) in enumerate(loader):
    if idx % 200 == 0:
        print('Image {:d} / {:d}'.format(idx, len(loader)))

    # output : 1 x 6
    data_var = Variable(data, requires_grad=False)
    data_var = data_var.to(device)

    with torch.set_grad_enabled(False):
        output = model(data_var)
    s = output.size()
    output = output.cpu().data.numpy().reshape((-1, s[-1]))
    target = target.numpy().reshape((-1, s[-1]))

    # normalize the predicted quaternions
    q = [qexp(p[3:]) for p in output]
    output = np.hstack((output[:, :3], np.asarray(q)))
    q = [qexp(p[3:]) for p in target]
    target = np.hstack((target[:, :3], np.asarray(q)))

    # un-normalize the predicted and target translations
    output[:, :3] = (output[:, :3] * pose_s) + pose_m
    target[:, :3] = (target[:, :3] * pose_s) + pose_m

    # take the middle prediction
    pred_poses[idx, :] = output[len(output) // 2]
    targ_poses[idx, :] = target[len(target) // 2]

# calculate losses
t_loss = np.asarray([t_criterion(p, t) for p, t in zip(pred_poses[:, :3], targ_poses[:, :3])])
q_loss = np.asarray([q_criterion(p, t) for p, t in zip(pred_poses[:, 3:], targ_poses[:, 3:])])
errors = np.zeros((L, 2))
print('Error in translation: median {:3.2f} m,  mean {:3.2f} m \nError in rotation: median {:3.2f} degrees, mean {:3.2f} degree'\
      .format(np.median(t_loss), np.mean(t_loss), np.median(q_loss), np.mean(q_loss)))

# fig = plt.figure()
# real_pose = (pred_poses[:, :3] - pose_m) / pose_s
# gt_pose = (targ_poses[:, :3] - pose_m) / pose_s
# plt.plot(gt_pose[:, 1], gt_pose[:, 0], color='black')
# plt.plot(real_pose[:, 1], real_pose[:, 0], color='red')
# plt.xlabel('x [m]')
# plt.ylabel('y [m]')
# plt.plot(gt_pose[0, 1], gt_pose[0, 0], 'y*', markersize=15)
# plt.show(block=True)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
# plot on the figure object
ss = max(1, int(len(data_set) / 1000))  # 100 for stairs
# scatter the points and draw connecting line
x = np.vstack((pred_poses[::ss, 0].T, targ_poses[::ss, 0].T))
y = np.vstack((pred_poses[::ss, 1].T, targ_poses[::ss, 1].T))
z = np.vstack((pred_poses[::ss, 2].T, targ_poses[::ss, 2].T))
for xx, yy, zz in zip(x.T, y.T, z.T):
    ax.plot(xx, yy, zs=zz, c='gray', alpha=0.6)
ax.scatter(x[0, :], y[0, :], zs=z[0, :], c='r', depthshade=0, alpha=0.8)
ax.scatter(x[1, :], y[1, :], zs=z[1, :], c='g', depthshade=0, alpha=0.8)
ax.view_init(azim=119, elev=13)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
plt.show()
image_filename = osp.join(osp.expanduser(opt.results_dir), '{:s}.png'.format(opt.exp_name))
fig.savefig(image_filename)
