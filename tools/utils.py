import os
import torch
import transforms3d.quaternions as txq
import transforms3d.euler as txe
import numpy as np
import sys
import matplotlib.pyplot as plt
from torch.nn import Module
from torch.autograd import Variable
from torch.nn.functional import pad
from torchvision.datasets.folder import default_loader
from collections import OrderedDict
from typing import Dict, List
from torch import nn, Tensor
import torch.nn.functional as F


##############################
#       Model Loss           #
##############################
class CALocCriterion(nn.Module):
    def __init__(self, sax=0.0, saq=-3.0, learn_beta=False):
        super(CALocCriterion, self).__init__()
        self.sax = torch.nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = torch.nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)

    def forward(self, pred, targ):
        # Position loss
        l_x = torch.norm(targ[:, 0:3] - pred[:, 0:3], dim=1, p=2).mean()
        # Orientation loss (normalized to unit norm)
        l_q = torch.norm(F.normalize(targ[:, 3:], p=2, dim=1) - F.normalize(pred[:, 3:], p=2, dim=1), dim=1, p=2).mean()

        loss = l_x * torch.exp(-self.sax) + self.sax + l_q * torch.exp(-self.saq) + self.saq
        return loss


class AtLocCriterion(nn.Module):
    def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0, saq=0.0, learn_beta=False):
        super(AtLocCriterion, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)

    def forward(self, pred, targ):
        loss = torch.exp(-self.sax) * self.t_loss_fn(pred[:, :3], targ[:, :3]) + self.sax + \
               torch.exp(-self.saq) * self.q_loss_fn(pred[:, 3:], targ[:, 3:]) + self.saq
        return loss


class AtLocPlusCriterion(nn.Module):
    def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0, saq=0.0, srx=0.0, srq=0.0,
                 learn_beta=False, learn_gamma=False):
        super(AtLocPlusCriterion, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)
        self.srx = nn.Parameter(torch.Tensor([srx]), requires_grad=learn_gamma)
        self.srq = nn.Parameter(torch.Tensor([srq]), requires_grad=learn_gamma)

    def forward(self, pred, targ):
        # absolute pose loss
        s = pred.size()
        abs_loss = torch.exp(-self.sax) * self.t_loss_fn(pred.view(-1, *s[2:])[:, :3],
                                                         targ.view(-1, *s[2:])[:, :3]) + self.sax + \
                   torch.exp(-self.saq) * self.q_loss_fn(pred.view(-1, *s[2:])[:, 3:],
                                                         targ.view(-1, *s[2:])[:, 3:]) + self.saq

        # get the VOs
        pred_vos = calc_vos_simple(pred)
        targ_vos = calc_vos_simple(targ)

        # VO loss
        s = pred_vos.size()
        vo_loss = torch.exp(-self.srx) * self.t_loss_fn(pred_vos.view(-1, *s[2:])[:, :3],
                                                        targ_vos.view(-1, *s[2:])[:, :3]) + self.srx + \
                  torch.exp(-self.srq) * self.q_loss_fn(pred_vos.view(-1, *s[2:])[:, 3:],
                                                        targ_vos.view(-1, *s[2:])[:, 3:]) + self.srq

        # total loss
        loss = abs_loss + vo_loss
        return loss


##############################
#       Model log            #
##############################
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def delink(self):
        self.log.close()

    def writeTerminalOnly(self, message):
        self.terminal.write(message)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


##############################################
#       Get backbone IntermediateLayer       #
##############################################
class IntermediateLayerGetter(nn.ModuleDict):
    """
        Module wrapper that returns intermediate layers from a model

        It has a strong assumption that the modules have been registered
        into the model in the same order as they are used.
        This means that one should **not** reuse the same nn.Module
        twice in the forward if you want this to work.

        Additionally, it is only able to query submodules that are directly
        assigned to the model. So if `model` is passed, `model.feature1` can
        be returned, but not `model.feature1.layer2`.

        Args:
            model (nn.Module): model on which we will extract the features
            return_layers (Dict[name, new_name]): a dict containing the names
                of the modules for which the activations will be returned as
                the key of the dict, and the value of the dict is the name
                of the returned activation (which the user can specify).
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        # 因为有些backbone的name是序号，是数值型
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class AverageMeter(object):
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


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_image(filename, loader=default_loader):
    try:
        img = loader(filename)
    except IOError as e:
        print('Could not load image {:s}, IOError: {:s}'.format(filename, e))
        return None
    except:
        print('Could not load image {:s}, unexpected error'.format(filename))
        return None
    return img


def qlog(q):
    if all(q[1:] == 0):
        q = np.zeros(3)
    else:
        q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
    return q


def qexp(q):
    n = np.linalg.norm(q)
    q = np.hstack((np.cos(n), np.sinc(n / np.pi) * q))
    return q


def calc_vos_simple(poses):
    vos = []
    for p in poses:
        pvos = [p[i + 1].unsqueeze(0) - p[i].unsqueeze(0) for i in range(len(p) - 1)]
        vos.append(torch.cat(pvos, dim=0))
    vos = torch.stack(vos, dim=0)
    return vos


def quaternion_angular_error(q1, q2):
    d = abs(np.dot(q1, q2))
    d = min(1.0, max(-1.0, d))
    theta = 2 * np.arccos(d) * 180 / np.pi
    return theta


def process_poses(poses_in, mean_t, std_t, align_R, align_t, align_s):
    poses_out = np.zeros((len(poses_in), 6))
    # 位置信息
    poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]

    # align
    for i in range(len(poses_out)):
        R = poses_in[i].reshape((3, 4))[:3, :3]
        # 旋转矩阵转四元数
        q = txq.mat2quat(np.dot(align_R, R))
        q *= np.sign(q[0])  # constrain to hemisphere
        q = qlog(q)         # unitize quaternion
        poses_out[i, 3:] = q
        t = poses_out[i, :3] - align_t
        poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

    # normalize translation
    poses_out[:, :3] -= mean_t
    poses_out[:, :3] /= std_t
    return poses_out


def process_poses_simplify(poses_in, mean_t, std_t):
    """

    :param poses_in: non-utilized quaternion
    :param mean_t
    :param std_t
    :return: utilized quaternion
    :date: 2022/10/31 20:44
    """
    poses_out = np.zeros((len(poses_in), 6))
    poses_out[:, 0:3] = poses_in[:, 0:3]

    for i in range(len(poses_in)):
        poses_out[i, 3:] = qlog(poses_in[i, 3:])  # unitize quaternion

    # normalize translation
    poses_out[:, :3] -= mean_t
    poses_out[:, :3] /= std_t
    return poses_out



def load_state_dict(model, state_dict):
    model_names = [n for n, _ in model.named_parameters()]
    state_names = [n for n in state_dict.keys()]

    # find prefix for the model and state dicts from the first param name
    if model_names[0].find(state_names[0]) >= 0:
        model_prefix = model_names[0].replace(state_names[0], '')
        state_prefix = None
    elif state_names[0].find(model_names[0]) >= 0:
        state_prefix = state_names[0].replace(model_names[0], '')
        model_prefix = None
    else:
        model_prefix = model_names[0].split('.')[0]
        state_prefix = state_names[0].split('.')[0]

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if state_prefix is None:
            k = model_prefix + k
        else:
            k = k.replace(state_prefix, model_prefix)
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict)


##############################
#       loss plotting        #
##############################
def plot_loss_func(epochs, loss_vals, loss_trains, loss_fig_path):
    plt.figure()
    plt.plot(range(epochs), loss_trains, 'b', label='Training loss')
    plt.plot(range(epochs), loss_vals, 'r', label='Validation loss')
    plt.grid()
    plt.title('Camera Pose Loss')
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    plt.savefig(loss_fig_path)
