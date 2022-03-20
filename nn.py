'''
network
'''
import torch
import torch.nn as nn
from collections import OrderedDict
import copy
import torchvision as tv
from torch.nn.utils import clip_grad_value_
from torch.optim.optimizer import Optimizer, required

class NoiseScheduler(object):
    def __init__(self, noise_std, factor, epochs, sched_type = "inc"):
        if sched_type != "inc":
            raise NotImplemented()

        if factor < 0.5:
            raise Exception("noise factor must be greater than 0.5, otherwise\
                            we will get negative noise")

        self.std = noise_std
        self.cur_noise = factor*noise_std
        self.rho = 1/factor
        self.beta = self.rho
        self.epochs = epochs

    def step(self):
        self.beta += 2*(1 - self.rho)/(self.epochs - 1)
        self.cur_noise = self.std/self.beta

    def getNoise(self):
        return self.cur_noise

class NoisyOptim(Optimizer):
    def __init__(self, params, lr = required, clip_v = 0, noise_std = 0,
                 cuda_device_id = 0, noise_on_success = (False, -1, False),
                 noise_sched = None):
        self.cuda_device_id = cuda_device_id
        defaults = dict(lr = lr)
        self.modelParams = params
        self.noise_std = noise_std
        self.base_noise = noise_std
        self.clip_v = clip_v
        self.nos = {'active' : noise_on_success[0], 'lim' : noise_on_success[1],
                    'half_sched_way' : noise_on_success[2]}
        self.total_nos_repeats = 0
        self.number_of_steps = 0
        self.noise_sched = noise_sched
        if not (noise_sched is None):
            self.noise_std = noise_sched.getNoise()
        super(NoisyOptim, self).__init__(params, defaults)

    def noise_step(self):
        print("used noise = " + str(self.noise_std))
        if not (self.noise_sched is None):
            self.noise_sched.step()
            self.noise_std = self.noise_sched.getNoise()
        print("new noise = " + str(self.noise_std))

    @torch.no_grad()
    def step(self, closure = None):
        self.number_of_steps += 1

        if self.clip_v > 0:
            clip_grad_value_(self.modelParams, self.clip_v)

        cid = self.cuda_device_id
        if cid == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:" + str(cid)
                                  if torch.cuda.is_available() else "cpu")
        
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

            # Separate into two for loops so we can show loss after gd and
            # before adding noise
            for i, param in enumerate(params_with_grad):
                d_p = d_p_list[i]
                param.add_(d_p, alpha = -lr)

            if not (closure is None):
                mid_loss, mid_loss_r, trainer = closure('mid')
            ctr = 0
            found = False

            sched_reached = True
            if self.nos['half_sched_way'] and (not (self.noise_sched is None)):
                if self.noise_std < self.base_noise:
                    sched_reached = False

            while (ctr == 0) or\
            (self.nos['active'] and (ctr < self.nos['lim']) and sched_reached\
             and not found):
                self.total_nos_repeats += 1
                if self.noise_std > 0:
                    for i, param in enumerate(params_with_grad):
                        mean = torch.zeros(param.shape)
                        noise = torch.normal(mean, self.noise_std).to(device)
                        param.add_(noise, alpha = -lr)

                if not (closure is None):
                    post_loss, post_loss_r, trainer = closure('post')

                if post_loss <= mid_loss:
                    found = True
                ctr += 1

            if not (closure is None):
                trainer.mid_gd_loss_arr.append(mid_loss)
                trainer.mid_gd_r_loss_arr.append(mid_loss_r)
                trainer.post_gd_loss_arr.append(post_loss)
                trainer.post_gd_r_loss_arr.append(post_loss_r)
                trainer.noise_retries_arr.append(ctr)


class NoisyNN(object):
    def __init__(self, nn_type = 'LeNet'):
        if nn_type == 'LeNet':
            self.nn = nn.Sequential(OrderedDict([
                                    ('conv1', nn.Conv2d(1, 6, 5)),
                                    ('relu1', nn.ReLU()),
                                    ('pool1', nn.MaxPool2d(2, 2)),
                                    ('conv2', nn.Conv2d(6, 16, 5)),
                                    ('relu2', nn.ReLU()),
                                    ('pool2', nn.MaxPool2d(2, 2)),
                                    ('conv3', nn.Conv2d(in_channels = 16,
                                                        out_channels = 120,
                                                        kernel_size = 5)),
                                    ('flatn', nn.Flatten()),
                                    ('relu3', nn.ReLU()),
                                    ('line4', nn.Linear(120, 84)),
                                    ('relu4', nn.ReLU()),
                                    ('line5', nn.Linear(84, 10)),
                                    ('softm', nn.LogSoftmax(dim = -1))
                                    ]))
        elif nn_type == 'ResNet34':
            self.nn = tv.models.resnet34(pretrained = False, num_classes = 10)
        elif nn_type == 'ResNet18':
            self.nn = tv.models.resnet18(pretrained = False, num_classes = 10)
        else:
            raise NotImplementedError(str(nn_type) +
                                      " model is not implemented")

    '''
    Returns a copy of the module weights
    this is x2.5 faster implementation than copy.deepcopy(model.state_dict))
    '''
    def createCheckPoint(self):
        return copy.deepcopy(dict(self.nn.named_parameters()))

    def loadCheckPoint(self, cp):
        netp = dict(self.nn.named_parameters())
        for name in cp:
            netp[name].data.copy_(cp[name].data)

    def saveWeights(self, path, use_wandb = False, wandb_run = None):
        torch.save(self.nn.state_dict(), path)
        if use_wandb:
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(path)
            wandb_run.log_artifact(artifact)
            wandb_run.join()

    def loadWeights(self, path, use_wandb = False, wandb_path = None,
                    wandb_run = None):
        if use_wandb:
            artifact = wandb_run.use_artifact(wandb_path, type = 'model')
            artifact_dir = artifact.download(path)
            wandb_run.join()
        self.nn.load_state_dict(torch.load(path))
