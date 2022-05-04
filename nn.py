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
    def __init__(self, noise_std, factor, epochs, step_start):
        self.noise_std = noise_std
        self.factor = factor
        self.epochs = epochs
        self.step_start = step_start

    def step(self):
        return NotImplemented()

    def getNoise(self):
        return self.cur_noise


class ContNoiseScheduler(NoiseScheduler):
    def __init__(self, *args):
        super(ContNoiseScheduler, self).__init__(*args)
        if self.factor < 0.5:
            raise Exception("noise factor must be greater than 0.5, otherwise\
                            we will get negative noise")

        self.std = self.noise_std
        self.cur_noise = self.factor*self.noise_std
        self.rho = 1/self.factor
        self.beta = self.rho

    def step(self):
        self.beta += 2*(1 - self.rho)/(self.epochs - 1)
        self.cur_noise = self.std/self.beta


class StepNoiseScheduler(NoiseScheduler):
    def __init__(self, *args):
        super(StepNoiseScheduler, self).__init__(*args)
        if self.step_start < 1:
            raise Exception("Cant start step on the first epoch, i.e. on epoch\
                             0")
        if self.factor <= self.step_start/self.epochs:
            raise Exception("can't waste all your privacy budget on the first step")
        self.std = self.noise_std
        self.cur_noise = self.factor*self.noise_std
        self.steps = 0

    def step(self):
        if self.steps == self.step_start - 1:
            alpha = (self.epochs - self.step_start)/(self.epochs - self.step_start/self.factor)
            self.cur_noise = alpha*self.std
        self.steps += 1

class ReverseStepNoiseScheduler(NoiseScheduler):
    def __init__(self, *args):
        super(ReverseStepNoiseScheduler, self).__init__(*args)
        if self.step_start > self.epochs - 1:
            raise Exception("step_start must be smaller than the number of epochs")

        self.small_noise_epochs = self.epochs - self.step_start

        if self.factor <= self.small_noise_epochs/self.epochs:
            raise Exception("can't waste all your privacy budget on the small noise steps")

        self.std = self.noise_std
        alpha = (self.epochs - self.small_noise_epochs)/(self.epochs - self.small_noise_epochs/self.factor)
        self.cur_noise = self.noise_std*alpha
        self.steps = 0

    def step(self):
        if self.steps == self.step_start - 1:
            self.cur_noise = self.factor*self.noise_std
        self.steps += 1

def NoiseSchedulerFactory(*args, ns_type):
    if ns_type == 'step':
        cls = StepNoiseScheduler
    elif ns_type == 'inc':
        cls = ContNoiseScheduler
    elif ns_type == 'revstep':
        cls = ReverseStepNoiseScheduler
    else:
        raise NotImplemented()
    return cls(*args)

class NoisyOptim(Optimizer):
    def __init__(self, params_gen_f, named_params_f, lr = required, clip_v = 0,
                 noise_std = 0, cuda_device_id = 0,
                 noise_on_success = (False, -1), noise_sched = None,
                 accept_rej = False):
        if accept_rej and noise_on_success[0]:
            raise Exception("Accept Reject can't work with noise on success")
        if noise_std == 0 and noise_on_success[0]:
            print("Are you sure you want to use noise retry w. noise_std == 0?")
        self.cuda_device_id = cuda_device_id
        defaults = dict(lr = lr)
        self.modelParams = list(params_gen_f())
        self.model_named_params_f = named_params_f
        self.noise_std = noise_std
        self.clip_v = clip_v
        self.nos = noise_on_success
        self.total_nos_repeats = 0
        self.number_of_steps = 0
        self.noise_sched = noise_sched
        self.accept_rej = accept_rej
        if not (noise_sched is None):
            self.noise_std = noise_sched.getNoise()
        super(NoisyOptim, self).__init__(params_gen_f(), defaults)

    '''
    Returns a copy of the module weights
    this is x2.5 faster implementation than copy.deepcopy(model.state_dict))
    '''
    def createCheckPoint(self):
        a = copy.deepcopy(dict(self.model_named_params_f()))
        return a

    def loadCheckPoint(self, cp):
        netp = dict(self.model_named_params_f())
        for name in cp:
            netp[name].data.copy_(cp[name].data)

    def noise_sched_step(self):
        print("used noise = " + str(self.noise_std))
        if not (self.noise_sched is None):
            self.noise_sched.step()
            self.noise_std = self.noise_sched.getNoise()
        print("new noise = " + str(self.noise_std))

    def getDevice(self):
        cid = self.cuda_device_id
        if cid == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:" + str(cid)
                                  if torch.cuda.is_available() else "cpu")
        return device

    @torch.no_grad()
    def retry_noise_add(self, device, closure, params_with_grad, lr,
                        mid_loss):
        cp = self.createCheckPoint()
        ctr = 0
        retry_active = self.nos[0] and not (closure is None)
        while (ctr == 0) or retry_active:
            self.total_nos_repeats += 1
            if self.noise_std > 0:
                for i, param in enumerate(params_with_grad):
                    mean = torch.zeros(param.shape)
                    noise = torch.normal(mean, self.noise_std).to(device)
                    param.add_(noise, alpha = -lr)

            if not (closure is None):
                post_loss, post_loss_r, trainer = closure('post')
                if retry_active:
                    if (post_loss <= mid_loss) or (ctr == self.nos[1]):
                        retry_active = False
                    else:
                        self.loadCheckPoint(cp)
            ctr += 1

        trainer.noise_retries_arr.append(ctr)
        return post_loss, post_loss_r, trainer

    @torch.no_grad()
    def pre_accept_rej(self):
        self.accept_rej_cp = self.createCheckPoint()

    @torch.no_grad()
    def accept_rej_step(self, device, closure, params_with_grad, lr,
                        mid_loss_r):
        for i, param in enumerate(params_with_grad):
            mean = torch.zeros(param.shape)
            noise = torch.normal(mean, self.noise_std).to(device)
            param.add_(noise, alpha = -lr)

        post_loss, post_loss_r, trainer = closure('post')
        if (post_loss_r > mid_loss_r):
            self.loadCheckPoint(self.accept_rej_cp)
            trainer.accept_rej_arr.append(0)
        else:
            trainer.accept_rej_arr.append(1)

        return post_loss, post_loss_r, trainer

    @torch.no_grad()
    def step(self, closure = None):
        self.number_of_steps += 1

        if self.clip_v > 0:
            clip_grad_value_(self.modelParams, self.clip_v)

        device = self.getDevice()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            lr = group['lr']

            if self.accept_rej:
                self.pre_accept_rej()

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

            if self.accept_rej:
                res = self.accept_rej_step(device, closure, params_with_grad,
                                           lr, mid_loss_r)
            else:
                res = self.retry_noise_add(device, closure, params_with_grad,
                                           lr, mid_loss)
            post_loss, post_loss_r, trainer = res

            if not (closure is None):
                trainer.mid_gd_loss_arr.append(mid_loss)
                trainer.mid_gd_r_loss_arr.append(mid_loss_r)
                trainer.post_gd_loss_arr.append(post_loss)
                trainer.post_gd_r_loss_arr.append(post_loss_r)


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
