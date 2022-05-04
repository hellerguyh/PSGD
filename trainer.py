import torch
import wandb
from data import *
from torch.nn.utils import clip_grad_value_

from utils import useWB, WBLogger

def logProgress(num_batches):
    idx = 0
    last_per = 0
    while True:
        idx += 1
        per = int(idx*5/num_batches)
        if per > last_per:
            last_per = per
            print("Finished " + str(idx*100/num_batches) + "%")
        yield None

 
class Trainer(object):
    def __init__(self):
        self.wblogger = WBLogger()
    def predict(self, inputs, labels, criterion, model_ft): 
        inputs = inputs.to(self.device)
        outputs = model_ft(inputs)
        
        labels = labels.to(self.device)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs.detach(), 1)
        corr = torch.sum(preds == labels.detach().data)
        
        return loss, corr, outputs

    def gradStep(self, loss, optimizer, inputs, labels, criterion, model_ft):
        loss.backward()
        optimizer.step()

    def valLoop(self, model_ft, criterion, dl, loss_arr, acc_arr, ds_len):
        logger_itr = logProgress(len(dl))
        model_ft.eval()
        with torch.no_grad():
            loss_sum = 0
            corr_sum = 0
            for inputs, labels in dl:
                next(logger_itr)
                loss, corr,_ = self.predict(inputs, labels, criterion,
                                            model_ft)
                
                loss_sum += float(loss.detach().cpu().data)
                corr_sum += float(corr.cpu().data)
            loss_arr.append(loss_sum)
            acc_arr.append(corr_sum/ds_len)
        self.wblogger.set('val_loss', loss_sum)
        self.wblogger.set('val_acc', corr_sum/ds_len)

    def train(self, model, criterion, optimizer, ds_name, num_epochs,
              t_bs, v_bs, cuda_id = -1):
        t_dl, td_len = getDL(t_bs, True, ds_name)
        v_dl, vd_len = getDL(v_bs, True, ds_name)
        
        self.device = torch.device("cuda:" + str(cuda_id) if\
                      torch.cuda.is_available() and cuda_id > -1 else "cpu")
        model_ft = model.nn
        model_ft.to(self.device)
    
        t_loss_arr = []
        v_loss_arr = []
        t_acc_arr = []
        v_acc_arr = []
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            self.wblogger.set('epoch', epoch)
            logger_itr = logProgress(len(t_dl))

            model_ft.train()
            loss_sum = 0
            corr_sum = 0
            for inputs, labels in t_dl:
                next(logger_itr)
                optimizer.zero_grad()
                loss, corr,_ = self.predict(inputs, labels, criterion,
                                            model_ft)                
                loss_sum += float(loss.detach().cpu().data)
                corr_sum += float(corr.cpu().data)
                self.gradStep(loss, optimizer, inputs, labels, criterion,
                              model_ft)
                self.wblogger.inc()
            t_loss_arr.append(loss_sum)
            t_acc_arr.append(corr_sum/td_len)
            self.valLoop(model_ft, criterion, v_dl, v_loss_arr, v_acc_arr, vd_len)
            optimizer.noise_sched_step()
        
        return {
                'train_loss': t_loss_arr, 
                'train_acc' : t_acc_arr,
                'val_loss'  : v_loss_arr,
                'val_acc'   : v_acc_arr
                }

class MetaCollectTrainer(Trainer):
    def _gradStepLogging(self):
        self.wblogger.set('pre_gd_loss', self.pre_gd_loss_arr[-1])
        self.wblogger.set('post_gd_loss', self.post_gd_loss_arr[-1])
        self.wblogger.set('pre_gd_r_loss', self.pre_gd_r_loss_arr[-1])
        self.wblogger.set('post_gd_r_loss', self.post_gd_r_loss_arr[-1])
        self.wblogger.set('mid_gd_loss', self.mid_gd_loss_arr[-1])
        self.wblogger.set('mid_gd_r_loss', self.mid_gd_r_loss_arr[-1])
        self.wblogger.set('noise_retries', self.noise_retries_arr[-1])
        
    '''
    We want to collect the following debug data:
        1. difference in loss on current batch, before and after gradient step
        2. difference in loss on a reference batch, before and after gradient
           step
        3. loss after gradients update and before adding the noise
    To accomadate 1, we will save the loss at each step before and after update
    '''
    def gradStep(self, loss, optimizer, inputs, labels, criterion, model_ft):
        def lossClosure(state):
            with torch.no_grad():
                tloss, _, _ = self.predict(inputs, labels, criterion, model_ft)
                tloss = float(tloss.detach().cpu().data)
                tloss_r, _, _ = self.predict(self.rinputs, self.rlabels, criterion, model_ft)
                tloss_r = float(tloss_r.detach().cpu().data)
                return tloss, tloss_r, self

        self.pre_gd_loss_arr.append(float(loss.detach().cpu().data))
        with torch.no_grad():
            tloss, _, _ = self.predict(self.rinputs, self.rlabels, criterion, model_ft)
            self.pre_gd_r_loss_arr.append(float(tloss.detach().cpu().data))

        loss.backward()
        optimizer.step(lossClosure)
        
        self._gradStepLogging()
        
    def trainPreprocess(self, ds_name, r_bs):
        print("Using reference batch size of " + str(r_bs))
        r_dl, _ = getDL(r_bs, True, ds_name)
        itr = iter(r_dl)
        self.rinputs, self.rlabels = next(itr)
        self.pre_gd_loss_arr = []
        self.post_gd_loss_arr = []
        self.pre_gd_r_loss_arr = []
        self.post_gd_r_loss_arr = []
        self.mid_gd_loss_arr = []
        self.mid_gd_r_loss_arr = []
        self.noise_retries_arr = []

    def train(self, model, criterion, optimizer, ds_name, num_epochs,
              t_bs, v_bs, r_bs, cuda_id = -1):
        
        self.trainPreprocess(ds_name, r_bs)
        log = super(MetaCollectTrainer, self).train(model, criterion, optimizer,
                                                    ds_name, num_epochs,
                                                    t_bs, v_bs, cuda_id)
        log.update({
                    'pre_gd_loss'   : self.pre_gd_loss_arr,
                    'post_gd_loss'  : self.post_gd_loss_arr,
                    'pre_gd_r_loss' : self.pre_gd_r_loss_arr,
                    'post_gd_r_loss': self.post_gd_r_loss_arr,
                    'mid_gd_loss'   : self.mid_gd_loss_arr,
                    'mid_gd_r_loss'   : self.mid_gd_r_loss_arr,
                    'noise_retries_arr'   : self.noise_retries_arr,
                    })
        return log
