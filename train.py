import torch
import wandb
from data import *
from torch.nn.utils import clip_grad_value_


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

def getRefData(rdl_itr, model_ft, device, criterion):
  with torch.no_grad():
    inputs, labels = next(rdl_itr)
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model_ft(inputs)
    loss = criterion(outputs, labels)
    return loss, labels, inputs

def runModel(model_ft, inputs, labels, loss_sum, corr_sum, device, criterion):
    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = model_ft(inputs)
    loss = criterion(outputs, labels)
    _, preds = torch.max(outputs, 1)
    corr = torch.sum(preds == labels.data)
    corr_sum += corr
    loss_sum += loss.item()
    return loss, corr_sum, loss_sum

def logEpochResult(loss_sum, corr_sum, ds_size, phase, loss_arr):
    epoch_loss = loss_sum / ds_size 
    epoch_acc = corr_sum.double() / ds_size
    loss_arr[phase].append(epoch_loss)
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
           phase, epoch_loss, epoch_acc))
    wandb.log({"Epoch" : len(loss_arr[phase]) - 1,
                phase + " Loss" : epoch_loss,
                phase + " Accuracy" : epoch_acc},
               step = len(loss_arr[phase]) - 1)

def gradStep(model, rdl_itr, loss, optimizer, scheduler, device, criterion, lr,
             bs, dbg_inputs, dbg_labels):
    model_ft = model.nn
    
    with torch.no_grad():
        cp_0 = model.createCheckPoint()
        rloss, rlabels, rinputs = getRefData(rdl_itr, model_ft, device,
                                             criterion)
        '''
        dbg_inputs = dbg_inputs.to(device)
        dbg_labels = dbg_labels.to(device)
        dbg_outputs = model_ft(dbg_inputs)
        rloss = criterion(dbg_outputs, dbg_labels)
        rinputs = dbg_inputs
        rlabels = dbg_labels
        '''
    
    loss.backward()
    clip_grad_value_(model_ft.parameters(), wandb.config.clip_v)
    optimizer.step()
    
    unusable_sample = False
    with torch.no_grad():
        troutputs = model_ft(rinputs)
        trloss = criterion(troutputs, rlabels)
        if trloss > rloss:
            unusable_sample = True
    
    if scheduler:
        eff_lr = scheduler.get_last_lr()[0]
    else:
        eff_lr = lr
    with torch.no_grad():
        cp_1 = model.createCheckPoint()
        found_noise = False
        for i in range(wandb.config.max_tries):
            std = eff_lr*wandb.config.sigma*wandb.config.clip_v
            model.addNoise(std, device, bs)

            if wandb.config.polling is False:
                found_noise = True
                break

            troutputs = model_ft(rinputs)
            trloss = criterion(troutputs, rlabels)
            if trloss < rloss:
                found_noise = True
                break
            else:
                model.loadCheckPoint(cp_1)
    
    if found_noise is False:
        model.loadCheckPoint(cp_0)
    
    
    return found_noise, unusable_sample

def train_model(model, criterion, optimizer, scheduler):

    t_dl, v_dl, r_dl = getDataLoaders(wandb.config.train_bs, 
                                      wandb.config.val_bs,
                                      wandb.config.ref_bs)
    dataloaders = {'train' : t_dl, 'val' : v_dl}
    rdl_itr = getULItr(r_dl)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model.nn
    model_ft.to(device)
    
    loss_arr = {'train':[],'val':[]}
    skipped_batches_arr = []
    grad_pos_arr = []
    num_epochs = wandb.config.epochs
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            dl = dataloaders[phase]
            if phase == 'train':
                model_ft.train()
            else:
                model_ft.eval()

            if phase == 'train':
                logger_itr = logProgress(len(dl))
            loss_sum = 0
            corr_sum = 0
            skipped_batches = 0
            grad_pos_cntr = 0
            for inputs, labels in dl:
                next(logger_itr)
                
                optimizer.zero_grad()
                
                loss, corr_sum, loss_sum = runModel(model_ft, inputs, labels, 
                                                    loss_sum, corr_sum, device,
                                                    criterion)
                
                if phase == 'train':
                    found_noise, unusable_sample = gradStep(model, rdl_itr,
                                                            loss, optimizer, 
                                                            scheduler, device,
                                                            criterion,
                                                            wandb.config.lr,
                                                            dl.batch_size,
                                                            inputs, labels)
                    skipped_batches += int(found_noise == False)
                    grad_pos_cntr += int(unusable_sample == True)

            if phase == 'train' and scheduler is not None:
                scheduler.step()

            ds_size = len(dl)*dl.batch_size
            logEpochResult(loss_sum, corr_sum, ds_size, phase, loss_arr)

            if phase == 'train':
                skipped_batches_arr.append(skipped_batches/len(dl))
                grad_pos_arr.append(grad_pos_cntr/len(dl))
                print("% skipped_batches: " + str(skipped_batches*100/len(dl)))
                print("% Unusable batches :"  + str(grad_pos_cntr*100/len(dl)))

                wandb.log({
                           "% Skipped Batches" : skipped_batches*100/len(dl),
                           "% Unusable Batches" : grad_pos_cntr*100/len(dl)},
                           step = len(loss_arr[phase]) - 1)
