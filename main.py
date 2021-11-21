import torch
import torch.optim as optim
import wandb
import os

from data import *
from nn import *
from train import *

# Default values
#MAX_TRIES = 50
#CLIP_V = 1
#SIGMA = 4
#T_BS = 64
#V_BS = 128
#R_BS = 256
#POLLING = True
#LR = 0.001
#EPOCHS = 8

DEFAULT_PARAMS = {
    'polling'   : True, #Poll to check for best noise
    'all_samples' : True, #Take also batches which failed to improve
    'max_tries' : 50, #Maximum tries creating noise
    'clip_v'    : 1, #Clipping the gradients to this value
    'sigma'     : 4, #Noise factor
    'train_bs'  : 64, #Train batch size
    'val_bs'    : 128, #Valdiation batch size
    'ref_bs'    : 256, #reference batch size
    'lr'        : 0.001, #learning rate
    'epochs'    : 8, #Number of epochs to run
    'nn_type'   : 'ResNet34', #backbone
    'db'        : 'CIFAR10' #database
}

def main(config=None):
    torch.manual_seed(0)

    with wandb.init(name='Test Run',\
           project = 'PolledSGD',\
           notes = 'This is a test run',\
           tags = ['Test Run', 'LOCAL', 'SUBSET_DATA'],\
           entity = 'hellerguyh',
           config = config):

        for k in DEFAULT_PARAMS:
            if not k in wandb.config:
                wandb.config[k] = DEFAULT_PARAMS[k]

        model = NoisyNN(wandb.config.nn_type)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_ft = model.nn
        model_ft.to(device)

        lr = wandb.config.lr

        criterion = nn.CrossEntropyLoss(reduction = 'mean')
        optimizer = optim.SGD(model_ft.parameters(), lr)
        schd_step = 1
        schd_gamma = 0.75
        scheduler = None#optim.lr_scheduler.StepLR(optimizer, schd_step, gamma = schd_gamma)

        if scheduler:
            wandb.config.scheduler_step = schd_step
            wandb.config.scheduler_gamma = schd_gamma
        else:
            wandb.config.scheduler_step = -1
            wandb.config.scheduler_gamma = -1


        model_ft = train_model(model, criterion, optimizer, scheduler)
        print("Done")

if __name__ == "__main__":
    sweeping = os.getenv('POLLING_SGD_WANDB_SWEEPING', False) == 'True'
    if not sweeping:
        main()
    else:
        sweep_config = {
                        'method': 'random',
                        'parameters': {
                            'sigma'     : {'values' : list(range(4,16))},
                            'polling'   : {'values' : [True, False]}
                        }
                      }
        sweep_id = wandb.sweep(sweep_config, project="PolledSGD")
        wandb.agent(sweep_id, function=main,count = 3)
