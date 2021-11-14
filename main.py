import torch
import torch.optim as optim
import wandb

from data import *
from nn import *
from train import *

wandb.init(name='Test Run',
           project='PolledSGD',
           notes='This is a test run',
           tags=['Test Run', 'LOCAL', 'SUBSET_DATA'],
           entity='hellerguyh')

torch.manual_seed(0)

model = NoisyNN()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model.nn
model_ft.to(device)

lr = 0.001

wandb.config.lr = lr

criterion = nn.CrossEntropyLoss(reduction = 'mean')
optimizer = optim.SGD(model_ft.parameters(), lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma = 0.75)

model_ft = train_model(model, criterion, optimizer, scheduler, lr , 8)
print("Done")
