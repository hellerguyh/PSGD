from nn import *
from trainer import *
from utils import *
import pickle

DEFAULT_PARAMS = {
    'polling': True,  # Poll to check for best noise
    'all_samples': True,  # Take also batches which failed to improve
    'max_tries': 50,  # Maximum tries creating noise
    'clip_v': 4,  # Clipping the gradients to this value
    'noise_std': 0,  # Noise factor
    'train_bs': 32,  # Train batch size
    'val_bs': 32,  # Valdiation batch size
    'ref_bs': 32,  # reference batch size
    'lr': 0.001,  # learning rate
    'epochs': 1,  # Number of epochs to run
    'nn_type': 'ResNet18',  # backbone
    'db': 'CIFAR10',  # database
    'noise_retry': True,  # Apply noise until it doesn't hurt the loss or
                         # maximum retries threshold was reached
    'noise_retry_thrsld' : 50
}


def _main(config=None):
    model = NoisyNN(config['nn_type'])
    device = torch.device("cuda:" + str(config['cuda_id']) if torch.cuda.is_available() else "cpu")
    model_ft = model.nn
    model_ft.to(device)

    lr = config['lr']
    criterion = nn.CrossEntropyLoss(reduction='mean')

    optimizer = NoisyOptim(model_ft.parameters(), model_ft.named_parameters,
                           lr, config['clip_v'], config['noise_std'],
                           config['cuda_id'],
                           (config['noise_retry'], config['noise_retry_thrsld'])
                          )
    trainer = MetaCollectTrainer()
    log = trainer.train(model, criterion, optimizer, config['db'], config['epochs'],
                        config['train_bs'], config['val_bs'], config['ref_bs'],
                        config['cuda_id'])

    avg_nos_tries = float(optimizer.total_nos_repeats/optimizer.number_of_steps)
    trainer.wblogger.set('avg_nos_tries', avg_nos_tries)
    print(avg_nos_tries)

    print(log)
    if not (config['path'] is None):
        with open(config['path'][:-4] + "_val_acc_" +\
                  str(log['val_acc'][-1]) + config['path'][-4:],
                  'wb') as wf:
            log.update(config)
            pickle.dump(log, wf)
    print("Done")


'''
load_default_params() - updates dictionary with default params
d - a dictionary

if a value is not in the dictionary it is taken from DEFAULT PARAMS
'''
def load_default_params(d):
    for k in DEFAULT_PARAMS:
        if not k in d:
            d[k] = DEFAULT_PARAMS[k]


def main(config=None):
    if useWB():
        with wandb.init(name='Test Run', \
                        project='PolledSGD', \
                        notes='This is a test run', \
                        tags=['LOCAL', 'SUBSET_DATA', 'MetaData-Collect'], \
                        entity='hellerg-biu-dp',
                        config=config):
            _main(wandb.config)
    else:
        _main(config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create Victims")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--noise_retry", action="store_true")
    parser.add_argument("--noise_retry_thrsld", type=int,
                         default=DEFAULT_PARAMS['noise_retry_thrsld'])
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--lr", type=float, default=DEFAULT_PARAMS['lr'])
    parser.add_argument("--noise_std", type=float,
                        default=DEFAULT_PARAMS['noise_std'])
    parser.add_argument("--path", type=str, default=None)
    args = parser.parse_args()
    vargs = vars(args)
    load_default_params(vargs)

    if not (args.path is None):
        if args.path[-4:] != ".pkl":
            raise Exception("Path should end with .pkl")

    if not args.sweep:
        main(vargs)
    else:
        sweep_config = {
            'method': 'random',
            'parameters': {
                'noise_std': {'values': [0, 1, 2, 4, 8]},
                'ref_bs': {'values': [32, 64, 128]},
                'lr': {'values': [0.001, 0.01]},
            }
        }
        for k in vargs:
            if not k in sweep_config['parameters']:
                sweep_config['parameters'][k] = {'values': [vargs[k]]}
        sweep_id = wandb.sweep(sweep_config, project="PolledSGD")
        wandb.agent(sweep_id, function=main, count=1)
