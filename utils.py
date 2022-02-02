import os
import wandb

def useWB():
    return os.getenv('WANDB_ACTIVE_DEBUG', False) == 'True'


class WBLogger(object):
    def __init__(self):
        self.step = 0
        self.dic = {}
        self.on = useWB()

    def set(self, name, value):
        self.dic[name] = value

    def inc(self):
        if self.on:
            wandb.log(self.dic, step = self.step)
            self.step += 1
     
                
if __name__ == "__main__":
    a = WBLogger()
    a['koko'] = 5
    print(a)
