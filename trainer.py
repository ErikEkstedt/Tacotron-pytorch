import torch
from collections import OrderedDict
from glob import glob
from os import makedirs
from os.path import join, isdir


class Trainer(object):
    def __init__(self, model, optimizer=None, train_loader=None, test_loader=None, writer=None, args=None):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.writer = writer
        self.args = args

        # keep track
        self.best_loss = 999
        self.epoch = 0
        self.total_batches = 0

    @staticmethod
    def get_item(tensor):
        return tensor.cpu().detach().item()

    def train(self):
        for _ in range(self.args.n_epochs): 
            train_loss = self.train_epoch()
            print(f"====> Train loss: {train_loss}")

            test_loss = self.test_epoch()
            print(f"====> Test loss: {test_loss}")

            self.writer.add_scalars('Epoch Loss', {
                'Train': train_loss,
                'Test': test_loss
            }, self.epoch)

            if test_loss < self.best_loss: 
                print(f'Saving best model')
                self.best_loss = test_loss
                self.save_checkpoint()

            self.epoch += 1

    def save_checkpoint(self):
        checkpoint_path = join(self.writer.log_dir, 'checkpoints')
        makedirs(checkpoint_path, exist_ok=True)
        file_name = f"checkpoint_{str(self.epoch).zfill(4)}_{self.best_loss:.2e}.pt"
        file_path = join(checkpoint_path, file_name)
        torch.save({'epoch': self.epoch,
                    'total_batches': self.total_batches,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.best_loss}, file_path)

    def load_checkpoint(self):
        if isdir(self.args.load_path):
            path = sorted(glob(join(self.args.load_path, 'checkpoints', '*.pt')))[-1]
        if self.args.cpu:
            checkpoint = torch.load(path, map_location='cpu')
        else:
            checkpoint = torch.load(path)

        # Handles models save directly from nn.DataParallel
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except:
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model_state_dict'].items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            # load params
            self.model.load_state_dict(new_state_dict)

        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['loss']
        self.total_batches = checkpoint['total_batches'] 
        print()
        print('#'*70)
        print('Loaded model:')
        print(f'Epoch {self.epoch}')
        print(f'Batches {self.total_batches}')
        print(f'Loss {self.best_loss}')
        print('#'*70)
        print()
