import torch
import torch.nn as nn
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import time
from collections import OrderedDict
from glob import glob
from os import makedirs
from os.path import join, isdir
from tqdm import tqdm
from scipy.io.wavfile import read, write
import hyperparams as hp
from data import inv_spectrogram, find_endpoint, _prepare_data
from text import text_to_sequence


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
            self.writer.add_scalar('Epoch Loss', train_loss, self.epoch)

            if train_loss < self.best_loss: 
                self.best_loss = train_loss
                if self.epoch > 5:
                    self.generate_audio()
                    print(f'Saving best model')
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


class TacotronTrainer(Trainer):
    # Sentences for generation
    sentences = [
            # From July 8, 2017 New York Times:
            'There’s a way to measure the acute emotional intelligence that has never gone out of style.',
            'President Trump met with other leaders at the Group of 20 conference.',
            # From Google's Tacotron example page:
            'The buses aren\'t the problem, they actually provide a solution.',
            'Does the quick brown fox jump over the lazy dog?',
            ]

    def __init__(self, model, criterion=None, **kwargs):
        super().__init__(model, **kwargs)
        self.criterion = criterion

        # Loss for frequency of human register
        self.n_priority_freq = n_priority_freq = int(3000 / (hp.sample_rate * 0.5) * hp.num_freq)

    def loss_fn(self, mel_output, linear_output, mel_spectrogram, linear_spectrogram):
        # Calculate loss
        mel_loss = self.criterion(mel_output, mel_spectrogram).mean()
        linear_loss = self.criterion(linear_output, linear_spectrogram).mean()
        # linear_loss = 0.5 * torch.mean(linear_loss) + 0.5 * torch.mean(linear_loss[:, :self.n_priority_freq,:])
        loss = mel_loss + linear_loss
        return loss, mel_loss, linear_loss

    def create_fig(self, mel, linear, mel_target, linear_target):
        mel = mel.detach().cpu().numpy()
        linear = linear.detach().cpu().numpy()
        linear_target = linear_target.detach().cpu().numpy()
        mel_target = mel_target.detach().cpu().numpy()
        plt.close()
        plt.clf()
        fig = plt.figure(figsize=(20,20))
        plt.subplot(2, 2, 1)
        plt.title('Mel Spec')
        plt.imshow(mel, aspect='auto')
        plt.gca().invert_yaxis()
        plt.subplot(2, 2, 2)
        plt.title('Linear Spec')
        plt.imshow(linear, aspect='auto')
        plt.gca().invert_yaxis()
        plt.subplot(2, 2, 3)
        plt.title('Target Mel')
        plt.imshow(mel_target, aspect='auto')
        plt.gca().invert_yaxis()
        plt.subplot(2, 2, 4)
        plt.title('Target Linear')
        plt.imshow(linear_target, aspect='auto')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        return fig

    def train_epoch(self):
        for i, data in enumerate(tqdm(self.train_loader)):
            self.model.train()
            self.optimizer.zero_grad()

            characters, linear_spectrogram, mel_spectrogram = data

            # Make decoder input by concatenating [GO] Frame
            zero_frame = torch.zeros((mel_spectrogram.shape[0], mel_spectrogram.shape[1], 1), dtype=torch.float)
            mel_input = torch.cat((zero_frame, mel_spectrogram[:, :, 1:]), dim=2).to(mel_spectrogram.device)

            # Forward
            start_time = time.time()
            mel_output, linear_output = self.model.forward(characters, mel_input)
            loss, mel_loss, linear_loss = self.loss_fn(mel_output,
                    linear_output, mel_spectrogram, linear_spectrogram)

            # Calculate gradients
            loss.backward()

            # clipping gradients
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.)

            # Update weights
            self.optimizer.step()
            time_per_step = time.time() - start_time

            loss = self.get_item(loss)

            if self.total_batches % hp.log_step == 0:
                print(f"time per step: {time_per_step:.2f}")
                print(f"At timestep {self.total_batches:d}")
                print(f"linear loss: {linear_loss:.4f}")
                print(f"mel loss: {mel_loss:.4f}")
                print(f"total loss: {loss:.4f}")
                fig = self.create_fig(mel_output[0], linear_output[0],
                        mel_spectrogram[0], linear_spectrogram[0]) 
                self.writer.add_figure('specs', fig, self.total_batches)
                self.writer.add_scalar('Batch Loss', loss, self.total_batches)
            self.total_batches += 1
        return loss

    def generate_audio(self):
        print('Generating Audio Samples')
        # Text to index sequence
        characters = []
        for text in self.sentences:
            text = np.asarray(text_to_sequence(text, [hp.cleaners]), dtype=np.int32)
            characters.append(text)

        characters = _prepare_data(characters).astype(np.int32)
        characters = torch.from_numpy(characters).long().to(self.args.device)

        # Provide [GO] Frame
        mel_input = torch.zeros([characters.shape[0], hp.num_mels, 1], dtype=torch.float).to(self.args.device)

        print('char: ', characters.shape)
        print('mel input: ', mel_input.shape)

        self.model.eval()
        # Spectrogram to wav
        _, linear_output = self.model(characters, mel_input)

        for i in range(linear_output.shape[0]):
            wav = inv_spectrogram(linear_output[i].data.cpu().numpy())
            wav = wav[:find_endpoint(wav)].astype(np.float32)
            print('wav: ', wav.shape)
            print('wav max: ', wav.max())
            print('wav min: ', wav.min())
            print('wav: ', wav.dtype)
            self.writer.add_audio('audio', wav, self.epoch, sample_rate=16000)

    def adjust_learning_rate(self):
        # for param_group in self.optimizer.param_groups:
        #     param_group['lr'] = lr
        pass


