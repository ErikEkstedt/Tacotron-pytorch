from network import Tacotron
import hyperparams as hp
from data import get_dataset, DataLoader, collate_fn, get_param_size
from torch import optim
import numpy as np
import argparse
import os
import time
import torch
from tqdm import tqdm
import torch.nn as nn




def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def adjust_learning_rate(optimizer, step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if step == 500000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005

    elif step == 1000000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0003

    elif step == 2000000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    return optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, help='Global step to restore checkpoint', default=0)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
    parser.add_argument('--device_ids', nargs='+', default=None, type=int)
    args = parser.parse_args()

    if not args.device_ids:
        args.device = 'cuda'
    else:
        args.device = f'cuda:{args.device_ids[0]}'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Get dataset
    dataset = get_dataset()

    # Construct model
    print('device ids: ', args.device_ids)
    print('continue?')
    input()
    model = Tacotron().to(device)
    model = nn.DataParallel(model, device_ids=args.device_ids)
    # model = nn.DataParallel(model)

    # Make optimizer
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)

    # Load checkpoint if exists
    try:
        checkpoint = torch.load(os.path.join(hp.checkpoint_path,'checkpoint_%d.pth.tar'% args.restore_step))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n--------model restored at step %d--------\n" % args.restore_step)
    except:
        print("\n--------Start New Training--------\n")

    # Training
    model = model.train()

    # Make checkpoint directory if not exists
    if not os.path.exists(hp.checkpoint_path):
        os.mkdir(hp.checkpoint_path)

    # Decide loss function
    criterion = nn.L1Loss().to(device)

    # Loss for frequency of human register
    n_priority_freq = int(3000 / (hp.sample_rate * 0.5) * hp.num_freq)

    for epoch in range(hp.epochs):

        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=True, collate_fn=collate_fn, drop_last=True, num_workers=8)

        for i, data in tqdm(enumerate(dataloader)):

            current_step = i + args.restore_step + epoch * len(dataloader) + 1
            optimizer.zero_grad()

            # Make decoder input by concatenating [GO] Frame
            try:
                mel_input = np.concatenate( (np.zeros( [args.batch_size, hp.num_mels, 1], dtype=np.float32), data[2][:, :, 1:]), axis=2)
            except:
                raise TypeError("not same dimension")

            characters = torch.from_numpy(data[0]).long().to(device)
            mel_input = torch.from_numpy(mel_input).float().to(device)
            mel_spectrogram = torch.from_numpy(data[2]).float().to(device)
            linear_spectrogram = torch.from_numpy(data[1]).float().to(device)

            # Forward
            mel_output, linear_output = model.forward(characters, mel_input)

            # Calculate loss
            mel_loss = criterion(mel_output, mel_spectrogram)
            linear_loss = torch.abs(linear_output-linear_spectrogram)
            linear_loss = 0.5 * torch.mean(linear_loss) + 0.5 * torch.mean(linear_loss[:,:n_priority_freq,:])
            loss = mel_loss + linear_loss
            loss = loss.cuda()

            start_time = time.time()

            # Calculate gradients
            loss.backward()

            # clipping gradients
            nn.utils.clip_grad_norm(model.parameters(), 1.)

            # Update weights
            optimizer.step()

            time_per_step = time.time() - start_time

            if current_step % hp.log_step == 0:
                print("time per step: %.2f sec" % time_per_step)
                print("At timestep %d" % current_step)
                print("linear loss: %.4f" % linear_loss.item())
                print("mel loss: %.4f" % mel_loss.data.item())
                print("total loss: %.4f" % loss.data.item())

            if current_step % hp.save_step == 0:
                save_checkpoint({'model': model.state_dict(),
                                 'optimizer': optimizer.state_dict()},
                                os.path.join(
                                    hp.checkpoint_path,
                                    'checkpoint_%d.pth.tar' % current_step))
                print("save model at step %d ..." % current_step)

            if current_step in hp.decay_step:
                optimizer = adjust_learning_rate(optimizer, current_step)
