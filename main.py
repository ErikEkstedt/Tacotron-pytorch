import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import hyperparams as hp
from network import Tacotron
from trainer import TacotronTrainer
from data_processed import get_dataloader 
import argparse




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, help='Global step to restore checkpoint', default=0)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
    parser.add_argument('--device_ids', nargs='+', default=None, type=int)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--small', action='store_true')
    args = parser.parse_args()

    if not args.device_ids:
        args.device = 'cuda:6' if torch.cuda.is_available() else 'cpu'
        args.device_ids = [6]
    else:
        args.device = f'cuda:{args.device_ids[0]}'

    for k,v in vars(args).items():
        print(f'{k}: {v}')


    writer = SummaryWriter()

    dloader = get_dataloader(
            'Data/LJSpeech-1.1',
            batch_size=args.batch_size,
            small=args.small,
            drop_last=True)

    # Construct model
    print('device ids: ', args.device_ids)
    print('device: ', args.device)
    # print('continue?')
    # input()
    model = Tacotron().to(args.device)
    model = nn.DataParallel(model, device_ids=args.device_ids)
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    criterion = nn.DataParallel(nn.L1Loss(), device_ids=args.device_ids)

    trainer = TacotronTrainer(
            model,
            criterion,
            optimizer=optimizer,
            train_loader=dloader,
            writer=writer,
            args=args)

    trainer.train()
