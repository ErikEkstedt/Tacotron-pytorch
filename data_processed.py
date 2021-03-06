import torch
import hyperparams as hp
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from os.path import join
import numpy as np
from text import text_to_sequence


def _pad_data(x, length):
    _pad = 0.
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)


def _prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])


def _pad_specs(inputs):
    def pad(x, max_len):
        diff = max_len - x.shape[1]
        pad = np.zeros((x.shape[0], diff), dtype=np.float32)
        return np.hstack((x, pad))
    max_len = max((x.shape[1] for x in inputs))
    return np.stack([pad(x, max_len) for x in inputs])


def _pad_per_step(inputs):
    timesteps = inputs.shape[-1]
    return np.pad(inputs, [[0,0],[0,0],[0, hp.outputs_per_step - (timesteps % hp.outputs_per_step)]], mode='constant', constant_values=0.0)


class LJProcessedDatasets(Dataset):
    """LJSpeech dataset."""

    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the wavs.
        """
        self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        text = self.landmarks_frame.ix[idx, 1]
        text = np.asarray(text_to_sequence(text, [hp.cleaners]), dtype=np.int32)

        npy_name = join(self.root_dir, self.landmarks_frame.ix[idx, 0]) + '.npy'
        data = np.load(npy_name).item()
        sample = {'text': text, 'mel': data['mel_spec'], 'linear': data['linear_spec']}
        return sample


def processed_collate_fn(batch):
    text = []
    mel = []
    linear = []

    for d in batch:
        text.append(d['text'])
        linear.append(d['linear'])
        mel.append(d['mel'])

    mel = _pad_specs(mel)
    linear = _pad_specs(linear)

    timesteps = mel.shape[-1]

    # PAD sequences with largest length of the batch
    text = _prepare_data(text).astype(np.int32)

    # PAD with zeros that can be divided by outputs per step
    if not timesteps % hp.outputs_per_step == 0:
        linear = _pad_per_step(linear)
        mel = _pad_per_step(mel)

    text = torch.from_numpy(text).long()
    linear = torch.from_numpy(linear)
    mel = torch.from_numpy(mel)

    return text, linear, mel


def get_dataloader(
            datapath,
            batch_size=32,
            small=False,
            **kwargs):

    csv_file = join(datapath, 'metadata.csv')
    wavs = join(datapath, 'wavs')

    if small:
        csv_file = join(datapath, 'metadata_small.csv')

    dset = LJProcessedDatasets(csv_file, wavs)
    dloader = DataLoader(dset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=processed_collate_fn,
            **kwargs)

    return dloader 


if __name__ == "__main__":

    csv_file = 'Data/LJSpeech-1.1/metadata.csv'
    wavs = 'Data/LJSpeech-1.1/wavs'
    dset = LJProcessedDatasets(csv_file, wavs)

    for d in dset:
        print(d['text'].shape)
        print(d['mel'].shape)
        print(d['linear'].shape)
        break


    dloader = DataLoader(dset, batch_size=32, shuffle=True, collate_fn=processed_collate_fn, drop_last=True)

    for text, linear, mel in dloader:
        print('text: ', text.shape)
        print('text: ', text.dtype)
        print('linear: ', linear.shape)
        print('linear: ', linear.dtype)
        print('mel: ', mel.shape)
        print('mel: ', mel.dtype)
        break

