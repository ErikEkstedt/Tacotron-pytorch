import hyperparams as hp
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import librosa
import numpy as np
# from Tacotron.text import text_to_sequence
from text import text_to_sequence
import collections
from scipy import signal


def _pad_data(x, length):
    _pad = 0
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)


def _prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])


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
        # return len(self.landmarks_frame)
        return 5 

    def __getitem__(self, idx):
        text = self.landmarks_frame.ix[idx, 1]
        text = np.asarray(text_to_sequence(text, [hp.cleaners]), dtype=np.int32)

        npy_name = os.path.join(self.root_dir, self.landmarks_frame.ix[idx, 0]) + '.npy'
        data = np.load(npy_name).item()
        sample = {'text': text, 'mel': data['mel_spec'], 'linear': data['linear_spec']}
        return sample


def processed_collate_fn(batch):

    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.Mapping):
        keys = list()

        text = []
        mel = []
        magnitude = []

        for d in batch:
            text.append(d['text'])
            magnitude.append(d['linear'])
            mel.append(d['mel'])

        timesteps = mel.shape[-1]

        # PAD sequences with largest length of the batch
        text = _prepare_data(text).astype(np.int32)

        # PAD with zeros that can be divided by outputs per step
        if timesteps % hp.outputs_per_step != 0:
            magnitude = _pad_per_step(magnitude)
            mel = _pad_per_step(mel)

        return text, magnitude, mel

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))


if __name__ == "__main__":

    csv_file = 'Data/LJSpeech-1.1/metadata.csv'
    wavs = 'Data/LJSpeech-1.1/wavs'
    dset = LJProcessedDatasets(csv_file, wavs)

    d = dset[0]

    for d in dset[:4]:
        print(d['text'].shape)
        print(d['mel'].shape)
        print(d['linear'].shape)

    dloader = DataLoader(dset, batch_size=2, shuffle=True, collate_fn=processed_collate_fn, drop_last=True)
