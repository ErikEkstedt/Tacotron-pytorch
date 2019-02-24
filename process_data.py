import numpy as np
from glob import glob
from os.path import join, expanduser, basename, split
from tqdm import tqdm
from multiprocessing import Pool
from scipy.io.wavfile import read

from data import _stft, _amp_to_db, _linear_to_mel, _normalize, preemphasis
import hyperparams as hp


def read_wav(path):
    sr, y = read(path)
    y = y.astype(np.float32)
    y /= 2**15
    return y.astype(np.float32), sr


def extract_spectrograms(y):
    '''
    Extracts linear and Mel Spectrograms
    '''
    D = np.abs(_stft(preemphasis(y))).astype(np.float32)
    S = _amp_to_db(D) - hp.ref_level_db
    MelS = _amp_to_db(_linear_to_mel(D)) - hp.ref_level_db
    return _normalize(MelS).astype(np.float32), _normalize(S).astype(np.float32)


def save_spectrogram(wavpath):
    name = basename(wavpath).strip('.wav')
    savepath = join(split(wavpath)[0], name + '.npy') 
    y, sr = read_wav(wavpath)
    mel, lin = extract_spectrograms(y)
    np.save(savepath, {'mel_spec': mel, 'linear_spec': lin})


def run_in_parallel(func, iterable, desc):
    ''' pool.map is faster but wont work with tqdm 
    Arguments:
        func:           function to process iterable with
        iterable:       iterable of data
        desc:           str, description for tqdm
    Returns:
        out:            list of result
    '''
    with Pool() as pool:
        out = list(tqdm(pool.imap(func, iterable),
            total=len(iterable),
            desc=desc))
    return out


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Tacotron Preprocess')
    parser.add_argument('--datapath', default='Data/LJSpeech-1.1')
    parser.add_argument('--outpath', default='Data/specs')
    args = parser.parse_args()

    wavpaths = glob(join(args.datapath, '**', '*.wav'))
    run_in_parallel(save_spectrogram, wavpaths, 'Extracting spectrograms')
