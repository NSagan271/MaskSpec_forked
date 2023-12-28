import torch
import numpy as np
import h5py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from scv2.dataset import decode_mp3, pad_or_truncate
from models.models_mae import AugmentMelSTFT 

def get_mean_std(n_mel=128, sample_number=10000, hopsize=10):
    print('Start...')
    hdf5_file = './batlab_audio/single_mic_data/batlab_data_train_mp3.hdf'
    dataset = h5py.File(hdf5_file, 'r')
    length = len(dataset['audio_name'])
    Mel = AugmentMelSTFT(
        n_mels=n_mel, sr=32000, win_length=800, hopsize=hopsize, n_fft=1024, freqm=0,
        timem=0, htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=10,
        fmax_aug_range=2000, device='cpu')
    all_mean = 0.
    all_std = 0.
    count = 0
    frames = []
    if sample_number > length:
        sample_number = length
    index = np.random.choice(np.arange(length), size=sample_number, replace=False)
    for i in index:
        count += 1
        if count % 200 == 0:
            print(count, length)
        waveform = decode_mp3(dataset['mp3'][i])
        waveform = torch.Tensor(pad_or_truncate(waveform, 32000))
        waveform = waveform[None, :]
        mel = Mel(waveform)
        mel = mel.numpy()
        all_mean += mel.mean()
        all_std += (np.max(mel) - np.min(mel)) / 4 # using the maximum as a proxy for the stdev
                                       # for this dataset
        frames.append(mel[0])
    dataset.close()
    all_mean = 0#all_mean / sample_number
    all_std = all_std / sample_number
    print('all mean', all_mean)
    print('all std', all_std)
    # frequencies = np.concatenate(frames, 0)
    frames = np.concatenate(frames, 1)
    frame_mean = np.mean(frames, axis=1) * 0
    frame_std = (np.max(frames, axis=1) - np.min(frames, axis=1)) / 4 # using the maximum as a proxy for the stdev
    print('frame mean', frame_mean)
    print('frame std', frame_std)
    print(frame_mean.shape, frames.shape)
    dict = {
        'all_mean':all_mean,'all_std':all_std, 
        'frame_mean':frame_mean, 'frame_std':frame_std}
    np.save('./batlab_audio/mean_std_'+str(n_mel)+'.npy', dict)
    return {"done": True}

def test(n_mel=128):
    mean_std_file = np.load('./batlab_audio/mean_std_'+str(n_mel)+'.npy',allow_pickle=True).item()
    frame_mean = torch.Tensor(mean_std_file['frame_mean']).cuda()
    frame_std = torch.Tensor(mean_std_file['frame_std']).cuda()
    all_mean = mean_std_file['all_mean']
    all_std = mean_std_file['all_std']
    print(all_mean, all_std)

if __name__ == '__main__':
    get_mean_std(n_mel=128) # for vit
    get_mean_std(n_mel=64) # for swin
    test(n_mel=128)
    test(n_mel=64)
