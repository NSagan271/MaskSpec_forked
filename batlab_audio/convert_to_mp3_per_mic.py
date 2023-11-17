import os
import h5py
from scipy import io, signal
import argparse
import numpy as np

source_path = "./batlab_audio/single_mic_data/"
out_path = "./batlab_audio/single_mic_data/mp3_audio/"

def process_one(i, chirp, outdir, FS=32000):
    if i % 100 == 0:
        print(i)
    wav = out_path + 'tmp.wav'
    io.wavfile.write(wav, FS, chirp)
    
    os.system(f"ffmpeg  -hide_banner -nostats -loglevel error -n -i {wav} -codec:a mp3 -ar 32000 {outdir}chirp_{i}.mp3")

def read_all_chirps(dir):
    chirps = []
    files = map(lambda filename: source_path + filename, os.listdir(dir))
    for filename in files:
        if '.mat' in filename:
            with h5py.File(filename, 'r') as f:
                chirps.extend(f['chirp_array'][:])
    return chirps

def get_sample(chirp, upsample_rate=4):
    x = signal.resample(chirp, len(chirp) * upsample_rate)
    x = x * signal.windows.tukey(len(x))

    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=False, default=None,
                        help='Path of folder containing the MAT files')
    parser.add_argument('--out', type=str, required=False, default=None,
                        help='Directory to save out the converted mp3s.')
    parser.add_argument('--perc_eval', type=float, required=False, default=20,
                        help='Percentage of files to use for evaluation')
    parser.add_argument('--perc_test', type=float, required=False, default=5,
                        help='Percentage of files to use for testing')

    args = parser.parse_args()

    source_path = args.source or source_path
    out_path = args.out or out_path
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(out_path + 'train', exist_ok=True)
    os.makedirs(out_path + 'eval', exist_ok=True)
    os.makedirs(out_path + 'test', exist_ok=True)

    chirps = read_all_chirps(source_path)
    idxs = np.arange(len(chirps))
    np.random.shuffle(idxs)
    for i, idx in enumerate(idxs):
        out_dir = out_path + 'train/'
        if i < idxs.shape[0] * args.perc_test / 100:
            out_dir = out_path + 'test/'
        elif i < idxs.shape[0] * (args.perc_eval + args.perc_test) / 100:
            out_dir = out_path + 'eval/'
        
        process_one(i, chirps[idx], out_dir)
        

    os.system('stty sane')