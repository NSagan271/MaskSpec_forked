import os
import h5py
from scipy import io
import argparse
import numpy as np

source_path = "./batlab_audio/single_mic_data_combined/"
out_path = "./batlab_audio/single_mic_data_combined/mp3_audio/"

FFMPEG = "~/ffmpeg/ffmpeg-git-20231103-amd64-static/ffmpeg"

def process_one(i, chirp, outdir, FS=32000):
    if i % 100 == 0:
        print(i)
    wav = out_path + 'tmp.wav'
    io.wavfile.write(wav, FS, chirp)
    
    os.system(f"{FFMPEG}  -hide_banner -nostats -loglevel error -n -i {wav} -codec:a mp3 -ar 32000 {outdir}chirp_{i}.mp3")

def read_all_chirps(dir):
    chirps = []
    max_len = 0
    files = map(lambda filename: dir + filename, os.listdir(dir))
    for filename in files:
        if '.mat' in filename:
            with h5py.File(filename, 'r') as f:
                chirp_array = f['chirp_array'][:]
                chirp_lens = f['chirp_lengths'][:]
                for i, chirp_len in enumerate(chirp_lens):
                    beginning_pad = max(chirp_len//10, 50)
                    chirps.append(np.hstack((np.zeros(beginning_pad), chirp_array[i, :chirp_len])))
                    max_len = max(max_len, chirp_len + beginning_pad)
    return chirps, max_len


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

    chirps, max_len = read_all_chirps(source_path)
    with open(source_path + "max_len.txt", "w") as f:
        f.write(str(max_len))

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
