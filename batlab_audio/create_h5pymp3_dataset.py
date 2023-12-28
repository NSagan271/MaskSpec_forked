import h5py
import numpy as np
import os

base_dir = "./batlab_audio/single_mic_data_combined/"
mp3_base_path = "./batlab_audio/single_mic_data_combined/mp3_audio/"

max_len = None
with open(base_dir + "max_len.txt") as f:
    max_len = np.ones(1, dtype='uint32') * int(f.read())


for subdir in ['train', 'eval', 'test']:
    mp3_path = mp3_base_path + subdir + '/'
    save_file = 'batlab_data_' + subdir
    print("now working on ", mp3_path)

    all_count = 0
    available_files = os.listdir(mp3_path)

    available_size = len(available_files)
    dt = h5py.vlen_dtype(np.dtype('uint8'))
    dt_len = h5py.vlen_dtype(np.dtype('uint32'))
    with h5py.File(base_dir + save_file + "_mp3.hdf", 'w') as hf:
        audio_name = hf.create_dataset('audio_name', shape=((available_size,)), dtype='S20')
        waveform = hf.create_dataset('mp3', shape=((available_size,)), dtype=dt)
        max_len_ds = hf.create_dataset('max_len', shape=((1,)), dtype=dt_len)
        max_len_ds[0] = max_len
        for i, file in enumerate(available_files):
            if i % 100 == 0:
                print(f"{i}/{available_size}")
            f = file
            a = np.fromfile(mp3_path + f, dtype='uint8')
            audio_name[i] = f.encode()
            waveform[i] = a
print("Done!")
