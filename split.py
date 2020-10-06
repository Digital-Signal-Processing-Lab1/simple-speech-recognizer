import VAD
import os
import sys
import pandas as pd
import scipy.signal as signal
import pickle

N, M = 256, 128
raw_file_path = "./dataset/raw/"
unzip_path = "./dataset/unzip/"
prefix = "./dataset/processed/"
logfile = "./log"

log_f = open(logfile, "w")
assert(log_f)

if not os.path.exists(prefix):
    os.makedirs(prefix)

print("unzip...")
for ren in os.listdir(raw_file_path):
    print("unzip file {}".format(ren))
    VAD.unzip(os.path.join(raw_file_path, ren), unzip_path)
print("done.")

for window_type in ["rect", "hamming", "hanning"]:
    if window_type == 'rect':
        winfunc = 1
    elif window_type == 'hamming':
        winfunc = signal.windows.hamming(N)
    else:
        winfunc = signal.windows.hanning(N)
    print("split wave using {}...".format(window_type))
    store = []
    for ren in os.listdir(unzip_path):
        person_id = int(ren[3:])
        for content in range(10):
            wave_file = os.path.join(unzip_path, ren, "{}.wav".format(content))
            print("split {}".format(wave_file))
            wave_data, params = VAD.readWav(wave_file)
            frames, num_frame = VAD.addWindow(wave_data, N, M, winfunc)
            energy = VAD.calEnergy(frames, N).reshape(1, num_frame)
            amplitude = VAD.calAmplitude(frames, N).reshape(1, num_frame)
            zerocrossingrate = VAD.calZeroCrossingRate(frames, N).reshape(1, num_frame)
            endpoint = VAD.detectEndPoint(wave_data, energy[0], zerocrossingrate[0])

            sorted_endpoint = sorted(set(endpoint))

            if len(sorted_endpoint) %2 != 0:
                print("error raise!!! please see the log file!!")
                log_f.writelines("error at {} while using window {}. length of endpoints is not even\n"
                                 .format(window_type, wave_file))
                continue

            VAD.writeWav(store, person_id, content, wave_data, sorted_endpoint, params, N, M)

    df = pd.DataFrame(data=store, columns=['wave_data', 'person_id', 'content'])
    file_store = os.path.join(prefix, "{}.pkl".format(window_type))
    with open(file_store, "wb") as f:
        pickle.dump(df, f)
        f.close()
    print("done.")

log_f.close()