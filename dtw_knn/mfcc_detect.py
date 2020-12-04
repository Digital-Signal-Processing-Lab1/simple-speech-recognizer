from librosa.feature import mfcc
import numpy as np


def mfcc_power(d: np.ndarray, n_mfcc=20, hop_length=256, n_fft=512, winfunc=1):
    mfcc_f = mfcc(d, 16000, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft, window=winfunc)
    return mfcc_f


def distance(d0, d1):
    d = np.sum(np.power(d0[1:]-d1[1:], 2)) + np.power(d0[0] - d1[0], 2)
    return np.sqrt(d)


def cal_mfcc_distances(d: np.ndarray, p=1, n0=4, winfunc=1):
    mfcc_f = mfcc_power(d.astype(float), winfunc=winfunc)
    c_bar = np.mean(mfcc_f[:, :n0//2]+mfcc_f[:, -n0//2:], axis=1)/2
    n = mfcc_f.shape[1]
    d = np.zeros([n])
    for t in range(n0, n):
        c_t = mfcc_f[:, t]
        c_bar = p*c_bar + (1-p)*c_t
        d[t] = distance(c_t, c_bar)
    return d


def endpoint_detect(d: np.ndarray, p=1, n1=20, winfunc=1):
    distances = cal_mfcc_distances(d, p, winfunc=winfunc)
    d_min = np.mean(distances[:int(n1/2)]+distances[-int(n1/2):])/2
    d_max = np.max(distances)
    h_l = d_min + (d_max - d_min)*0.1
    h_h = d_min + (d_max - d_min)*0.15
    starts = []
    ends = []
    tmp = -1
    hold = False
    for i in range(len(distances)):
        if distances[i] < h_l:
            if hold:
                ends.append(i)
                hold = False
            else:
                tmp = i
        elif distances[i] > h_h and not hold:
            starts.append(tmp)
            hold = True

    return np.array(starts), np.array(ends)


def frame2index(f):
    return f*256

if __name__ == '__main__':
    import VAD
    import utils
    import matplotlib.pyplot as plt

    wave_data, params = VAD.readWav("./dataset/unzip/ren3/" + '5' + ".wav")
    wave_data = utils.resample(wave_data, params[2], 16000).reshape(-1)
    plt.plot(wave_data.reshape(-1))
    plt.savefig("test.png")
    plt.close()
    starts, ends = endpoint_detect(wave_data, p=1, n1=20)
    starts = frame2index(starts)
    ends = frame2index(ends)
    plt.plot(wave_data)
    for i in range(starts.shape[0]):
        plt.axvspan(starts[i], ends[i], alpha=0.3)
    # plt.savefig("test2.png")

