import librosa.sequence as ls
import librosa.feature as lf
import numpy as np
import pandas as pd
from librosa.feature import mfcc
import utils


def normalize(d):
    return (d-np.mean(d))/np.std(d)


N = 512
def get_features(wave_data: pd.Series, n_mfcc, window):
    """
    feature
    """
    x = wave_data.apply(lambda d: (d-np.mean(d))/(np.std(d)))
    # x = wave_data
    x, max_length = utils.padding_to_max(x)
    features = []
    for i in range(x.shape[0]):
        t1 = mfcc(x[i], sr=16000, n_mfcc=n_mfcc, n_fft=N, hop_length=256, window=window)
        t2 = utils.diff(t1, axis=0)
        t3 = utils.diff(t1, axis=0, delta=2)
        t = np.concatenate([t1, t2, t3], axis=0).flatten()
        features.append(t)
    return np.array(features), max_length


def get_dist_func(k):
    def distance(X: np.ndarray, Y: np.ndarray):
            d = ls.dtw(X.reshape([k, -1]), Y.reshape([k, -1]), backtrack=False)
            return np.min(d[-1])
    return distance


def distance(X: np.ndarray, Y: np.ndarray, k):
        d = ls.dtw(X.reshape([k, -1]), Y.reshape([k, -1]), backtrack=False)
        return np.min(d[-1])

def downsampling(array, dim):
    array_len = len(array)
    if array_len >= dim:
        inc = int(array_len/dim)
        ret = array[:array_len:inc]
        return ret
    else : return -1


# def get_feature(x, N, M):
#     frames, num_frame = VAD.addWindow(x.reshape(-1, 1), N, M, 1)
#     energy = normalize(VAD.calEnergy(frames, N)).reshape(1, num_frame)
#     amplitude = normalize(VAD.calAmplitude(frames, N)).reshape(1, num_frame)
#     zerocrossingrate = normalize(VAD.calZeroCrossingRate(frames, N)).reshape(1, num_frame)
#     return np.concatenate([energy, amplitude, zerocrossingrate], axis=0)


if __name__ == '__main__':
    import pickle
    import utils
    import matplotlib.pyplot as plt
    import VAD
    from librosa import display

    N, M = 512, 256
    with open("../dataset/processed/rect.pkl", "rb") as f:
        df = pickle.load(f)
        df = df[~df.has_noisy]
        features, max_length = get_features(df.wave_data, 20, 1)
        x = features[1].reshape([60, -1])
        y = features[10].reshape([60, -1])
        D, wp = ls.dtw(x, y, subseq=True)
        fig, ax = plt.subplots(1,1)
        img = display.specshow(D, ax=ax)
        fig.colorbar(img, ax=ax)
        ax.plot(wp[:, 1], wp[:, 0], color='y')
        plt.tight_layout()
        plt.savefig("./dtw_0_5.png")
        plt.close()

        label = df.content
        for k in range(10):
            print(k)
            ones = features[label==k]
            others = features[label!=k]
            cross_dtw = np.zeros([ones.shape[0],others.shape[0]])
            self_dtw = np.zeros([ones.shape[0], ones.shape[0]])
            for i in range(len(ones)):
                print(i)
                x = ones[i].reshape([60, -1])
                for j in range(len(others)):
                    y = others[j].reshape([60, -1])
                    D, _ = ls.dtw(x, y)
                    cross_dtw[i, j] = np.min(D[-1])

                for j in range(len(ones)):
                    if i==j:
                        continue
                    y = ones[j].reshape([60, -1])
                    D, _ = ls.dtw(x, y)
                    self_dtw[i, j] = np.min(D[-1])

            t1 = plt.hist(cross_dtw.reshape(-1), density=True, alpha=0.5, bins=100, label='others')
            t2 = plt.hist(self_dtw.reshape(-1), alpha=0.5, density=True, bins=100, label='self')
            plt.legend()
            plt.tight_layout()
            plt.savefig("dtw_{}.png".format(k))
            plt.close()


