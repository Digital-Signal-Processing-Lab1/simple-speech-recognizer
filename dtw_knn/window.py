import tkinter
import ctypes
import dtw_predictor as predictor
import VAD
import utils
import time
import mfcc_detect as md
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

class Window:
    def __init__(self):
        winmm = ctypes.windll.winmm

        root = tkinter.Tk()
        root.title("win")
        root.geometry('400x300+300+150')
        root.resizable(False, False)
        self.target_fs =16000

        self.btn_start_record = tkinter.Button(root, text='start audio', command=self.start_record)
        # self.btn_start_record.place(x=10, y=10, width=80, height=20)
        self.btn_stop_record = tkinter.Button(root, text='stop audio', command=self.stop_record)
        # self.btn_stop_record.place(x=190, y=10, width=80, height=20)
        self.btn_play_audio = tkinter.Button(root, text='play audio', command=self.play_audio)
        self.btn_start_record.pack()
        self.btn_stop_record.pack()
        self.btn_play_audio.pack()
        root.protocol('WM_DELETE_WINDOW', self.close_win)

        self.btn_predict_and_show = tkinter.Button(root, text='predict', command=self.predict_and_show)
        # self.btn_predict_and_show['state'] = 'disable'
        self.btn_predict_and_show.pack()

        self.result_var = tkinter.StringVar()
        self.result_var.set('None')
        result_label = tkinter.Label(root, textvariable=self.result_var)
        result_label.pack()

        self.error_message = tkinter.StringVar()
        self.error_message.set('None')
        error_message = tkinter.Label(root, textvariable=self.error_message)
        error_message.pack()
        self.winmm = winmm
        self.init_winmm()
        self.root = root
        self.filename = r'./.tmp.wav'

    def predict_and_show(self):
        pass

    def play_audio(self):
        self.winmm.mciSendStringW('open {} alias audio2'.format(self.filename), '', 0, 0)
        buf = ctypes.c_buffer(255)
        self.winmm.mciSendStringA(str('status audio2 length').encode(), buf, 254, 0)
        l = int(buf.value)
        self.winmm.mciSendStringW('play audio2 from {:d} to {:d}'.format(0, l), '', 0, 0)
        time.sleep(int(l/1000)+1)
        self.winmm.mciSendStringW('stop audio2', '', 0, 0)
        self.winmm.mciSendStringW('close audio2','', 0, 0)

    def init_winmm(self):
        self.winmm.mciSendStringW('set audio bitspersample 16'.encode(), '', 0, 0)
        self.winmm.mciSendStringW('set audio samplespersec {}'.format(self.target_fs), '', 0, 0)
        self.winmm.mciSendStringW('set audio channels 1', '', 0, 0)
        self.winmm.mciSendStringW('set audio format tag pcm', '', 0, 0)

    def set_result(self, result):
        self.result_var.set(result)

    def start_record(self):
        self.winmm.mciSendStringW('close audio', '', 0, 0)
        self.winmm.mciSendStringW('open new type waveaudio alias audio', '', 0, 0)
        self.init_winmm()
        self.winmm.mciSendStringW('record audio', '', 0, 0)
        self.btn_start_record['state'] = 'disable'
        self.btn_stop_record['state'] = 'normal'

    def stop_record(self):
        self.winmm.mciSendStringW('stop audio', '', 0, 0)
        self.winmm.mciSendStringW('save audio {}'.format(self.filename), '', 0, 0)
        self.winmm.mciSendStringW('close audio', '', 0, 0)
        self.btn_start_record['state'] = 'normal'
        self.btn_stop_record['state'] = 'disable'

    def close_win(self):
        if self.btn_stop_record['state'] == 'normal':
            self.stop_record()
        self.winmm.mciSendStringW('close audio', '', 0, 0)
        self.root.destroy()

    def mainloop(self):
        self.root.mainloop()


class TestWindow(Window):
    def __init__(self):
        super(TestWindow, self).__init__()
        self.wave_data = None
        self.error = False

    def set_predictor(self, predictor):
        """
        predictor must be a function which accept array data
        """
        self.predictor = predictor

    def process_data(self):
        wave_data, params = VAD.readWav(self.filename)
        source_fs = params[2]
        if source_fs != self.target_fs:
            wave_data = utils.resample(wave_data, source_fs, self.target_fs)
        wave_data = wave_data.reshape(-1)
        starts, ends = md.endpoint_detect(wave_data.astype(float), p=1, n1=20, winfunc=self.predictor.detect_win)
        if starts.shape[0] != 1 or  ends.shape[0] != 1:
            self.error_message.set('Error while detect the endpoints')
            self.error = True
            self.wave_data = wave_data.reshape([-1])
        else:
            starts = md.frame2index(starts)
            ends = md.frame2index(ends)
            self.wave_data = np.array([wave_data[s:e] for s, e in zip(starts, ends)])
            self.error = False
            self.error_message.set('None')
        plt.plot(wave_data)
        for i in starts:
            plt.axvline(i)
        for i in ends:
            plt.axvline(i)
        plt.show()

    def predict_and_show(self):
        self.process_data()
        result = self.predictor.predict(self.wave_data)
        self.result_var.set("predict: {}".format(result))

    def stop_record(self):
        self.winmm.mciSendStringW('stop audio', '', 0, 0)
        self.winmm.mciSendStringW('save audio {}'.format(self.filename), '', 0, 0)
        self.winmm.mciSendStringW('close audio', '', 0, 0)
        self.btn_start_record['state'] = 'normal'
        self.btn_stop_record['state'] = 'disable'
        self.btn_predict_and_show['state'] = 'normal'


if __name__ == '__main__':
    w = TestWindow()
    predictor = predictor.Predictor("../dtw_knn/data.pkl")
    w.set_predictor(predictor)
    w.mainloop()
