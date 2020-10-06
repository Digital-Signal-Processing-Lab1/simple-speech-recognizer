# -*- coding: utf-8 -*-
import struct
import wave
import numpy as np
import matplotlib.pyplot as plt

# 将record.wav输入信号 采样 切割 转化成若干processedi.pcm文件

# 通过高门限的一定是需要截取的音，但是只用高门限，会漏掉开始的清音
# 如果先用低门限，噪音可能不会被滤除掉
# 先用高门限确定是不是噪音，再用低门限确定语音开始

def sgn(x):
    if x >= 0: return 1
    else: return -1

def calEnergy(frames, N):
    """ 返回每一帧的短时能量energy
        frames: 帧信号矩阵
        N: 一帧采样点个数
    """
    energy = []
    energy = np.sum(frames**2, axis=1)  # 计算帧信号矩阵每一行的平方和
    return energy

def calAmplitude(frames, N):
    """ 返回每一帧的短时幅度amplitude
        frames: 帧信号矩阵
        N: 一帧采样点个数
    """
    amplitude = []
    amplitude = np.sum(np.abs(frames), axis=1)  # 计算帧信号矩阵每一行的绝对值和
    return amplitude

def calZeroCrossingRate(frames, N):
    """ 返回每一帧的短时过零率zerocrossrate
        frames: 帧信号矩阵
        N: 一帧采样点个数
    """
    zerocrossingrate = []
    zerocrossingrate = np.sum(np.abs(frames[:, 1:N-1]-frames[:, 0:N-2]), axis=1)
    return zerocrossingrate
 
def detectEndPoint(wave_data, energy, zerocrossingrate):
    """ 利用短时能量/短时幅度，短时过零率，使用双门限法进行端点检测
        返回端点对应的帧序号endpoint0
        wave_data: 向量存储的语音信号
        energy: 一帧采样点个数
    """

    # [TODO: 继续调试TH，TL，T0的值]
    TH = np.mean(energy)                                    # 较高能量门限
    TL = np.mean(energy[:5]) * 0.999 + TH * 0.001           # 较低能量门限
    T0 = np.mean(zerocrossingrate[:5]) * 0.999 + TL * 0.001 # 过零率门限
    endpointH = []  # 存储高能量门限 端点帧序号
    endpointL = []  # 存储低能量门限 端点帧序号
    endpoint0 = []  # 存储过零率门限 端点帧序号

    # 先利用较高能量门限 TH 筛选语音段
    flag = 0
    for i in range(len(energy)):
        # 左端点判断
        if flag == 0 and energy[i] >= TH:
            endpointH.append(i)
            flag = 1

        # 右端点判断
        if flag == 1 and energy[i] < TH:
            endpointH.append(i)
            flag = 0
    
    X = np.arange(len(energy)).reshape(len(energy), 1)
    Y = np.array(energy).reshape(len(energy), 1)

    plt.figure(2)
    plt.subplot(3, 1, 1)
    plt.plot(X, Y)
    plt.title("energy")
    for i in range(len(endpointH)):
        plt.axvline(x=endpointH[i], ymin=0, ymax=1, color="red")

    # 再利用较低能量门限 TL 扩展语音段
    for j in range(len(endpointH)):
        i = endpointH[j]

        # 对右端点向右搜索
        if j % 2 == 1:
            while i < len(energy) and energy[i] >= TL:
                i = i + 1
            endpointL.append(i)

        # 对左端点向左搜索
        else:
            while i > 0 and energy[i] >= TL:
                i = i - 1
            endpointL.append(i)
    
    X = np.arange(len(energy)).reshape(len(energy), 1)
    Y = np.array(energy).reshape(len(energy), 1)
    plt.subplot(3, 1, 2)
    plt.plot(X, Y)
    plt.title("energy")
    for i in range(len(endpointL)):
        plt.axvline(x=endpointL[i], ymin=0, ymax=1, color="red")

    # 最后利用过零率门限 T0 得到最终语音段
    for j in range(len(endpointL)) :
        i = endpointL[j]

        # 对右端点向右搜索
        if j % 2 == 1:
            while i < len(zerocrossingrate) and zerocrossingrate[i] >= T0:
                i = i + 1
            endpoint0.append(i)

        # 对左端点向左搜索
        else:
            while i > 0 and zerocrossingrate[i] >= T0:
                i = i - 1
            endpoint0.append(i)

    X = np.arange(len(zerocrossingrate)).reshape(len(zerocrossingrate), 1)
    Y = np.array(zerocrossingrate).reshape(len(zerocrossingrate), 1)
    plt.subplot(3, 1, 3)
    plt.plot(X, Y)
    plt.title("zerocrossingrate")
    for i in range(len(endpoint0)):
        plt.axvline(x=endpoint0[i], ymin=0, ymax=1, color="red")

    return endpoint0

def addWindow(wave_data, N, M, winfunc):
    """ 将音频信号转化为帧并加窗
        返回帧信号矩阵:维度(帧个数, N)以及帧数num_frame
        wave_data: 待处理语音信号
        N: 一帧采样点个数
        M: 帧移（帧交叠间隔）
        winfunc: 加窗函数
    """
    wav_length = len(wave_data)     # 音频信号总长度
    inc = N - M                     # 相邻帧的间隔
    if wav_length <= N:             # 若信号长度小于帧长度，则帧数num_frame=1
        num_frame = 1
    else:
        num_frame = int(np.ceil((1.0*wav_length-N)/inc + 1))
    pad_length = int((num_frame-1)*inc+N)               # 所有帧加起来铺平后长度
    zeros = np.zeros((pad_length-wav_length, 1))        # 不够的长度用0填补
    pad_wavedata = np.concatenate((wave_data, zeros))   # 填补后的信号
    indices = np.tile(np.arange(0, N), (num_frame, 1)) + \
              np.tile(np.arange(0, num_frame*inc, inc), (N, 1)).T   # 用来对pad_wavedata进行抽取，得到num_frame*N的矩阵
    frames = pad_wavedata[indices].reshape(num_frame, N)            # 得到帧信号矩阵
    window = np.tile(winfunc, (num_frame, 1))
    return window * frames, num_frame                   # 加窗

def readWav(filename):
    """ 读取音频信号并转化为向量
        返回向量存储的语音信号wave_data及参数信息params
    """

    # 读入wav文件
    f = wave.open(filename, "rb")
    params = f.getparams()  # nchannels: 声道数, sampwidth: 量化位数, framerate: 采样频率, nframes: 采样点数
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = f.readframes(nframes)

    # 转成二字节数组形式（每个采样点占两个字节）
    wave_data = np.fromstring(str_data, dtype = np.short)
    wave_data = np.reshape(wave_data, [nframes, nchannels]) # 转化为向量形式
    print("采样点数目：" + str(len(wave_data)))     #输出应为采样点数目
    f.close()

    # 画图
    plt.figure(1)
    X = np.arange(len(wave_data)).reshape(len(wave_data), 1)
    Y = np.array(wave_data).reshape(len(wave_data), 1)
    plt.title("wave_data")
    plt.plot(X, Y)

    return wave_data, params

def writeWav(filename, wave_data, endpoint, params, N, M):
    """ 将切割好的语音信号输出
        生成多个切割好的wav文件
    """

    # 输出为 wav 格式
    i = 0
    j = 1
    inc = N - M                     # 相邻帧的间隔
    nchannels, sampwidth, framerate = params[:3]
    while i < len(endpoint):
        with wave.open(filename + str(j) + ".wav", "wb") as out_wave:
            comptype = "NONE"
            compname = "not compressed"
            nframes = (endpoint[i+1] - endpoint[i]) * inc
            out_wave.setparams((nchannels, sampwidth, framerate, nframes, comptype, compname))
            for v in wave_data[endpoint[i] * inc : endpoint[i+1] * inc]:
                out_wave.writeframes(struct.pack("h", int(v)))
            out_wave.close()
        j = j + 1
        i = i + 2

    # 画图
    plt.figure(3)
    X = np.arange(len(wave_data)).reshape(len(wave_data), 1)
    Y = np.array(wave_data).reshape(len(wave_data), 1)
    plt.plot(X, Y)
    for i in range(len(endpoint)):
        plt.axvline(x=endpoint[i]*inc, ymin=0, ymax=1, color="red")
    plt.title("processed_data")
    plt.show()


if __name__=="__main__":
    import os
    import sys
    import struct
    import scipy.signal as signal

    # 读取音频信号
    wave_data, params = readWav("./record.wav")

    # 语音信号分帧加窗
    N = 256         # 一帧时间 = N / framerate, 得 N 的范围: 160-480, 取最近2的整数次方 256
    M = 128         # M 的范围应在 N 的 0-1/2
    winfunc = signal.windows.hamming(N)     # 汉明窗
    # winfunc = signal.windows.hanning(N)     # 海宁窗
    # winfunc = 1                             # 矩形窗
    frames, num_frame = addWindow(wave_data, N, M, winfunc)

    # 时域特征值计算
    energy = calEnergy(frames, N).reshape(1, num_frame)
    amplitude = calAmplitude(frames, N).reshape(1, num_frame)
    zerocrossingrate = calZeroCrossingRate(frames, N).reshape(1, num_frame)

    # 端点检测
    endpoint = detectEndPoint(wave_data, energy[0], zerocrossingrate[0])    # 利用短时能量
    # endpoint = detectEndPoint(wave_data, amplitude[0], zerocrossingrate[0]) # 利用短时幅度
    print(endpoint)

    # 输出为 wav 格式
    # [TODO: 规范化输出格式利于后续分类工作]
    writeWav("./processed", wave_data, endpoint, params, N, M)