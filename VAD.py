# -*- coding: utf-8 -*-
import wave
import numpy as np
import matplotlib.pyplot as plt

# 将record.wav输入信号 采样 切割 转化成若干processedi.pcm文件

# 通过高门限的一定是需要截取的音，但是只用高门限，会漏掉开始的清音
# 如果先用低门限，噪音可能不会被滤除掉
# 先用高门限确定是不是噪音，再用低门限确定语音开始

# 符号函数定义
def sgn(x):
    if x >= 0: return 1
    else: return -1

# 计算每一帧的短时能量 一帧采样点个数为N
def calEnergy(wave_data, N):
    energy = []
    sum = 0
    # [TODO: 本代码未考虑帧与帧之间交叠，对以下两个函数也是如此]
    for i in range(len(wave_data)):
        sum = sum + int(wave_data[i])**2
        if (i + 1) % N == 0:
            energy.append(sum)
            sum = 0
        elif i == len(wave_data) - 1:   #最后一帧
            energy.append(sum)
    return energy

# 计算每一帧的短时幅度 一帧采样点个数为N
def calAmplitude(wave_data, N):
    amplitude = []
    sum = 0
    for i in range(len(wave_data)):
        sum = sum + np.abs(int(wave_data[i]))
        if (i + 1) % N == 0:
            amplitude.append(sum)
            sum = 0
        elif i == len(wave_data) - 1:   #最后一帧
            amplitude.append(sum)
    return amplitude

# 计算每一帧的短时过零率 一帧采样点个数为N
def calZeroCrossingRate(wave_data, N):
    zerocrossingrate = []
    sum = 0
    for i in range(len(wave_data)):
        if i % N == 0:                  #从xn(1)计算到xn(N-1)
            continue
        sum = sum + np.abs(sgn(wave_data[i]) - sgn(wave_data[i - 1]))
        if (i + 1) % N == 0:
            zerocrossingrate.append(float(sum)/(2*N))
            sum = 0
        elif i == len(wave_data) - 1:   #最后一帧
            zerocrossingrate.append(float(sum)/(2*N))
    return zerocrossingrate

# 利用短时能量/短时幅度，短时过零率，使用双门限法进行端点检测
def detectEndPoint(wave_data, energy, zerocrossingrate):
    # [TODO: 确定TH，TL，T0的值]
    TH = np.mean(energy) / 4                  # 较高能量门限
    TL = (np.mean(energy[:5])+TH/300)       # 较低能量门限
    T0 = np.mean(zerocrossingrate[:5])      # 过零率门限
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
    plt.subplot(1, 3, 1)
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
    plt.subplot(1, 3, 2)
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
    plt.subplot(1, 3, 3)
    plt.plot(X, Y)
    plt.title("zerocrossingrate")
    for i in range(len(endpoint0)):
        plt.axvline(x=endpoint0[i], ymin=0, ymax=1, color="red")

    return endpoint0

if __name__=="__main__":
    import os
    import sys
    f = wave.open("./record.wav", "rb")
    str_data = f.readframes(f.getnframes())

    # 转成二字节数组形式（每个采样点占两个字节）
    wave_data = np.fromstring(str_data, dtype = np.short)
    print("采样点数目：" + str(len(wave_data)))     #输出应为采样点数目
    f.close()

    plt.figure(1)
    X = np.arange(len(wave_data)).reshape(len(wave_data), 1)
    Y = np.array(wave_data).reshape(len(wave_data), 1)
    plt.title("wave_data")
    plt.plot(X, Y)

    # 端点检测
    N = 256
    M = None
    energy = calEnergy(wave_data, N)
    amplitude = calAmplitude(wave_data, N)
    zerocrossingrate = calZeroCrossingRate(wave_data, N)

    endpoint = detectEndPoint(wave_data, energy, zerocrossingrate)

    # 输出为 pcm 格式
    # [TODO: 本代码未考虑帧与帧之间交叠]
    i = 0
    j = 1
    while i < len(endpoint):
        with open("./processed" + str(j) + ".pcm", "wb") as f:
            for num in wave_data[endpoint[i] * N : endpoint[i+1] * N]:
                f.write(num)
        j = j + 1
        i = i + 2

    plt.figure(3)
    X = np.arange(len(wave_data)).reshape(len(wave_data), 1)
    Y = np.array(wave_data).reshape(len(wave_data), 1)
    plt.plot(X, Y)
    for i in range(len(endpoint)):
        plt.axvline(x=endpoint[i]*N, ymin=0, ymax=1, color="red")
    plt.title("processed_data")
    plt.show()