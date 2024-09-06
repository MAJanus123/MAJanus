import re

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('model360P_n.txt', header=None, sep='\s+', names=['resolution', 'model', 'bitrate', 'modelsize', 'acc'])
all_acc = []
acc = []
old = '1280'
for i in range(len(df['acc'])):
    resolution = df['resolution'][i][11:].split('x')[0]
    if resolution != old:
        all_acc.append(acc)
        acc = []
        old = resolution
    acc.append(round(float(df['acc'][i][7:]), 4))
all_acc.append(acc)
print(all_acc)
x1 = [100,200,300,400,500,600,700,800,900,1000,1200,1500,1700,2000,2200,2500,2700,3000,3500,4000,4500,5000]
# x1 = [720, 480, 360, 270, 240, 144, 96]
# y1 = [data[4] for data in all_acc]
# y2 = [data[5] for data in all_acc]
# y3 = [data[6] for data in all_acc]
# y4 = [data[7] for data in all_acc]
# y5 = [data[8] for data in all_acc]
# y6 = [data[9] for data in all_acc]
# y7 = [data[11] for data in all_acc]
# y8 = [data[13] for data in all_acc]
# y9 = [data[15] for data in all_acc]
# y10 = [data[17] for data in all_acc]
# y11 = [data[18] for data in all_acc]
# y12 = [data[19] for data in all_acc]
# y13 = [data[20] for data in all_acc]
# y14 = [data[21] for data in all_acc]
y1 = all_acc[0]
y2 = all_acc[1]
y3 = all_acc[2]
y4 = all_acc[3]
y5 = all_acc[4]
y6 = all_acc[5]
y7 = all_acc[6]
# 创建画布和子图
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(14, 10), gridspec_kw={'hspace': 0.5})
# 在子图1中绘制折线图
axs[0,0].plot(x1, y1)
axs[0,0].set_title('r:720P')

# 在子图2中绘制折线图
axs[0,1].plot(x1, y2)
axs[0,1].set_title('r:480P')

axs[0,2].plot(x1, y3)
axs[0,2].set_title('r:360P')

axs[1,0].plot(x1, y4)
axs[1,0].set_title('r:270P')

axs[1,1].plot(x1, y5)
axs[1,1].set_title('r:240P')

axs[1,2].plot(x1, y6)
axs[1,2].set_title('r:144P')

axs[2,0].plot(x1, y7)
axs[2,0].set_title('r:96P')

# axs[0,0].plot(x1, y1)
# axs[0,0].set_title('bitrate:500k')
#
# axs[0,1].plot(x1, y2)
# axs[0,1].set_title('bitrate:600k')
#
# axs[0,2].plot(x1, y3)
# axs[0,2].set_title('bitrate:700k')
#
# axs[1,0].plot(x1, y4)
# axs[1,0].set_title('bitrate:800k')
#
# axs[1,1].plot(x1, y5)
# axs[1,1].set_title('bitrate:900k')
#
# axs[1,2].plot(x1, y6)
# axs[1,2].set_title('bitrate:1000k')
#
# axs[2,0].plot(x1, y7)
# axs[2,0].set_title('bitrate:1500k')
#
# axs[2,1].plot(x1, y8)
# axs[2,1].set_title('bitrate:2000k')
#
# axs[2,2].plot(x1, y9)
# axs[2,2].set_title('bitrate:2500k')
#
# axs[3,0].plot(x1, y10)
# axs[3,0].set_title('bitrate:3000k')
#
# axs[3,1].plot(x1, y11)
# axs[3,1].set_title('bitrate:3500k')
#
# axs[3,2].plot(x1, y12)
# axs[3,2].set_title('bitrate:4000k')
#
# axs[4,0].plot(x1, y13)
# axs[4,0].set_title('bitrate:4500k')
#
# axs[4,1].plot(x1, y14)
# axs[4,1].set_title('bitrate:5000k')

# 显示图形
plt.show()
