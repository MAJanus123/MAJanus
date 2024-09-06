# DATASET
## BDD100K
### 下载链接：https://bdd-data.berkeley.edu/portal.html#download
数据集选择 MOT 2020 Images，进入链接进行下载。7个train，1个val，2test（未下载）

### 数据集描述：
BDD100k是一个采用720P分辨率行车记录仪拍摄的道路交通视频数据集，该数据集拥有白天、黑夜、晴天、雨天等各种交通场景，
可用于目标检测、对象追踪、语义分割、实例分割、车道标记、模仿学习等领域。 
本项目中采用MOT2020图片作为训练、评估、测试数据集，MOT2020是BDD100K视频的子集，从原先的30hHz采样到5Hz，用于目标检测相关的项目使用。
在本项目中，我们采用MOT2020中的8个类进行训练， 分别是'pedestrian'、'rider'、 'car'、 'truck'、 'bus'、 'motorcycle'、 'bicycle'、 
'other vehicle'。在数据集的划分上，我们从原先的1400个train视频中划出50个视频用于test，而200个val视频不变，
最终我们得到了一个拥有1350个训练视频、200个评估视频（训练时的评估）、50个测试视频的完整数据集。