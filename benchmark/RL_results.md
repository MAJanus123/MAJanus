# 第一版

### 系统设计：

#### 		奖励函数: 0.1 - latency

#### 		动作空间：卸载率

####         状态：latency

#### 		系统配置：1对1

#### 		目标：通过卸载率最小化系统延迟

#### 		智能体：运行在cloud端

#### 		step间隔：10s

### 系统性能

​		

| 带宽                         | 总体平均延迟(s/帧) | 总体平均奖励 | 收敛时延迟(s/帧) | 收敛时action |
| ---------------------------- | ------------------ | ------------ | ---------------- | ------------ |
| 5G(无限制)                   | 0.0283             | 0.0717       | 0.0275           | 0.99         |
| 4G(带宽8mbit 1mbit 延迟40ms) | 0.03555            | 0.0644       | 0.03             | 0.93         |
| 3.5G(带宽4Mb, 延迟60ms)      | 0.03580            | 0.0675       | 0.0323           | 0.99         |

#### 

#### 3.5G

<img src="C:\Users\12865\AppData\Roaming\Typora\typora-user-images\image-20230227221141405.png" alt="image-20230227221141405" style="zoom:50%;" />

#### 4G

​                                       	<img src="C:\Users\12865\AppData\Roaming\Typora\typora-user-images\image-20230227221504724.png" alt="image-20230227221504724" style="zoom: 50%;" />

#### 5G

####                       <img src="C:\Users\12865\AppData\Roaming\Typora\typora-user-images\image-20230227221559357.png" alt="image-20230227221559357" style="zoom:50%;" />           																			



# 第二版

### 	系统设计

#### 				奖励函数: <img src="C:\Users\12865\AppData\Roaming\Typora\typora-user-images\image-20230227193830243.png" alt="image-20230227193830243" style="zoom:50%;" />

##### 									p为acc, d为一个chunk的latency, w为系数暂定为0.7, m为在queue中的等待时间，T 为一个固定值1s，若m>T,给一个大的负反馈F:-1

​	

#### 				动作空间：卸载率   模型切换  resolution  

##### 	边端模型初始为s，resolution初始为360P，卸载率为0.8

####         		状态：latency acc  queue

#### 				系统配置：1对1

#### 				目标：通过卸载率、模型和revolution切换 最小化系统延迟

#### 				智能体：运行在cloud端

#### 				step间隔：10s
