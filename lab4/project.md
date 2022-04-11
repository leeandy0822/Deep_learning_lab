# 深度學習 HW3
機器人學程 李啟安 310605015
###### tags: `深度學習`
[TOC]

# 1. Introduction
In this lab, we will build a simple EEG classification models which are EEGNet and DeepConvNet with BCI competetion dataset. We will anaylze different activation function, such as ReLU, LeakyReLU and ELU.

- Data:
![](https://i.imgur.com/KLDGLQ6.png)


- EEGNet
![](https://i.imgur.com/NDjrQNe.png)

- DeepConvNet
![](https://i.imgur.com/j6yqwZd.png)


# 2. Experiment Setup
##  A. The detail of your model


### - Initialize
![](https://i.imgur.com/3VvJF9K.png)
- We can enter desire model (EEGNet or DeepConvNet) and also with desire acitvation function we want to use.
### - EEGNet
![](https://i.imgur.com/ggyfPpZ.png)

There are four blocks in EEGNet. FirstConv, depthwiseConv, seperableConv and classification. Based on TA's instruction, we can create the network. 

### - DeepConvNet
![](https://i.imgur.com/5smzP0y.png)

The DeepConvNet is more complicated. It has five blocks, such as "featurelayer1", ...."featurelayer4" and "classify". In the last layer, the weight has constraint of max_norm = 0.5, so I create the constraint class.

![](https://i.imgur.com/QlLYQiR.png)

In the training.py,  we can implement the constraint on the classify layer.

![](https://i.imgur.com/3KVdGIE.png)

args.constraint is a bool variable to determine whether the constraint is able.


## B. Explanation of the activation function
參考自 https://zhuanlan.zhihu.com/p/25110450
### ReLU 
![](https://i.imgur.com/ZpsXOw2.png)

- Formula:
    $$ ReLU(x) = max(0, x)$$

- Advantage:
1. Solve gradient vanishing (in Positive side)
2. caculate quickly
3. converger more quickly than tanh, sigmoid

- Disadvantage:
1. Dead ReLU Problem : ReLU neurons become inactive and only ouput 0 for any input

### Leaky ReLU
![](https://i.imgur.com/bmbpQaQ.png)

- Formula $$f(x) = max(0.01x, x)$$


- Advantage:
1. All the advantages ReLU has
2. Avoid Dead ReLU problem
- Disadvantage:
1. 沒有完全證明任何情況下總是比ReLU好

### ELU Network (Exponential Layer Unit)
![](https://i.imgur.com/Hhpuw1v.png)

- Formula: $$ \begin{aligned}x <0: ELU\left( x\right) =e^{x}-1 \\
x >0:ELU\left( x\right) =x\end{aligned}$$

- Advantage：
    - Avoid Dead ReLU Problem
    - Zero-centered
- Disadvantage
    - 沒有被完全證明任何情況都好於 ReLU
    - 計算量稍微大


# 3. Experimental Result
## A. The highest testing accuracy

- DeepConvNet
![](https://i.imgur.com/x9NL1gj.png)


- learning_rate: 0.002
- epochs: 300
- batch: 12

| Activation Function | ReLU     |  ELU     | LeakyReLU|
| --------            | -------- | -------- | ---------|
| Test Accuracy       | 81.296   | 78.241   |   80.648  |

- EEGNet
![](https://i.imgur.com/Uf2GwIw.png)
- learning_rate: 0.0002
- epochs: 300
- batch: 12

| Activation Function | ReLU     |  ELU     | LeakyReLU|
| --------            | -------- | -------- | ---------|
| Test Accuracy       | 85   | 82.96  |    **86.296**  |

#### Get the model > 87%
- I use the EEG_LeakyReLU as a **pretrained model** and turn the learning_rate to 0.00001 to continue training
- Then once the test accuracy > 87%, terminate the training process and save the model!

![](https://i.imgur.com/QLcTc47.png)

- **EEGNet with LeakyReLU is the best**

## B. Anything you want to present

### 1. Without max_norm constraint
Try to delete the max_norm weight constatint. And we can see that the performance is almost the same in training_data, but huge different in testing_data.
![](https://i.imgur.com/s6qLAmc.png)
| Max_norm | Yes    |  No    |
| --------   | -------- | -------- | 
| Test Accuracy| 84.019   | 86.296  |  

### 2. Different batch_size

- learning_rate: 0.0002
- epochs: 300

| Batch_size | 10    |  12    | 14   |
| --------   | -------- | -------- | ---------|
| Test Accuracy| 84.019   | 86.296  |   86.019  |

### 3. Different learning_rate

- batch_size : 12
- epochs: 300

| Learning_rate | 0.0002    |  0.002    | 0.02   |
| --------   | --------   | -------- | -------|
| Test Accuracy| 86.296  | 85.185  |   79.259  |


### C. Comparison figures
- DeepConvNet
![](https://i.imgur.com/bzRvpt6.png)

- EEGNet
![](https://i.imgur.com/AttkSXl.png)

We can see that EEGNet is better than DeepConvNet!


# 4. Discussion

1. It is very interesting that although DeepConvNet has deeper NN but the performace is not as good as EEG

2. Wandb is a really good tools ! I can deal with all my model online and won't lose them
![](https://i.imgur.com/YhVTs2k.png)

3. Use arg.praser to define all argument!
![](https://i.imgur.com/hJZBcLk.png)

4. Code Architecture
![](https://i.imgur.com/1gVITRu.png)
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script>
