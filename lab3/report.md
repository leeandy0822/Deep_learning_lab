# 深度學習 HW2
機器人學程 李啟安 310605015
###### tags: `深度學習`

## Q1. A plot shows episode scores of at least 100,000 training episodes

![](https://i.imgur.com/zsNQPv6.png)
- In episodes 208000, we can get the best 2048 score for 98.6%

<img src="https://i.imgur.com/2Ci6uqj.png"  width="450" height="400" />

- Score plot for 250000 episodes (trained for 18 hours)
    - Change the learning_rate from 0.1 -> 0.01 after 200000 episodes

## Q2. Describe the implementation and the usage of n-tuple network
Every single position can have the probability to be $[null, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, ...]$. Total will be at least $12^{16}$ combinations. It is impossible to store all these possiblities in computer.


Compared to recording all the state of boards, **n-tuple network can use multiple small tuples for caculation** because a tuple can present the characteristic of board. It is also easy for caculations and saves a lot of spaces.

---

![](https://i.imgur.com/Lw3VzN1.png)

可以看到我們宣告了一個 6-tuple 然後根據鏡像以及旋轉，可以得到八種不同的結果。六個位置可以對應到一組 index，然後可以從 network中找到對應的value，例如： **fig.3 右側的網路就是以第三種版面為例子，所以可以看到第三種版面為 1 ，其餘都是 0，我們即可拿到第三種版面的權重，相乘就可以得到估計值**，因此八種版面估計值加總，就可以成為此版面的價值了！

我們可以在多取一點feature，助教的sample code 有四種，那我又自己加上三種，總共有七種 feature。

fig.3 新增 窗形 2 x 3 feature
![](https://i.imgur.com/PWmKcqU.png)

fig.4 黃色與藍色分別代表窗形feature的兩種情況。

<img src="https://i.imgur.com/ySV1IMC.jpg"  width="200" height="200" />


## Q3. Explain the mechanism of TD(0).
TD(0) is special case of TD($\lambda$)，it will look one step ahead. Unlike the MC, we will get the immediate reward plus the discount estimate value of 1 step ahead. 

<img src="https://i.imgur.com/syTltum.png"  width="300" height="30" />

In the project, the discount fector is set to 1, We will keep update our value function in each state. The transition is $s \rightarrow s^{''}$. We wait until arrive the next time step $s''$ and then **combine immediate reward $r$ in state $s$ and prediction $V(s'')$ to update our $V(s)$.**

<img src="https://i.imgur.com/Nx4gQyI.png"  width="300" height="30" />

## Q4. Explain the TD-backup diagram of V(after-state).
![](https://i.imgur.com/zNbYA4M.png)

The after-state method only cares about the after state. We want to update $V(s')$ by selecting $a_{next}$ and chooseing TD target as $r_{next} + V(s'_{next})$. Then we can get TD error by $(r_{next} + V(s'_{next}) - V(s'))$. Finally update $V(s')$ by multiplying the learning_rate $\alpha$.

## Q5. Explain the action selection of V(after-state) in a diagram.
To select the best action, it will caculate all possible actions and select the action that can go to state with highest value function.

<img src="https://i.imgur.com/H4vQn1P.png"  width="300" height="80" />

## Q6. Explain the TD-backup diagram of V(state).
![](https://i.imgur.com/iXMu6mS.png)

The before-state method cares about the state "after the transition" or "before the action". We want to update $V(s)$ by selecting $a$ and chooseing TD target as $r + V(s'')$. Then we can get TD error by $(r + V(s'') - V(s))$. Finally update $V(s)$ by multiplying the learning_rate $\alpha$.

## Q7. Explain the action selection of V(state) in a diagram.
Action selection will take all possible action (up, down, left, right) and go through all the possible transition to next state $s''$. Then we can choose the best action with the highest state-action value function.

<img src="https://i.imgur.com/GwLnADe.png"  width="300" height="110" />

## Q8. Describe your implementation in detail.
In this lab, I change the following function : 
- select_best_move : 
    - To select best action, build a for loop that will take all actions and build an inner for loop to caculate the expectation of choosing that action
    1. define a vector to store the empty position on board
![](https://i.imgur.com/DWpQ38Z.png)

    2. Run a for loop to run through all possible action. **Choose one possible action and run through all possible transition to predict the expectation of choosing that action**. (Remember: the probability of popup 4-tile is 10% but 2-tile is 90%)
![](https://i.imgur.com/V9kL3oZ.png)

    
    3. Return the best action with the highest value function
![](https://i.imgur.com/FlI1nVf.png)

- update_episode : 

<img src="https://i.imgur.com/bFSv9m1.png"  width="300" height="50" />


![](https://i.imgur.com/l7FdUeW.png)

1. Caculate the value function of terminal state
2. Loop through the path by updating the value function
3. Then you can get the new value function of the whole path! 

- main :
    - 跑程式
    - 自己加上新的 feature
    
<img src="https://i.imgur.com/Mgw1ygr.png"  width="300" height="300" />
<img src="https://i.imgur.com/0eiETkv.png"  width="300" height="300" />



## Q9. Other diccussions or improvement
1. We can adjust the learning rate after 100000 episodes, turn 0.1 to 0.01 can get a better performance

2. To draw the score and episode graph, define a function to automatically save score in txt file so we can plot from python or matlab!
![](https://i.imgur.com/46RmHYi.png)


<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script>
