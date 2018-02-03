
# Task 1: XOR


```python
# Import modules
from __future__ import print_function
import tensorflow as tf
import numpy as np
from numpy.random import shuffle
import time
import matplotlib.pyplot as plt

# Plot configurations
% matplotlib inline

# Notebook auto reloads code. (Ref: http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython)
% load_ext autoreload
% autoreload 2
```

## Task 1, Part 1: Backpropagation through time (BPTT)

**Question:** Consider a simple RNN network shown in the following figure, where __ _wi, wh, b, a, c_ __ are the scalar parameters of the network. The loss function is the **mean squared error (MSE)**. Given input (x0, x1) = (1, 0), ground truth (g1, g2) = (1, 1), h0 = 0, (wi, wh, b, a, c) = (1, 1, 1, 1, 1), compute __ _(dwi, dwh, db, da, dc)_ __, which are the gradients of loss with repect to 5 parameters __ _(wi, wh, b, a, c)_ __.

![bptt](./img/bptt.png)


```python
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))
```


```python
(x0, x1) = (1, 0)
(g1, g2) = (1, 1)
h0 = 0
(wi, wh, b, a, c) = (1, 1, 1, 1, 1)
h1 = sigmoid(wi*x0 + wh*h0 + b)
y1 = a*h1 + c
print(h1)
print(y1)
h2 = sigmoid(wi*x1 + wh*h1 + b)
y2 = a*h2 + c
print(h2)
print(y2)
L = 1/2 * ((y1-g1)**2 + (y2-g2)**2)
d1 = y1-g1
d2 = y2-g2
print(d1)
print(d2)
```

    0.880797077978
    1.88079707798
    0.867702653653
    1.86770265365
    0.880797077978
    0.867702653653


$\dfrac{\delta L}{\delta wi} = \dfrac{1}{2}\sum_{i=1}^n \dfrac{\delta Li}{\delta wi} = \dfrac{\delta L1}{\delta wi} + \dfrac{\delta L2}{\delta wi}$ 

$ \dfrac{\delta L1}{\delta wi}=\dfrac{\delta L1}{\delta y1}\dfrac{\delta y1}{\delta h1}\dfrac{\delta h1}{\delta wi} =  \dfrac{\delta }{\delta wi} (a*h1+c)= \dfrac{\delta }{\delta wi} (a*sigmoid(wi*x0+wh*h0+b)+c) = d1*a*x0*sigmoid(wi*x0+wh*h0+b)*(1-sigmoid(wi*x0+wh*h0+b)) = d1*a*x0*h1*(1-h1) = 0.092478043229829859$

$ \dfrac{\delta L2}{\delta wi}=\dfrac{\delta L2}{\delta y2}\dfrac{\delta y2}{\delta h2}\dfrac{\delta h2}{\delta wi}+\dfrac{\delta L2}{\delta y2}\dfrac{\delta y2}{\delta h2}\dfrac{\delta h2}{\delta h1}\dfrac{\delta h1}{\delta wi}= \dfrac{\delta }{\delta wi} (a*h2+c) = \dfrac{\delta }{\delta wi} (a*sigmoid(wi*x1+wh*h1+b)+c) =  d2*(a*x1*sigmoid(wi*x1+wh*h1+b)*(1-sigmoid(wi*x1+wh*h1+b)) + a*wh*sigmoid(wi*x1+wh*h1+b)*(1-sigmoid(wi*x1+wh*h1+b))*x0*sigmoid(wi*x0+wh*h0+b)*(1-sigmoid(wi*x0+wh*h0+b))) = d2*(a*x1*h2*(1-h2) + a*wh*h2*(1-h2)*x0*h1*(1-h1)) = 0.010458171296869908$

$\dfrac{\delta L}{\delta wi} = \dfrac{\delta L1}{\delta wi} + \dfrac{\delta L2}{\delta wi} = 0.10293621452669977$

$\dfrac{\delta L}{\delta wh} = \dfrac{1}{2}\sum_{i=1}^n \dfrac{\delta Li}{\delta wh} = \dfrac{\delta L1}{\delta wh} + \dfrac{\delta L2}{\delta wh}$

$ \dfrac{\delta L1}{\delta wh}=\dfrac{\delta L1}{\delta y1}\dfrac{\delta y1}{\delta h1}\dfrac{\delta h1}{\delta wh} =  \dfrac{\delta }{\delta wh} (a*h1+c)= \dfrac{\delta }{\delta wi} (a*sigmoid(wi*x0+wh*h0+b)+c) = d1*a*x0*sigmoid(wi*x0+wh*h0+b)*(1-sigmoid(wi*x0+wh*h0+b)) = d1*a*h0*h1*(1-h1) = 0$

$ \dfrac{\delta L2}{\delta wh}=\dfrac{\delta L2}{\delta y2}\dfrac{\delta y2}{\delta h2}\dfrac{\delta h2}{\delta wh}+\dfrac{\delta L2}{\delta y2}\dfrac{\delta y2}{\delta h2}\dfrac{\delta h2}{\delta h1}\dfrac{\delta h1}{\delta wh}= \dfrac{\delta }{\delta wh} (a*h2+c) = \dfrac{\delta }{\delta wi} (a*sigmoid(wi*x1+wh*h1+b)+c) =  d2*(a*x1*sigmoid(wi*x1+wh*h1+b)*(1-sigmoid(wi*x1+wh*h1+b)) + a*wh*sigmoid(wi*x1+wh*h1+b)*(1-sigmoid(wi*x1+wh*h1+b))*x0*sigmoid(wi*x0+wh*h0+b)*(1-sigmoid(wi*x0+wh*h0+b))) = d2*(a*h1*h2*(1-h2) + a*wh*h2*(1-h2)*h0*h1*(1-h1)) = 0.0877341857017$

$\dfrac{\delta L}{\delta wh} = \dfrac{\delta L1}{\delta wh} + \dfrac{\delta L2}{\delta wh} = 0.0877341857017$

$\dfrac{\delta L}{\delta b} = \dfrac{1}{2}\sum_{i=1}^n \dfrac{\delta Li}{\delta b} = \dfrac{\delta L1}{\delta b} + \dfrac{\delta L2}{\delta b}$

$ \dfrac{\delta L1}{\delta b}=\dfrac{\delta L1}{\delta y1}\dfrac{\delta y1}{\delta h1}\dfrac{\delta h1}{\delta b} =  \dfrac{\delta }{\delta b} (a*h1+c)= \dfrac{\delta }{\delta b} (a*sigmoid(wi*x0+wh*h0+b)+c) = d1*a*x0*sigmoid(wi*x0+wh*h0+b)*(1-sigmoid(wi*x0+wh*h0+b)) = d1*a*h1*(1-h1) = 0.0924780432298$

$ \dfrac{\delta L2}{\delta b}=\dfrac{\delta L2}{\delta y2}\dfrac{\delta y2}{\delta h2}\dfrac{\delta h2}{\delta b}+\dfrac{\delta L2}{\delta y2}\dfrac{\delta y2}{\delta h2}\dfrac{\delta h2}{\delta h1}\dfrac{\delta h1}{\delta b}= \dfrac{\delta }{\delta b} (a*h2+c) = \dfrac{\delta }{\delta b} (a*sigmoid(wi*x1+wh*h1+b)+c) =  d2*(a*x1*sigmoid(wi*x1+wh*h1+b)*(1-sigmoid(wi*x1+wh*h1+b)) + a*wh*sigmoid(wi*x1+wh*h1+b)*(1-sigmoid(wi*x1+wh*h1+b))*x0*sigmoid(wi*x0+wh*h0+b)*(1-sigmoid(wi*x0+wh*h0+b))) = d2*(a*h2*(1-h2) + a*h2*(1-h2)*wh*h1*(1-h1)) = 0.11006588787$

$\dfrac{\delta L}{\delta b} = \dfrac{\delta L1}{\delta b} + \dfrac{\delta L2}{\delta b} = 0.20254393109983646
$

$\dfrac{\delta L}{\delta a} = \dfrac{1}{2}\sum_{i=1}^n \dfrac{\delta Li}{\delta a} = \dfrac{\delta L1}{\delta a} + \dfrac{\delta L2}{\delta a}$

$ \dfrac{\delta L1}{\delta a}=\dfrac{\delta L1}{\delta y1}\dfrac{\delta y1}{\delta a} =  \dfrac{\delta }{\delta a} (a*h1+c)= \dfrac{\delta }{\delta a} (a*sigmoid(wi*x0+wh*h0+b)+c) = d1*sigmoid(wi*x0+wh*h0+b) = d1*h1 = 0.775803492574$

$ \dfrac{\delta L2}{\delta a}=\dfrac{\delta L2}{\delta y2}\dfrac{\delta y2}{\delta a} = \dfrac{\delta }{\delta a} (a*h2+c) = \dfrac{\delta }{\delta a} (a*sigmoid(wi*x1+wh*h1+b)+c) = d2*sigmoid(wi*x1+wh*h1+b)= d2*h2 = 0.752907895156$

$\dfrac{\delta L}{\delta a} = \dfrac{\delta L1}{\delta a} + \dfrac{\delta L2}{\delta a} = 1.5287113877300644
$

$\dfrac{\delta L}{\delta c} = \dfrac{1}{2}\sum_{i=1}^n \dfrac{\delta Li}{\delta c} = \dfrac{\delta L1}{\delta c} + \dfrac{\delta L2}{\delta c}$

$ \dfrac{\delta L1}{\delta c}=\dfrac{\delta L1}{\delta y1}\dfrac{\delta y1}{\delta c} =  \dfrac{\delta }{\delta c} (a*h1+c)= \dfrac{\delta }{\delta a} (a*sigmoid(wi*x0+wh*h0+b)+c) = d1 = 0.880797077978$

$ \dfrac{\delta L2}{\delta c}=\dfrac{\delta L2}{\delta y2}\dfrac{\delta y2}{\delta c} = \dfrac{\delta }{\delta c} (a*h2+c) = \dfrac{\delta }{\delta a} (a*sigmoid(wi*x1+wh*h1+b)+c) = d2 = 0.867702653653$

$\dfrac{\delta L}{\delta c} = \dfrac{\delta L1}{\delta c} + \dfrac{\delta L2}{\delta c} = 1.7484997316304389
$

<span style="color:red">TODO:</span>

Answer the above question. 

* The answers are listed above.
* You can use LATEX to edit the equations, and Jupyter notebook can recognize basic LATEX syntax. Alternatively, you can edit equations in some other environment and then paste the screenshot of the equations here.

So,

$dwi =0.10293621452669977$

$dwh =0.0877341857017$

$db =0.20254393109983646$

$da =1.5287113877300644$

$dc =1.7484997316304389$

## Task 1, Part 2: Use tensorflow modules to create XOR network

In this part, you need to build and train an XOR network that can learn the XOR function. It is a very simple implementation of RNN and will give you an idea how RNN is built and how to train it.

### XOR network

XOR network can learn the XOR $\oplus$ function

As shown in the figure below, and for instance, if input $(x0, x1, x2)$=(1,0,0), then output $(y1, y2, y3)$=(1,1,1). That is, $y_n = x_0\oplus x_1 \oplus ... \oplus x_{n-1}$

![xor_net](./img/xor.png)

### Create data set
This function provides you the way to generate the data which is required for the training process. You should utilize it when building your training function for the LSTM. Please read the source code for more information.


```python
from ecbm4040.xor.utils import create_dataset
```

### Build a network using a Tensorlow LSTMCell
This section shows an example how to build a RNN network using an LSTM cell. LSTM cell is an inbuilt class in tensorflow which implements the real behavior of the LSTM neuron. 

Reference: [TensorFlow LSTM cell](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMCell)


```python
from tensorflow.contrib.rnn import LSTMCell

tf.reset_default_graph()

# Input shape: (num_samples, seq_length, input_dimension)
# Output shape: (num_samples, output_ground_truth), and output_ground_truth is 0/1. 
input_data = tf.placeholder(tf.float32,shape=[None,None,1])
output_data = tf.placeholder(tf.int64,shape=[None,None])

# define LSTM cell
lstm_units = 64
cell = LSTMCell(lstm_units,num_proj=2,state_is_tuple=True)

# create LSTM network: you can also choose other modules provided by tensorflow, like static_rnn etc.
out,_ = tf.nn.dynamic_rnn(cell,input_data,dtype=tf.float32)
pred = tf.argmax(out,axis=2)

# loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=output_data,logits=out))

# optimization
optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

# accuracy
correct_num = tf.equal(output_data,pred)
accuracy = tf.reduce_mean(tf.cast(correct_num,tf.float32))
```

### Training 

<span style='color:red'>TODO:</span> 
1. Build your training funciton for RNN; 
2. Plot the cost during the traning


```python
# YOUR TRAINING AND PLOTTING CODE HERE
init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)
costs = []

for epoch in range(100):
    inp_data,out_data = create_dataset(4096)
    _,c,acc = sess.run([optimizer,loss,accuracy],feed_dict={input_data: inp_data, output_data: out_data})
    print("Epoch: {}, Cost: {}, Accuracy: {}%".format(epoch,c,acc*100))
    costs.append(c)
```

    WARNING:tensorflow:From /Users/jhuang/anaconda/lib/python3.6/site-packages/tensorflow/python/util/tf_should_use.py:175: initialize_all_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.
    Instructions for updating:
    Use `tf.global_variables_initializer` instead.
    Epoch: 0, Cost: 0.6945266723632812, Accuracy: 49.9664306640625%
    Epoch: 1, Cost: 0.7909407019615173, Accuracy: 50.1251220703125%
    Epoch: 2, Cost: 0.6902364492416382, Accuracy: 54.425048828125%
    Epoch: 3, Cost: 0.7074846029281616, Accuracy: 49.9786376953125%
    Epoch: 4, Cost: 0.694115400314331, Accuracy: 49.7528076171875%
    Epoch: 5, Cost: 0.6875132322311401, Accuracy: 51.6448974609375%
    Epoch: 6, Cost: 0.689591646194458, Accuracy: 45.184326171875%
    Epoch: 7, Cost: 0.6866284608840942, Accuracy: 46.2310791015625%
    Epoch: 8, Cost: 0.6774929165840149, Accuracy: 51.116943359375%
    Epoch: 9, Cost: 0.6651238203048706, Accuracy: 53.204345703125%
    Epoch: 10, Cost: 0.6472246050834656, Accuracy: 65.41748046875%
    Epoch: 11, Cost: 0.6040103435516357, Accuracy: 73.9593505859375%
    Epoch: 12, Cost: 0.7692430019378662, Accuracy: 50.7568359375%
    Epoch: 13, Cost: 0.534562349319458, Accuracy: 75.091552734375%
    Epoch: 14, Cost: 0.480993390083313, Accuracy: 85.4461669921875%
    Epoch: 15, Cost: 0.44931545853614807, Accuracy: 87.396240234375%
    Epoch: 16, Cost: 0.3809756636619568, Accuracy: 93.7713623046875%
    Epoch: 17, Cost: 0.3892737030982971, Accuracy: 86.602783203125%
    Epoch: 18, Cost: 0.2969946265220642, Accuracy: 99.3133544921875%
    Epoch: 19, Cost: 0.2947588562965393, Accuracy: 97.7691650390625%
    Epoch: 20, Cost: 0.21926428377628326, Accuracy: 98.8494873046875%
    Epoch: 21, Cost: 0.21866172552108765, Accuracy: 98.199462890625%
    Epoch: 22, Cost: 0.18473997712135315, Accuracy: 99.951171875%
    Epoch: 23, Cost: 0.16714827716350555, Accuracy: 100.0%
    Epoch: 24, Cost: 0.1439284384250641, Accuracy: 100.0%
    Epoch: 25, Cost: 0.10538510233163834, Accuracy: 100.0%
    Epoch: 26, Cost: 0.08255763351917267, Accuracy: 100.0%
    Epoch: 27, Cost: 0.10855208337306976, Accuracy: 98.5321044921875%
    Epoch: 28, Cost: 0.07385584712028503, Accuracy: 99.151611328125%
    Epoch: 29, Cost: 0.06075176224112511, Accuracy: 100.0%
    Epoch: 30, Cost: 0.050613123923540115, Accuracy: 100.0%
    Epoch: 31, Cost: 0.04097375273704529, Accuracy: 100.0%
    Epoch: 32, Cost: 0.03267980366945267, Accuracy: 100.0%
    Epoch: 33, Cost: 0.024934250861406326, Accuracy: 100.0%
    Epoch: 34, Cost: 0.020131153985857964, Accuracy: 100.0%
    Epoch: 35, Cost: 0.022283067926764488, Accuracy: 99.9237060546875%
    Epoch: 36, Cost: 0.016906065866351128, Accuracy: 99.932861328125%
    Epoch: 37, Cost: 0.012956753373146057, Accuracy: 100.0%
    Epoch: 38, Cost: 0.010629123076796532, Accuracy: 100.0%
    Epoch: 39, Cost: 0.009134603664278984, Accuracy: 100.0%
    Epoch: 40, Cost: 0.008165840059518814, Accuracy: 100.0%
    Epoch: 41, Cost: 0.007562497165054083, Accuracy: 100.0%
    Epoch: 42, Cost: 0.007031270768493414, Accuracy: 100.0%
    Epoch: 43, Cost: 0.006362444721162319, Accuracy: 100.0%
    Epoch: 44, Cost: 0.005739092361181974, Accuracy: 100.0%
    Epoch: 45, Cost: 0.005086345598101616, Accuracy: 100.0%
    Epoch: 46, Cost: 0.0045979078859090805, Accuracy: 100.0%
    Epoch: 47, Cost: 0.004075147211551666, Accuracy: 100.0%
    Epoch: 48, Cost: 0.0036408118903636932, Accuracy: 100.0%
    Epoch: 49, Cost: 0.0032400209456682205, Accuracy: 100.0%
    Epoch: 50, Cost: 0.0028766775503754616, Accuracy: 100.0%
    Epoch: 51, Cost: 0.0025095625314861536, Accuracy: 100.0%
    Epoch: 52, Cost: 0.002156311646103859, Accuracy: 100.0%
    Epoch: 53, Cost: 0.0019043690990656614, Accuracy: 100.0%
    Epoch: 54, Cost: 0.0016763345338404179, Accuracy: 100.0%
    Epoch: 55, Cost: 0.0014982974389567971, Accuracy: 100.0%
    Epoch: 56, Cost: 0.0013622110709547997, Accuracy: 100.0%
    Epoch: 57, Cost: 0.0012564060743898153, Accuracy: 100.0%
    Epoch: 58, Cost: 0.0011698990128934383, Accuracy: 100.0%
    Epoch: 59, Cost: 0.0011035732459276915, Accuracy: 100.0%
    Epoch: 60, Cost: 0.0010423329658806324, Accuracy: 100.0%
    Epoch: 61, Cost: 0.0009874905226752162, Accuracy: 100.0%
    Epoch: 62, Cost: 0.0009453060920350254, Accuracy: 100.0%
    Epoch: 63, Cost: 0.0008957330719567835, Accuracy: 100.0%
    Epoch: 64, Cost: 0.0008515184163115919, Accuracy: 100.0%
    Epoch: 65, Cost: 0.0008107451139949262, Accuracy: 100.0%
    Epoch: 66, Cost: 0.0007782327011227608, Accuracy: 100.0%
    Epoch: 67, Cost: 0.0007443674257956445, Accuracy: 100.0%
    Epoch: 68, Cost: 0.0007167308358475566, Accuracy: 100.0%
    Epoch: 69, Cost: 0.0006859105196781456, Accuracy: 100.0%
    Epoch: 70, Cost: 0.000663363840430975, Accuracy: 100.0%
    Epoch: 71, Cost: 0.0006420245626941323, Accuracy: 100.0%
    Epoch: 72, Cost: 0.0006189285777509212, Accuracy: 100.0%
    Epoch: 73, Cost: 0.0006010476499795914, Accuracy: 100.0%
    Epoch: 74, Cost: 0.000581238535232842, Accuracy: 100.0%
    Epoch: 75, Cost: 0.0005646746139973402, Accuracy: 100.0%
    Epoch: 76, Cost: 0.0005495158256962895, Accuracy: 100.0%
    Epoch: 77, Cost: 0.0005342328804545105, Accuracy: 100.0%
    Epoch: 78, Cost: 0.0005213780095800757, Accuracy: 100.0%
    Epoch: 79, Cost: 0.0005075117223896086, Accuracy: 100.0%
    Epoch: 80, Cost: 0.0004941643564961851, Accuracy: 100.0%
    Epoch: 81, Cost: 0.0004851994744967669, Accuracy: 100.0%
    Epoch: 82, Cost: 0.00047108373837545514, Accuracy: 100.0%
    Epoch: 83, Cost: 0.0004604866844601929, Accuracy: 100.0%
    Epoch: 84, Cost: 0.0004519902286119759, Accuracy: 100.0%
    Epoch: 85, Cost: 0.0004418535390868783, Accuracy: 100.0%
    Epoch: 86, Cost: 0.0004297451814636588, Accuracy: 100.0%
    Epoch: 87, Cost: 0.00042301934445276856, Accuracy: 100.0%
    Epoch: 88, Cost: 0.00041493450407870114, Accuracy: 100.0%
    Epoch: 89, Cost: 0.00040768703911453485, Accuracy: 100.0%
    Epoch: 90, Cost: 0.0003997471940238029, Accuracy: 100.0%
    Epoch: 91, Cost: 0.0003934311680495739, Accuracy: 100.0%
    Epoch: 92, Cost: 0.0003866484039463103, Accuracy: 100.0%
    Epoch: 93, Cost: 0.000382086233003065, Accuracy: 100.0%
    Epoch: 94, Cost: 0.00037471612449735403, Accuracy: 100.0%
    Epoch: 95, Cost: 0.0003708881267812103, Accuracy: 100.0%
    Epoch: 96, Cost: 0.0003647316771093756, Accuracy: 100.0%
    Epoch: 97, Cost: 0.0003620669012889266, Accuracy: 100.0%
    Epoch: 98, Cost: 0.00035700146690942347, Accuracy: 100.0%
    Epoch: 99, Cost: 0.0003509907692205161, Accuracy: 100.0%



```python
plt.grid("off")
plt.plot(costs,label="Cost Function")
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.legend()
plt.show()
 
sess.close()
```


![png](task1-xor_files/task1-xor_21_0.png)


## Task 1, Part 3 :  Build your own LSTMCell
In this part, you need to build your own LSTM cell to achieve the LSTM functionality. 

<span style="color:red">TODO:</span> 
1. Finish class **MyLSTMCell** in ecbm4040/xor/rnn.py;
2. Write the training function for your RNN;
3. Plot the cost during training.


```python
print(tf.__file__)
```

    /Users/jhuang/anaconda/lib/python3.6/site-packages/tensorflow/__init__.py



```python
from ecbm4040.xor.rnn import MyLSTMCell

# recreate xor netowrk with your own LSTM cell
tf.reset_default_graph()

#Input shape: (num_samples,seq_length,input_dimension)
#Output shape: (num_samples, output_ground_truth), and output_ground_truth is 0/1. 
input_data = tf.placeholder(tf.float32,shape=[None,None,1])
output_data = tf.placeholder(tf.int64,shape=[None,None])

# recreate xor netowrk with your own LSTM cell
lstm_units = 64
cell = MyLSTMCell(lstm_units,num_proj=2)

# create LSTM network: you can also choose other modules provided by tensorflow, like static_rnn etc.
out,_ = tf.nn.dynamic_rnn(cell,input_data,dtype=tf.float32)
pred = tf.argmax(out,axis=2)

# loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=output_data,logits=out))
# optimization
optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
# accuracy
correct = tf.equal(output_data,pred)
accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
```

### Training


```python
# YOUR TRAINING AND PLOTTING CODE HERE
init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)
costs = []

for epoch in range(100):
    inp_data,out_data = create_dataset(4096)
    _,c,acc = sess.run([optimizer,loss,accuracy],feed_dict={input_data: inp_data, output_data: out_data})
    print("Epoch: {}, Cost: {}, Accuracy: {}%".format(epoch,c,acc*100))
    costs.append(c)
```

    WARNING:tensorflow:From /Users/jhuang/anaconda/lib/python3.6/site-packages/tensorflow/python/util/tf_should_use.py:175: initialize_all_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.
    Instructions for updating:
    Use `tf.global_variables_initializer` instead.
    Epoch: 0, Cost: 3.482042074203491, Accuracy: 49.493408203125%
    Epoch: 1, Cost: 1.2660353183746338, Accuracy: 45.880126953125%
    Epoch: 2, Cost: 1.1222479343414307, Accuracy: 56.3873291015625%
    Epoch: 3, Cost: 0.793681263923645, Accuracy: 64.7613525390625%
    Epoch: 4, Cost: 0.6406052112579346, Accuracy: 62.4053955078125%
    Epoch: 5, Cost: 0.6366516351699829, Accuracy: 62.457275390625%
    Epoch: 6, Cost: 0.6351776123046875, Accuracy: 68.157958984375%
    Epoch: 7, Cost: 0.6116011738777161, Accuracy: 68.4783935546875%
    Epoch: 8, Cost: 0.5923513174057007, Accuracy: 65.3167724609375%
    Epoch: 9, Cost: 0.6044180989265442, Accuracy: 70.1416015625%
    Epoch: 10, Cost: 0.6386483907699585, Accuracy: 67.51708984375%
    Epoch: 11, Cost: 0.5430147647857666, Accuracy: 68.487548828125%
    Epoch: 12, Cost: 0.5479917526245117, Accuracy: 73.7274169921875%
    Epoch: 13, Cost: 0.5215020179748535, Accuracy: 78.2745361328125%
    Epoch: 14, Cost: 0.5215981006622314, Accuracy: 72.8729248046875%
    Epoch: 15, Cost: 0.517303466796875, Accuracy: 76.824951171875%
    Epoch: 16, Cost: 0.5043891072273254, Accuracy: 80.0628662109375%
    Epoch: 17, Cost: 0.49293217062950134, Accuracy: 78.9154052734375%
    Epoch: 18, Cost: 0.46126753091812134, Accuracy: 82.4951171875%
    Epoch: 19, Cost: 0.42243173718452454, Accuracy: 87.542724609375%
    Epoch: 20, Cost: 0.3746420741081238, Accuracy: 91.998291015625%
    Epoch: 21, Cost: 0.37540847063064575, Accuracy: 88.739013671875%
    Epoch: 22, Cost: 0.5068713426589966, Accuracy: 80.7952880859375%
    Epoch: 23, Cost: 0.494955837726593, Accuracy: 80.4534912109375%
    Epoch: 24, Cost: 0.4459386467933655, Accuracy: 83.2550048828125%
    Epoch: 25, Cost: 0.65921550989151, Accuracy: 68.84765625%
    Epoch: 26, Cost: 0.5752344131469727, Accuracy: 75.9063720703125%
    Epoch: 27, Cost: 1.1288864612579346, Accuracy: 66.6259765625%
    Epoch: 28, Cost: 0.6090895533561707, Accuracy: 71.8292236328125%
    Epoch: 29, Cost: 0.49751758575439453, Accuracy: 79.3853759765625%
    Epoch: 30, Cost: 0.6267199516296387, Accuracy: 71.221923828125%
    Epoch: 31, Cost: 0.6802867650985718, Accuracy: 67.962646484375%
    Epoch: 32, Cost: 0.559303879737854, Accuracy: 73.2574462890625%
    Epoch: 33, Cost: 0.5692689418792725, Accuracy: 69.86083984375%
    Epoch: 34, Cost: 0.5840338468551636, Accuracy: 75.030517578125%
    Epoch: 35, Cost: 0.5373003482818604, Accuracy: 74.3988037109375%
    Epoch: 36, Cost: 0.47069215774536133, Accuracy: 80.0323486328125%
    Epoch: 37, Cost: 0.5025640726089478, Accuracy: 70.3582763671875%
    Epoch: 38, Cost: 0.4022868871688843, Accuracy: 84.527587890625%
    Epoch: 39, Cost: 0.2947998046875, Accuracy: 92.7337646484375%
    Epoch: 40, Cost: 0.3142583966255188, Accuracy: 91.39404296875%
    Epoch: 41, Cost: 0.29434311389923096, Accuracy: 91.290283203125%
    Epoch: 42, Cost: 0.22096985578536987, Accuracy: 97.7142333984375%
    Epoch: 43, Cost: 0.22571256756782532, Accuracy: 97.137451171875%
    Epoch: 44, Cost: 0.17828942835330963, Accuracy: 99.1546630859375%
    Epoch: 45, Cost: 0.16425731778144836, Accuracy: 99.786376953125%
    Epoch: 46, Cost: 0.15667223930358887, Accuracy: 100.0%
    Epoch: 47, Cost: 0.1483973115682602, Accuracy: 100.0%
    Epoch: 48, Cost: 0.1331951916217804, Accuracy: 100.0%
    Epoch: 49, Cost: 0.131735697388649, Accuracy: 100.0%
    Epoch: 50, Cost: 0.12969136238098145, Accuracy: 99.969482421875%
    Epoch: 51, Cost: 0.12196507304906845, Accuracy: 100.0%
    Epoch: 52, Cost: 0.11654438823461533, Accuracy: 100.0%
    Epoch: 53, Cost: 0.11163084208965302, Accuracy: 100.0%
    Epoch: 54, Cost: 0.11087283492088318, Accuracy: 100.0%
    Epoch: 55, Cost: 0.10934635996818542, Accuracy: 100.0%
    Epoch: 56, Cost: 0.10804207623004913, Accuracy: 100.0%
    Epoch: 57, Cost: 0.10570874810218811, Accuracy: 100.0%
    Epoch: 58, Cost: 0.10462907701730728, Accuracy: 100.0%
    Epoch: 59, Cost: 0.10148459672927856, Accuracy: 100.0%
    Epoch: 60, Cost: 0.0983668863773346, Accuracy: 100.0%
    Epoch: 61, Cost: 0.09907491505146027, Accuracy: 100.0%
    Epoch: 62, Cost: 0.10171129554510117, Accuracy: 100.0%
    Epoch: 63, Cost: 0.10123620182275772, Accuracy: 100.0%
    Epoch: 64, Cost: 0.09921875596046448, Accuracy: 100.0%
    Epoch: 65, Cost: 0.09867212921380997, Accuracy: 100.0%
    Epoch: 66, Cost: 0.09604841470718384, Accuracy: 100.0%
    Epoch: 67, Cost: 0.0968579649925232, Accuracy: 100.0%
    Epoch: 68, Cost: 0.09898480772972107, Accuracy: 100.0%
    Epoch: 69, Cost: 0.0992082953453064, Accuracy: 100.0%
    Epoch: 70, Cost: 0.09563229233026505, Accuracy: 100.0%
    Epoch: 71, Cost: 0.09463508427143097, Accuracy: 100.0%
    Epoch: 72, Cost: 0.09279245138168335, Accuracy: 100.0%
    Epoch: 73, Cost: 0.09480009227991104, Accuracy: 100.0%
    Epoch: 74, Cost: 0.09654276072978973, Accuracy: 100.0%
    Epoch: 75, Cost: 0.08970564603805542, Accuracy: 100.0%
    Epoch: 76, Cost: 0.09438067674636841, Accuracy: 100.0%
    Epoch: 77, Cost: 0.09375324845314026, Accuracy: 100.0%
    Epoch: 78, Cost: 0.09289289265871048, Accuracy: 100.0%
    Epoch: 79, Cost: 0.09046365320682526, Accuracy: 100.0%
    Epoch: 80, Cost: 0.08997613936662674, Accuracy: 100.0%
    Epoch: 81, Cost: 0.09326943755149841, Accuracy: 100.0%
    Epoch: 82, Cost: 0.09131073951721191, Accuracy: 100.0%
    Epoch: 83, Cost: 0.09264369308948517, Accuracy: 100.0%
    Epoch: 84, Cost: 0.09183263778686523, Accuracy: 100.0%
    Epoch: 85, Cost: 0.09115315228700638, Accuracy: 100.0%
    Epoch: 86, Cost: 0.0957297682762146, Accuracy: 100.0%
    Epoch: 87, Cost: 0.09113448858261108, Accuracy: 100.0%
    Epoch: 88, Cost: 0.09231498837471008, Accuracy: 100.0%
    Epoch: 89, Cost: 0.0908622145652771, Accuracy: 100.0%
    Epoch: 90, Cost: 0.09105950593948364, Accuracy: 100.0%
    Epoch: 91, Cost: 0.09167163074016571, Accuracy: 100.0%
    Epoch: 92, Cost: 0.09084542840719223, Accuracy: 100.0%
    Epoch: 93, Cost: 0.08716800808906555, Accuracy: 100.0%
    Epoch: 94, Cost: 0.08986025303602219, Accuracy: 100.0%
    Epoch: 95, Cost: 0.08671185374259949, Accuracy: 100.0%
    Epoch: 96, Cost: 0.08617036044597626, Accuracy: 100.0%
    Epoch: 97, Cost: 0.08848012983798981, Accuracy: 100.0%
    Epoch: 98, Cost: 0.08917409181594849, Accuracy: 100.0%
    Epoch: 99, Cost: 0.09077800810337067, Accuracy: 100.0%



```python
import tensorflow as tf
print (tf.__file__)
```

    /Users/jhuang/anaconda/lib/python3.6/site-packages/tensorflow/__init__.py



```python
plt.grid("off")
plt.plot(costs,label="Cost Function")
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.legend()
plt.show()
 
sess.close()
```


![png](task1-xor_files/task1-xor_28_0.png)

