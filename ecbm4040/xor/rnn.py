#!/usr/bin/env python
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class MyLSTMCell(RNNCell):
    """
    Your own basic LSTMCell implementation that is compatible with TensorFlow. To solve the compatibility issue, this
    class inherits TensorFlow RNNCell class.

    For reference, you can look at the TensorFlow LSTMCell source code. To locate the TensorFlow installation path, do
    the following:

    1. In Python, type 'import tensorflow as tf', then 'print(tf.__file__)'

    2. According to the output, find tensorflow_install_path/python/ops/rnn_cell_impl.py

    So this is basically rewriting the TensorFlow LSTMCell, but with your own language.

    Also, you will find Colah's blog about LSTM to be very useful:
    http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    """

    def __init__(self, num_units, num_proj, forget_bias=1.0, activation=None):
        """
        Initialize a class instance.

        In this function, you need to do the following:

        1. Store the input parameters and calculate other ones that you think necessary.

        2. Initialize some trainable variables which will be used during the calculation.

        :param num_units: The number of units in the LSTM cell.
        :param num_proj: The output dimensionality. For example, if you expect your output of the cell at each time step
                         to be a 10-element vector, then num_proj = 10.
        :param forget_bias: The bias term used in the forget gate. By default we set it to 1.0.
        :param activation: The activation used in the inner states. By default we use tanh.

        There are biases used in other gates, but since TensorFlow doesn't have them, we don't implement them either.
        """
        super(MyLSTMCell, self).__init__(_reuse=True)
        #############################################
        #           TODO: YOUR CODE HERE            #
        self._num_units = num_units
        self._num_proj = num_proj
        self._forget_bias = forget_bias
        self._activation = activation or math_ops.tanh

        self._state_size = num_units + num_proj
        self._output_size = num_proj

        W_f = tf.Variable(tf.random_normal([self._num_proj + 1, self._num_units]))
        W_i = tf.Variable(tf.random_normal([self._num_proj + 1, self._num_units]))
        W_j = tf.Variable(tf.random_normal([self._num_proj + 1, self._num_units]))
        W_o = tf.Variable(tf.random_normal([self._num_proj + 1, self._num_units]))
        W_h = tf.Variable(tf.random_normal([self._num_units, self._num_proj]))
        
        self.W_f = W_f
        self.W_i = W_i
        self.W_j = W_j
        self.W_o = W_o
        self.W_h = W_h

        # shape = 3

        # output_size = 4 * self._num_units

        # init = tf.truncated_normal([shape, output_size], mean=0.0, stddev=1.0 / shape**0.5)
        # init_bias=0.0
        # W = tf.get_variable("weight", initializer=init)
        # b = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(init_bias))
        # self.W = W
        # self.b = b
        #############################################

    # The following 2 properties are required when defining a TensorFlow RNNCell.
    @property
    def state_size(self):
        """
        Overrides parent class method. Returns the state size of of the cell.

        state size = num_units + output_size

        :return: An integer.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        return self._state_size
        #############################################

    @property
    def output_size(self):
        """
        Overrides parent class method. Returns the output size of the cell.

        :return: An integer.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        return self._output_size
        #############################################

    def call(self, inputs, state):
        """
        Run one time step of the cell. That is, given the current inputs and the state from the last time step,
        calculate the current state and cell output.

        You will notice that TensorFlow LSTMCell has a lot of other features. But we will not try them. Focus on the
        very basic LSTM functionality.

        Hint 1: If you try to figure out the tensor shapes, use print(a.get_shape()) to see the shape.

        Hint 2: In LSTM there exist both matrix multiplication and element-wise multiplication. Try not to mix them.

        :param inputs: The input at the current time step. The last dimension of it should be 1.
        :param state:  The state value of the cell from the last time step. The state size can be found from function
                       state_size(self).
        :return: A tuple containing (output, new_state). For details check TensorFlow LSTMCell class.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #


        concat = linear(inputs, c, h)
            
        f = tf.sigmoid(tf.matmul(concat, self.W_f) + self._forget_bias)
        i = tf.sigmoid(tf.matmul(concat, self.W_i))
        new_c_hat = tf.tanh(tf.matmul(concat, self.W_j))
        o = tf.sigmoid(tf.matmul(concat, self.W_o))
                
        new_c = tf.multiply(c, f) + tf.multiply(i, new_c_hat)       
        new_h = tf.matmul(tf.multiply(o, tf.tanh(new_c)), self.W_h) # Add a parameter to make the dimension of new_h consistent
       
        new_state = tf.concat([new_c, new_h], axis = 1)

        return new_h, new_state
        #############################################

def linear(x, c, h):
    c = tf.slice(state, [0, 0], [-1, self._num_units])
    h = tf.slice(state, [0, self._num_units], [-1, self._num_proj])        
    concat = tf.concat([h, x], axis = 1)
    return concat
