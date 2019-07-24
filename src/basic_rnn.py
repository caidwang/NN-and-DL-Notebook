'''
Author: Caid Wang

Implement Basic Recurrent Neural Network in numpy.
'''

import numpy as np

def softmax(x):
    """
    Calculate the softmax value of x on 0 dim

    Arguments:
    x -- np.ndarray of (m, 1) dimension
    Returns:
    ret -- the softmax value, shape of (m, 1) 
    """
    e = np.exp(x)
    total = np.sum(e, axis=0)
    ret = e / total
    return ret


def rnn_cell_forward(xt, a_prev, parameters):
    """
    Implements a single forward step of the RNN-cell as described in Figure (2)

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt, parameters)
    """
    Wax = parameters['Wax']
    Waa = parameters['Waa']
    Wya = parameters['Wya']
    ba = parameters['ba']
    by = parameters['by']

    a_next = np.tanh(Wax.dot(xt) + Waa.dot(a_prev) + ba)
    yt_pred = softmax(Wya.dot(a_next) + by)

    cache = (a_next, a_prev, xt, parameters) # what I need

    return a_next, yt_pred, cache


def rnn_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network described in Figure (3).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y_pred -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of caches, x)
    """
    Wax = parameters['Wax']
    Waa = parameters['Waa']
    Wya = parameters['Wya']
    ba = parameters['ba']
    by = parameters['by']
    
    n_x, m, T_x = x.shape
    n_a = a0.shape[0]
    n_y = by.shape[0]
    at = a0
    y_pred = np.zeros((n_y, m, T_x))
    a = np.zeros((n_a, m, T_x))
    caches = []

    for t in range(T_x):
        a[:, :, t], y_pred[:, :, t], cache = rnn_cell_forward(x[:, :, t], at, parameters)
        at = a[:, :, t]
        caches.append(cache)
    caches = [caches, x]

    return a, y_pred, caches


def initialize_parameters(x_dim, hidden_units, y_dim):
    Wax = np.random.randn((hidden_units, x_dim)) # (n_a, n_x)
    Waa = np.random.randn((hidden_units, hidden_units)) # (n_a, n_a)
    Wya = np.random.randn((y_dim, hidden_units)) # (n_y, n_a)
    ba = np.zeros((hidden_units, 1)) # (n_a, 1)
    by = np.zeros((y_dim, 1)) # (n_y, 1)
    parameters = {
        'Wax':Wax,
        'Waa':Waa,
        'Wya':Wya, 
        'by':by,
        'ba':ba
    }
    return parameters


def rnn_cell_backward(da_next, cache):
    """
    Implements the backward pass for the RNN-cell (single time-step).

    Arguments:
    da_next -- Gradient of loss with respect to next hidden state
    cache -- python dictionary containing useful values (output of rnn_cell_forward())

    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradients of input data, of shape (n_x, m)
                        da_prev -- Gradients of previous hidden state, of shape (n_a, m)
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dba -- Gradients of bias vector, of shape (n_a, 1)
    """
    a_next, a_prev, xt, parameters = cache
    Wax = parameters['Wax']
    Waa = parameters['Waa']
    Wya = parameters['Wya']
    ba = parameters['ba']
    by = parameters['by']

    # Loss对Wax和Waa的求导 等于 Loss对at的求导 乘以 at对Zt的求导 乘以 Zt对W的求导
    dtanh = da_next * (1 - a_next**2) # (n_a, m) 

    dWax = np.dot(dtanh, xt.T) # Zt对Wax的求导 # (n_a, n_x)
    dWaa = np.dot(dtanh, a_prev.T) # (n_a, n_a)
    dba = np.sum(dtanh, axis=1, keepdims=True) # (n_a, 1)
    dx = np.dot(Wax.T, dtanh) # (n_x, m) 纵向具有多层时 需要
    da_prev = np.dot(Waa.T, dtanh) # (n_a, m) 横向的传播 这时将a_t-1 看做一个变量而非函数, 因为之后链式求导它还会继续乘下去然后加到最终的结果里, 总之, 针对dWaa的求导不用急着把a_t-1展开, 只要有对它的求导就够了 
    '''
    在这个函数中, 只处理a_t对W等的求导, 纵向上从上方传来的对a_t的求导和横向上从右边传过来的对a_t的求导, 相加后作为Cost对a_t的求导整体, 再乘以at对以上变量的求导
    '''
    gradients = {
        'dx':dx,
        'da_prev':da_prev,
        'dWax':dWax,
        'dWaa':dWaa,
        'dba':dba
    }
    return gradients

def rnn_backward(da, caches):
    """
    Implement the backward pass for a RNN over an entire sequence of input data.

    Arguments:
    da -- Upstream gradients of all hidden states, of shape (n_a, m, T_x)
    caches -- tuple containing information from the forward pass (rnn_forward)
    
    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradient w.r.t. the input data, numpy-array of shape (n_x, m, T_x)
                        da0 -- Gradient w.r.t the initial hidden state, numpy-array of shape (n_a, m)
                        dWax -- Gradient w.r.t the input's weight matrix, numpy-array of shape (n_a, n_x)
                        dWaa -- Gradient w.r.t the hidden state's weight matrix, numpy-arrayof shape (n_a, n_a)
                        dba -- Gradient w.r.t the bias, of shape (n_a, 1)
    """
    caches, x = caches # caches -- list, 包含所有时间步的cache, x -- (n_x, m, T_x) 输入维度
    a1, a0, x1, parameters = caches[0]
    n_x, m, T_x = x.shape
    n_a = a1.shape[0]

    dx = np.zeros((n_x, m, T_x))
    dWax = np.zeros((n_a, n_x))
    dWaa = np.zeros((n_a, n_a))
    dba = np.zeros((n_a, 1))
    da_prevt = np.zeros((n_a, m))

    for t in reversed(range(T_x)):
        # 在这里传入da_next的时候, 传入从右方和从上方传来的对Cost对a_t的求导之和, 求得的t时刻的梯度, 需要累加起来, 因为Cost是对所有t的累加
        gradients = rnn_cell_backward(da[:, :, t] + da_prevt, caches[t])
        dxt, da_prevt, dWaxt, dWaat, dbat = gradients['dx'], gradients['da_prev'], gradients['dWax'], gradients['dWaa'], gradients['dba']
        dx[:, :, t] = dxt
        dWax += dWaxt
        dWaa += dWaat
        dba += dbat
    da0 = da_prevt

    gradients = {
        'dx': dx, 
        'da0': da0,
        'dWax': dWax, 
        'dWaa': dWaa, 
        'dba': dba
    }
    return gradients


def test_softmax():
    x = np.random.randn(5, 1)
    print(softmax(x))

def main():
    np.random.seed(1)
    x = np.random.randn(3,10,4)
    a0 = np.random.randn(5,10)
    Wax = np.random.randn(5,3)
    Waa = np.random.randn(5,5)
    Wya = np.random.randn(2,5)
    ba = np.random.randn(5,1)
    by = np.random.randn(2,1)
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}
    a, y, caches = rnn_forward(x, a0, parameters)
    da = np.random.randn(5, 10, 4)
    gradients = rnn_backward(da, caches)

    print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
    print("gradients[\"dx\"].shape =", gradients["dx"].shape)
    print("gradients[\"da0\"][2][3] =", gradients["da0"][2][3])
    print("gradients[\"da0\"].shape =", gradients["da0"].shape)
    print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
    print("gradients[\"dWax\"].shape =", gradients["dWax"].shape)
    print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
    print("gradients[\"dWaa\"].shape =", gradients["dWaa"].shape)
    print("gradients[\"dba\"][4] =", gradients["dba"][4])
    print("gradients[\"dba\"].shape =", gradients["dba"].shape)



if __name__ == '__main__':
    main()

'''
expect:

<table>
    <tr>
        <td>
            **gradients["dx"][1][2]** =
        </td>
        <td>
           [-2.07101689 -0.59255627  0.02466855  0.01483317]
        </td>
    </tr>
        <tr>
        <td>
            **gradients["dx"].shape** =
        </td>
        <td>
           (3, 10, 4)
        </td>
    </tr>
        <tr>
        <td>
            **gradients["da0"][2][3]** =
        </td>
        <td>
           -0.314942375127
        </td>
    </tr>
        <tr>
        <td>
            **gradients["da0"].shape** =
        </td>
        <td>
           (5, 10)
        </td>
    </tr>
         <tr>
        <td>
            **gradients["dWax"][3][1]** =
        </td>
        <td>
           11.2641044965
        </td>
    </tr>
        <tr>
        <td>
            **gradients["dWax"].shape** =
        </td>
        <td>
           (5, 3)
        </td>
    </tr>
        <tr>
        <td>
            **gradients["dWaa"][1][2]** = 
        </td>
        <td>
           2.30333312658
        </td>
    </tr>
        <tr>
        <td>
            **gradients["dWaa"].shape** =
        </td>
        <td>
           (5, 5)
        </td>
    </tr>
        <tr>
        <td>
            **gradients["dba"][4]** = 
        </td>
        <td>
           [-0.74747722]
        </td>
    </tr>
        <tr>
        <td>
            **gradients["dba"].shape** = 
        </td>
        <td>
           (5, 1)
        </td>
    </tr>
</table>
'''