import scipy.io as sio                     # import scipy.io for .mat file I/
import numpy as np                         # import numpy

# Implementated based on the PyTorch 
from memoryPyTorch import MemoryDNN
from optimization import bisection

import time


def plot_rate(rate_his, rolling_intv=50):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    rate_array = np.asarray(rate_his)
    df = pd.DataFrame(rate_his)


    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15, 8))
#    rolling_intv = 20

    plt.plot(np.arange(len(rate_array))+1, df.rolling(rolling_intv, min_periods=1).mean(), 'b')
    plt.fill_between(np.arange(len(rate_array))+1, df.rolling(rolling_intv, min_periods=1).min()[0], df.rolling(rolling_intv, min_periods=1).max()[0], color = 'b', alpha = 0.2)
    plt.ylabel('Normalized Computation Rate')
    plt.xlabel('Time Frames')
    plt.show()

def save_to_txt(rate_his, file_path):
    with open(file_path, 'w') as f:
        for rate in rate_his:
            f.write("%s \n" % rate)

if __name__ == "__main__":
    '''
        This algorithm generates K modes from DNN, and chooses with largest
        reward. The mode with largest reward is stored in the memory, which is
        further used to train the DNN.
        Adaptive K is implemented. K = max(K, K_his[-memory_size])
    '''

    N = 10                       # number of users
    n = 30000                    # number of time frames
    K = N                        # initialize K = N
    decoder_mode = 'OP'          # the quantization mode could be 'OP' (Order-preserving) or 'KNN'
    Memory = 1024                # capacity of memory structure
    Delta = 32                   # Update interval for adaptive K

    print('#user = %d, #channel=%d, K=%d, decoder = %s, Memory = %d, Delta = %d'%(N,n,K,decoder_mode, Memory, Delta))
    # Load data
    channel = sio.loadmat('./data/data_%d' %N)['input_h']
    rate = sio.loadmat('./data/data_%d' %N)['output_obj'] # this rate is only used to plot figures; never used to train DROO.

    # increase h to close to 1 for better training; it is a trick widely adopted in deep learning
    channel = channel * 1000000

    # generate the train and test data sample index
    # data are splitted as 80:20
    # training data are randomly sampled with duplication if n > total data size

    split_idx = int(.8 * len(channel))
    num_test = min(len(channel) - split_idx, n - int(.8 * n)) # training data size


    mem = MemoryDNN(net = [N, 120, 80, N],
                    learning_rate = 0.01,
                    training_interval=10,
                    batch_size=128,
                    memory_size=Memory
                    )

    start_time = time.time()

    rate_his = []
    rate_his_ratio = []
    mode_his = []
    k_idx_his = []
    K_his = []
    for i in range(n):
        if i % (n//10) == 0:
           print("%0.1f"%(i/n))
        if i> 0 and i % Delta == 0:
            # index counts from 0
            if Delta > 1:
                max_k = max(k_idx_his[-Delta:-1]) +1;
            else:
                max_k = k_idx_his[-1] +1;
            K = min(max_k +1, N)

        if i < n - num_test:
            # training
            i_idx = i % split_idx
        else:
            # test
            i_idx = i - n + num_test + split_idx

        h = channel[i_idx,:]

        # the action selection must be either 'OP' or 'KNN'
        m_list = mem.decode(h, K, decoder_mode)

        r_list = []
        for m in m_list:
            r_list.append(bisection(h/1000000, m)[0])

        # encode the mode with largest reward
        mem.encode(h, m_list[np.argmax(r_list)])
        # the main code for DROO training ends here




        # the following codes store some interested metrics for illustrations
        # memorize the largest reward
        rate_his.append(np.max(r_list))
        rate_his_ratio.append(rate_his[-1] / rate[i_idx][0])
        # record the index of largest reward
        k_idx_his.append(np.argmax(r_list))
        # record K in case of adaptive K
        K_his.append(K)
        mode_his.append(m_list[np.argmax(r_list)])


    total_time=time.time()-start_time
    mem.plot_cost()
    plot_rate(rate_his_ratio)

    print("Averaged normalized computation rate:", sum(rate_his_ratio[-num_test: -1])/num_test)
    print('Total time consumed:%s'%total_time)
    print('Average time per channel:%s'%(total_time/n))

    # save data into txt
    save_to_txt(k_idx_his, "k_idx_his.txt")
    save_to_txt(K_his, "K_his.txt")
    save_to_txt(mem.cost_his, "cost_his.txt")
    save_to_txt(rate_his_ratio, "rate_his_ratio.txt")
    save_to_txt(mode_his, "mode_his.txt")
