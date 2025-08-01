import numpy as np

def Add_Window_Horizon_12(data, window=3, horizon=1, single=False):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    length = len(data)
    end_index = length - horizon - window + 1
    X = []      #windows
    Y = []      #horizon
    index = 0
    if single:
        while index < end_index:
            # print(data[index:index+window]).shape
            X.append(np.mean(data[index:index+window], axis=0) / 12)
            Y.append(np.mean(data[index+window+horizon-1:index+window+horizon], axis=0) / 12)
            index = index + 1
    else:
        while index < end_index:
            X.append(np.mean(data[index:index+window], axis=0) / 12)
            Y.append(np.mean(data[index+window+horizon-1:index+window+horizon], axis=0) / 12)
            index = index + 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def Add_Window_Horizon_bike(data, window=3, horizon=1, single=False):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    length = len(data)
    end_index = length - horizon - window + 1
    X = []      #windows
    Y = []      #horizon
    index = 0
    if single:
        while index < end_index:
            # print(data[index:index+window]).shape
            X.append(data[index])
            Y.append(data[index+window])
            index = index + 1
    else:
        while index < end_index:
            X.append(data[index])
            Y.append(data[index+window])
            index = index + 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def Add_Window_Horizon(data, window=3, horizon=1, single=False):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    length = len(data)
    end_index = length - horizon - window + 1
    X = []      #windows
    Y = []      #horizon
    index = 0
    if single:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window+horizon-1:index+window+horizon])
            index = index + 1
    else:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window:index+window+horizon])
            index = index + 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


