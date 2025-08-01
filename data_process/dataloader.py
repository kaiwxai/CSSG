import torch
import numpy as np
import torch.utils.data
from lib.add_window import Add_Window_Horizon, Add_Window_Horizon_12, Add_Window_Horizon_bike
from data_process.load_dataset import load_st_dataset
from lib.normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler
import controldiffeq
import pickle

def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        #column min max, to be depressed
        #note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return data, scaler

def split_data_by_days(data, val_days, test_days, interval=60):
    '''
    :param data: [B, *]
    :param val_days:
    :param test_days:
    :param interval: interval (15, 30, 60) minutes
    :return:
    '''
    T = int((24*60)/interval)
    test_data = data[-T*test_days:]
    val_data = data[-T*(test_days + val_days): -T*test_days]
    train_data = data[:-T*(test_days + val_days)]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader

def data_loader(X, adj_in, adj_out, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # X, Y = TensorFloat(X), TensorFloat(Y)
    # X = tuple(TensorFloat(x) for x in X)
    # Y = TensorFloat(Y)
    data = torch.utils.data.TensorDataset(*X, torch.tensor(adj_in), torch.tensor(adj_out), torch.tensor(Y))
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=True):
    #load raw st dataset
    data, matrix_wr_in, matrix_wr_out = load_st_dataset(args.dataset)        # B, N, D
    #normalize st data
    data, scaler = normalize_dataset(data, normalizer, args.column_wise)
    #spilit dataset by days or by ratio
    if args.test_ratio > 1:
        data_train, data_val, data_test = split_data_by_days(data, args.val_ratio, args.test_ratio)
    else:
        data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
        matrix_wr_in_train, matrix_wr_in_val, matrix_wr_in_test = split_data_by_ratio(matrix_wr_in, args.val_ratio, args.test_ratio)
        matrix_wr_out_train, matrix_wr_out_val, matrix_wr_out_test = split_data_by_ratio(matrix_wr_out, args.val_ratio, args.test_ratio)
    #add time window
    print('matrix_wr_in_train:{}'.format(matrix_wr_in_train.shape))
    print('matrix_wr_in_val:{}'.format(matrix_wr_in_val.shape))
    print('matrix_wr_in_test:{}'.format(matrix_wr_in_test.shape))

    x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single)
    x_val, y_val = Add_Window_Horizon(data_val, args.lag, args.horizon, single)
    x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single)
    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)
    ##############get dataloader######################
    train_dataloader = data_loader(x_tra, y_tra, args.batch_size, shuffle=True, drop_last=True)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, args.batch_size, shuffle=False, drop_last=False)
    return train_dataloader, val_dataloader, test_dataloader, scaler

def get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=True):
    #load raw st dataset
    print(args.dataset)
    data, matrix_wr_in, matrix_wr_out = load_st_dataset(args.dataset)        # B, N, D
    #normalize st data

    #TODO:
    data, scaler = normalize_dataset(data, normalizer, args.column_wise)

    #spilit dataset by days or by ratio
    if args.test_ratio > 1:
        data_train, data_val, data_test = split_data_by_days(data, args.val_ratio, args.test_ratio)
    else:
        data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
        
    #add time window
    x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single)
    x_val, y_val = Add_Window_Horizon(data_val, args.lag, args.horizon, single)
    x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single)
    
    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)

    matrix_wr_in_train, matrix_wr_in_val, matrix_wr_in_test = split_data_by_ratio(matrix_wr_in, args.val_ratio, args.test_ratio)
    matrix_wr_out_train, matrix_wr_out_val, matrix_wr_out_test = split_data_by_ratio(matrix_wr_out, args.val_ratio, args.test_ratio)
    
    x_tra_in, _ = Add_Window_Horizon_bike(matrix_wr_in_train, args.lag, args.horizon, single)
    print(x_tra_in.shape)
    print('---------------------------------------------------------------------')
    x_val_in, _ = Add_Window_Horizon_bike(matrix_wr_in_val, args.lag, args.horizon, single)
    print('---------------------------------------------------------------------')
    x_test_in, _ = Add_Window_Horizon_bike(matrix_wr_in_test, args.lag, args.horizon, single)
    print('---------------------------------------------------------------------')

    x_tra_out, _ = Add_Window_Horizon_bike(matrix_wr_out_train, args.lag, args.horizon, single)
    print('---------------------------------------------------------------------')
    x_val_out, _ = Add_Window_Horizon_bike(matrix_wr_out_val, args.lag, args.horizon, single)
    print('---------------------------------------------------------------------')
    x_test_out, _ = Add_Window_Horizon_bike(matrix_wr_out_test, args.lag, args.horizon, single)
    print('---------------------------------------------------------------------')

    data_category = 'traffic'
    if data_category == 'traffic':
        times = torch.linspace(0, 11, 12)
    elif data_category == 'token':
        times = torch.linspace(0, 6, 7)
    else:
        raise ValueError
    augmented_X_tra = []
    augmented_X_tra.append(times.unsqueeze(0).unsqueeze(0).repeat(x_tra.shape[0],x_tra.shape[2],1).unsqueeze(-1).transpose(1,2))
    augmented_X_tra.append(torch.Tensor(x_tra[..., :]))
    x_tra = torch.cat(augmented_X_tra, dim=3)
    augmented_X_val = []
    augmented_X_val.append(times.unsqueeze(0).unsqueeze(0).repeat(x_val.shape[0],x_val.shape[2],1).unsqueeze(-1).transpose(1,2))
    augmented_X_val.append(torch.Tensor(x_val[..., :]))
    x_val = torch.cat(augmented_X_val, dim=3)
    augmented_X_test = []
    augmented_X_test.append(times.unsqueeze(0).unsqueeze(0).repeat(x_test.shape[0],x_test.shape[2],1).unsqueeze(-1).transpose(1,2))
    augmented_X_test.append(torch.Tensor(x_test[..., :]))
    x_test = torch.cat(augmented_X_test, dim=3)

   
    train_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, x_tra.transpose(1,2))
    valid_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, x_val.transpose(1,2))
    test_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, x_test.transpose(1,2))
    ##############get dataloader######################
    train_dataloader = data_loader(train_coeffs, x_tra_in, x_tra_out, y_tra, args.batch_size, shuffle=True, drop_last=True)

    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(valid_coeffs, x_val_in, x_val_out, y_val, args.batch_size, shuffle=False, drop_last=True)
        # val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=False, drop_last=True)
    # test_dataloader = data_loader(x_test, y_test, args.batch_size, shuffle=False, drop_last=False)
    test_dataloader = data_loader(test_coeffs, x_test_in, x_test_out, y_test, args.batch_size, shuffle=False, drop_last=False)
    return train_dataloader, val_dataloader, test_dataloader, scaler, times
