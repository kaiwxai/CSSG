import os
import numpy as np

def load_st_dataset(dataset):
    if dataset == 'NYTaxi':
        data_path = os.path.join('./data/NYTaxi/NYTaxi_flow.npy')
        data = np.load(data_path)
        matrix_wr_in = np.load('./data/NYTaxi/NYTaxi_od_matrix_in_walk_random.npy')
        matrix_wr_out = np.load('./data/NYTaxi/NYTaxi_od_matrix_out_walk_random.npy')
    else:
        data_path = os.path.join('./data/'+dataset+'/'+dataset+'_flow.npy')
        data = np.load(data_path)
        matrix_wr_in = np.load('./data/'+dataset+'/'+dataset+'_od_matrix_in_walk_random.npy')
        matrix_wr_out = np.load('./data/'+dataset+'/'+dataset+'_od_matrix_out_walk_random.npy')
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data, matrix_wr_in, matrix_wr_out
