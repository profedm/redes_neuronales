import numpy as np
import lmdb
import caffe
from caffe.proto import caffe_pb2

def create_lmdb_dataset(data_path,X,Y):

    #Define map size
    map_size = X.nbytes * 100
    print map_size
    
    #Define data size
    N = X.shape[0]

    #Create lmdb for training data
    env = lmdb.open(data_path, map_size=map_size)

    with env.begin(write=True) as txn:
        # txn is a Transaction object
        for i in range(N):
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = X.shape[1]
            datum.height = X.shape[2]
            datum.width = X.shape[3]
            if X.dtype == np.int:
              datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9 (int type)
            elif X.dtype == np.float:
              datum.float_data.extend(X[i].flat)  #Float type
            if Y.dtype == np.int:
              datum.label = int(Y[i])
            elif Y.dtype == np.float:
              datum.label = Y[i]
            str_id = '{:08}'.format(i)

            # The encode is only essential in Python 3
            txn.put(str_id.encode('ascii'), datum.SerializeToString())

def load_lmdb_dataset(data_path, batch, channels, width, height):
    
    #Init data arrays
    x_train = np.zeros([batch,channels,width,height])
    y_train = np.zeros([batch,1])

    #Init lmdb arrays
    lmdb_env_val = lmdb.open(data_path)
    lmdb_txn_val = lmdb_env_val.begin()
    lmdb_cursor_val = lmdb_txn_val.cursor()
    datum_val = caffe_pb2.Datum()

    #Main extraction loop
    j=0
    for key, value in lmdb_cursor_val:
        datum_val.ParseFromString(value)
        y_train[j] = datum_val.label
        data_val = np.array(datum_val.float_data).astype(float).reshape( datum_val.channels, datum_val.height, datum_val.width)
        x_train[j] = data_val
        j = j+1
        
    return x_train, y_train
