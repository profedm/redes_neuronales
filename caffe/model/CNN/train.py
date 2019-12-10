#Import libraries
import numpy as np
import lmdb
import caffe
from caffe.proto import caffe_pb2
from pylab import *
import cv2

#Main path
#net_root = '/home/edgar/caffe/models/stixel_net/'
net_root = '/home/edgar/Caffe_course/example2/model/'

#Processing mode
caffe.set_mode_cpu()


#Load model and solver
modelo = net_root + 'CNN_train.prototxt'
solver = net_root + 'solver.prototxt'

#Init solver
caffe_solver = caffe.get_solver(solver)

#Process solver
caffe_solver.solve()

