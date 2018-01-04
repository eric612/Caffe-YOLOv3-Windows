# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import sys
import cv2
#import matplotlib.pyplot as plt
# display plots in this notebook

# set display defaults
#plt.rcParams['figure.figsize'] = (10, 10)        # large images
#plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
#plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
caffe_root = '../../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe
import math
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

caffe.set_mode_cpu()

model_def = './tiny-yolo.prototxt'
#model_weights = './gnet_yolo_region_darknet_anchor_iter_32000.caffemodel'
model_weights = './tiny-yolo.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
#mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
#mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
#print 'mean-subtracted values:', zip('BGR', mu)

mu = np.array([105/255.0, 117/255.0, 123/255.0])
# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 1.0)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          416, 416)  # image size is 227x227

def sigmoid(p):
    try:
        ans = math.exp(-p * 1.0)
    except OverflowError:
        ans = float('inf')
    return 1.0 / (1 + ans)



def overlap(x1, w1, x2, w2): #x1 ,x2 are two box center x
    left = max(x1 - w1 / 2.0, x2 - w2 / 2.0)
    right = min(x1 + w1 / 2.0, x2 + w2 / 2.0)
    return right - left

def cal_iou(box, truth):
    w = overlap(box[0], box[2], truth[0], truth[2])
    h = overlap(box[1], box[3], truth[1], truth[3])
    if w < 0 or h < 0:
        return 0
    inter_area = w * h
    union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area
    return inter_area * 1.0 / union_area

def apply_nms(boxes, thres):
    sorted_boxes = sorted(boxes,key=lambda d: d[7])[::-1]
    p = dict()
    for i in range(len(sorted_boxes)):
        if i in p:
            continue
        
        truth =  sorted_boxes[i]
        for j in range(i+1, len(sorted_boxes)):
            if j in p:
                continue
            box = sorted_boxes[j]
            iou = cal_iou(box, truth)
            if iou >= thres:
                p[j] = 1
    
    res = list()
    for i in range(len(sorted_boxes)):
        if i not in p:
            res.append(sorted_boxes[i])
    return res


def det(image, image_id, pic):
    transformed_image = transformer.preprocess('data', image)
    #plt.imshow(image)

    net.blobs['data'].data[...] = transformed_image

    ### perform classification
    output = net.forward()

    res = output['bn1'][0]  # the output probability vector for the first image in the batch
    print(res.shape)
    res2 = res.flatten()
    print(res2.shape)
    #file = open("Output.txt","w")
    np.savetxt('Output.txt', res2,fmt='%f')
    swap = np.zeros((13*13,5,10))
    swap2 = np.zeros((13*13,50))
    file = open("Output.txt","w")
    for w in range(416):       
        for h in range(416):       
            for c in range(16):              
                file.write("%f\n"%res[c][h][w])


pic = sys.argv[1]

image = caffe.io.load_image(pic)
det(image, '10001', pic)
print 'over'
'''
data_root = '/home/robot/workspace/SSD/data/';
index = 0;
for line in open('test.txt', 'r'):
	index += 1
	print index
	image_name = line.split(' ')[0]
	image_id = image_name.split('/')[-1]
	image = caffe.io.load_image(data_root + image_name)
	det(image, image_id)

'''
