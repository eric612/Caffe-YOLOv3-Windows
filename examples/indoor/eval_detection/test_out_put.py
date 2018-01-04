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

model_def = './yolo_out.prototxt'
#model_weights = './gnet_yolo_region_darknet_anchor_iter_32000.caffemodel'
#model_weights = './yolo_new.caffemodel'
model_weights = './yolo.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
#mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
#mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
#print 'mean-subtracted values:', zip('BGR', mu)

mu = np.array([105, 117, 123])
# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          416, 416)  # image size is 227x227

def det(image, image_id, pic):
	transformed_image = transformer.preprocess('data', image)
	#plt.imshow(image)

	net.blobs['data'].data[...] = transformed_image

	### perform classification
	output = net.forward()
	
	res = output['detection_out'][0]  # the output probability vector for the first image in the batch
	
	print res.shape
	index = 0
	box = []
	boxes = []
	for c in range(res.shape[0]):
		for h in range(res.shape[1]):			
			for w in range(res.shape[2]):
				box.append(res[c][h][w])
			boxes.append(box)
			box = []
	print boxes
	w = image.shape[1]
	h = image.shape[0]

	im = cv2.imread(pic)
	for box in boxes:
		left = (box[3]-box[5]/2.0) * w;
		right = (box[3]+box[5]/2.0) * w;
		top = (box[4]-box[6]/2.0) * h;
		bot = (box[4]+box[6]/2.0) * h;
		if left < 0:
			left = 0
		if right > w:
			right = w
		if top < 0:
			top = 0
		if bot > h:
			bot = h
		color = (255, 242, 35)
		cv2.rectangle(im,(int(left), int(top)),(int(right),int(bot)),color,3)
	
	cv2.imshow('src', im)
	cv2.waitKey()
	print 'det'
					


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
