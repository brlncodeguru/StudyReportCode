
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16

from collections import OrderedDict

import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial.distance as distance

from numpy import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.spatial.distance as distance


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.spatial.distance as distance

import cv2

from matplotlib import pyplot as plt

# import the necessary packages
from skimage import feature

import numpy as np

import pandas as pd

import argparse
import glob





class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius
 
	def describe(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))
 
		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)
 
		# return the histogram of Local Binary Patterns
		return hist



def dist_euclidean(elem1, elem2):
    t_sum=0
    for i in range(len(elem1)):
        for j in range(len(elem1[0])):
            t_sum+= np.square(elem1[i][j]-elem2[i][j])
    return np.sqrt(t_sum)

def return_intersection(hist_1, hist_2):
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection

def chi2_distance(histA, histB, eps = 1e-10):
	# compute the chi-squared distance
	d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
		for (a, b) in zip(histA, histB)])
 
	# return the chi-squared distance
	return d


def key_calc_score(input_file_path,filename):
    print "file to compare ",input_file_path
    print "file  compare ",filename

    #print "file comparing",filename.split(':')[1]
    
    compare_path = filename
    model = ResNet50(weights='imagenet')
    img_path = input_file_path
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    res_features = model.predict(x)
    #print res_features
    #print np.shape(res_features)
    img_path = compare_path
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    res_features1 = model.predict(x)
    #print res_features1
    #print np.shape(res_features1)
    resnet_score = np.linalg.norm(res_features - res_features1, axis=1)
    resnet_score_cosine = distance.cosine(res_features,res_features1)
    print 'ResNet50 ',resnet_score
    print 'ResNet50-Cosine ',distance.cosine(res_features,res_features1)

    model = VGG16(weights='imagenet')
    img_path = input_file_path
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    vgg_features = model.predict(x)
    img_path1 = compare_path
    img1 = image.load_img(img_path1, target_size=(224, 224))
    x = image.img_to_array(img1)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    vgg_features1 = model.predict(x)
    #print np.shape(vgg_features)
    #print np.shape(vgg_features1)
    vggnet_score = np.linalg.norm(vgg_features - vgg_features1, axis=1)
    vggnet_score_cosine = distance.cosine(vgg_features,vgg_features1)
    print 'VGGNET ',vggnet_score
    print 'VGGNET-Cosine ',distance.cosine(vgg_features,vgg_features1)
    img = cv2.imread(input_file_path, 1)
    img1 = cv2.imread(compare_path,1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hist,bins = np.histogram(hsv.ravel(),256,[0,256])
    hist = np.asmatrix(hist)
    #print np.shape(hist)
    hist = cv2.calcHist([hsv], [0], None, [256],
        [0, 256])
    hist = cv2.normalize(hist,0).flatten()
    hist1 = cv2.calcHist([hsv1], [0], None, [256],
        [0, 256])
    hist1 = cv2.normalize(hist,0).flatten()

    # print  'ColorHist ',distance.cosine(hist, hist1)
    # colorhist_score = abs(return_intersection(hist,hist1)-1)
    colorhist_score_chisquared = cv2.compareHist(hist, hist1, cv2.HISTCMP_CHISQR)
    lbp_grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp_grayimage1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    desc = LocalBinaryPatterns(24, 8)
    hist = desc.describe(lbp_grayimage)
    #print np.shape(hist)
    hist1 = desc.describe(lbp_grayimage1)
    #print np.shape(hist)
    lbp_score = abs(return_intersection(hist,hist1)-1)
    lbp_cosine = distance.cosine(hist,hist1)
    print 'LBPCompare ',lbp_score
    print 'LBPCompare-Cosine ',distance.cosine(hist,hist1)
    grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayimage1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    shape_desc=cv2.HuMoments(cv2.moments(grayimage)).flatten()
    shape_desc1=cv2.HuMoments(cv2.moments(grayimage1)).flatten()
    shape_desc = np.asmatrix(shape_desc)
    shape_desc1 = np.asmatrix(shape_desc1)
    #print np.shape(shape_desc)
    #print np.shape(shape_desc1)
    #print return_intersection(shape_desc,shape_desc1)
    shape_score = distance.cosine(shape_desc, shape_desc1)
    print 'Shape-Cosine ',shape_score
    scores = []
    scores.append(resnet_score_cosine)
    scores.append(colorhist_score_chisquared)
    scores.append(lbp_cosine)
    scores.append(shape_score)
    scores = np.asmatrix(scores)
    print "Scores :",scores
    x = scores
    scores = (x - mean(x))/std(x)
    print "Scores after norm:",scores
    print np.shape(scores)
    weights = np.matrix('0.5;0.16;0.16;0.16')
    print np.shape(weights)
    print 'weights',weights
    print 'sum of weights',np.sum(weights)
    prod = (scores *weights)/np.sum(weights)
    print 'product',prod
    nn_rank_inv = 0
    color_rank_inv = 0
    lbp_cosine_rank_inv = 0
    shape_rank_inv = 0
    if resnet_score_cosine !=0 :
        nn_rank_inv = (1/resnet_score_cosine)
    if colorhist_score_chisquared !=0 :
        color_rank_inv = (1/colorhist_score_chisquared)
    if lbp_cosine != 0 :
        lbp_cosine_rank_inv = (1/lbp_cosine)
    if shape_score != 0:
        shape_rank_inv =  (1/shape_score)
                
            


    

    sum_inverse = nn_rank_inv + color_rank_inv + lbp_cosine_rank_inv + shape_rank_inv
    irp_score = 1/sum_inverse
    print 'IRP_Score',irp_score
    prod = abs(prod.item(0))
    print 'prod',prod

    return {'filename':filename,'ResNet' : resnet_score_cosine,'Color':colorhist_score_chisquared,'LBP':lbp_cosine,'Shape':shape_score,'IRP':irp_score,'Prod':prod}


resnet_score = []
resnet_score_cosine = []
vggnet_score = []
vggnet_score_cosine = []
colorhist_score = []
lbp_score = []
shape_score = []
results =[]



# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required = True,
#     help = "/Users/lakshminarayanabr/Downloads/new_queryset_allinone/short/")
# args = vars(ap.parse_args())



input_file_path = '/Users/lakshminarayanabr/Downloads/new_queryset_allinone/short/1-1.jpg'



for imagePath in glob.glob("/Users/lakshminarayanabr/Downloads/new_queryset_allinone/short/*.jpg"):
    # extract the image filename (assumed to be unique) and
    # load the image, updating the images dictionary
    filename = imagePath[imagePath.rfind("/") + 1:]
    filename = '/Users/lakshminarayanabr/Downloads/new_queryset_allinone/short/'+filename
    results.append(key_calc_score(input_file_path,filename))



     





my_df = pd.DataFrame(results)

my_df.to_csv('res-1.csv', float_format='%g',index=False, header=False)

  
		


