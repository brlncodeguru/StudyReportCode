from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import cv2

import pandas as pd

import scipy.spatial.distance as distance




 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "/Users/lakshminarayanabr/Downloads/new_queryset_allinone/short/")
args = vars(ap.parse_args())
 
# initialize the index dictionary to store the image name
# and corresponding histograms and the images dictionary
# to store the images themselves
index = {}
images = {}


for imagePath in glob.glob(args["dataset"] + "/*.jpg"):
	# extract the image filename (assumed to be unique) and
	# load the image, updating the images dictionary
	filename = imagePath[imagePath.rfind("/") + 1:]
	image = cv2.imread(imagePath)
	images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
 
	# extract a 3D RGB color histogram from the image,
	# using 8 bins per channel, normalize, and update
	# the index
	hist = cv2.calcHist([image], [0], None, [256],
		[0, 256])
	hist = cv2.normalize(hist,0).flatten()
	index[filename] = hist


OPENCV_METHODS = (
	("Correlation", cv2.HISTCMP_CORREL),
	("Chi-Squared", cv2.HISTCMP_CHISQR),
	("Intersection", cv2.HISTCMP_INTERSECT), 
	("Hellinger", cv2.HISTCMP_BHATTACHARYYA),
    ("Cosine",cv2.HISTCMP_INTERSECT))
 
# loop over the comparison methods
for (methodName, method) in OPENCV_METHODS:
	# initialize the results dictionary and the sort
	# direction
	results = {}
	reverse = False
 
	# if we are using the correlation or intersection
	# method, then sort the results in reverse order
	if methodName in ("Correlation", "Intersection"):
		reverse = True
	for (k, hist) in index.items():
		# compute the distance between the two histograms
		# using the method and update the results dictionary
		d = cv2.compareHist(index["1-5.jpg"], hist, method)
		if methodName in ("Cosine"):
			d = distance.cosine(index["1-5.jpg"],hist)
		results[k] = d

	results = sorted([(v, k) for (k, v) in results.items()], reverse = reverse)

	print "method ",methodName
	print "results",results

	my_df = pd.DataFrame(results)

	my_df.to_csv(methodName+'.csv', index=False, header=False)


	




