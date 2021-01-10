#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import pickle
import operator
import numpy as np
from extract_tweets import get_tweet_map, get_id_stance_map
from build_feature_vector import getfeaturevector
from feature_properties import findfeatureproperties
from sklearn import svm, tree
# from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel, f_classif
from sklearn.neural_network import MLPClassifier

def featureselection(features, train_tweets, train_truth):
	model = SelectKBest(score_func=chi2, k=500)
	fit = model.fit(np.array(train_tweets), np.array(train_truth))
	return fit.transform(np.array(features)).tolist()

def tenfoldcrossvalidation(feature_map, id_stance_map, index, id_tweet_map):
	feature_map = dict(sorted(feature_map.items(), key=operator.itemgetter(1)))

	tweets = []
	truth = []
	keys = []

	for key, feature in feature_map.iteritems():
		tweets.append(feature)
		truth.append(index[id_stance_map[key]])
		keys.append(key)

	accuracy = 0.0
	for i in xrange(10):
		tenth = len(tweets)/10
		start = i*tenth
		end = (i+1)*tenth
		test_index = xrange(start,end)
		train_index = [i for i in range(len(tweets)) if i not in test_index]
		train_tweets = []
		train_keys = []
		test_tweets = []
		test_keys = []
		train_truth = []
		test_truth = []
		
		for i in xrange(len(tweets)):
			if i in train_index:
				train_tweets.append(tweets[i])
				train_truth.append(truth[i])
				train_keys.append(keys[i])
			else:
				test_tweets.append(tweets[i])
				test_truth.append(truth[i])
				test_keys.append(keys[i])

		new_train_tweets = featureselection(train_tweets, train_tweets, train_truth)
		new_test_tweets = featureselection(test_tweets, train_tweets, train_truth)

		if sys.argv[1] == "rbfsvm":
			print "RBF kernel SVM"
			clf = svm.SVC(kernel='rbf', C=1000, gamma=0.0001)
			clf.fit(np.array(new_train_tweets), np.array(train_truth))
			test_predicted = clf.predict(np.array(new_test_tweets))

		elif sys.argv[1] == "randomforest":
		# # Using Random forest for classification.
			print 'Random forest'
			clf = RandomForestClassifier(n_estimators=10, max_depth=None)
			clf.fit(np.array(new_train_tweets), np.array(train_truth))
			test_predicted = clf.predict(np.array(new_test_tweets))
			# getaccuracy(test_predicted, test_truth)
		elif sys.argv[1] == "linersvm":
		# # Using Linear svm for classification.
			print 'Linear SVM'
			clf = svm.LinearSVC()
			clf.fit(np.array(new_train_tweets), np.array(train_truth))
			test_predicted = clf.predict(np.array(new_test_tweets))
		# getaccuracy(test_predicted, test_truth)

		accuracy += getaccuracy(test_predicted, test_truth)
	print "Accuracy:"
	print accuracy/10.0

def getfeaturevectorforalltweets():
	id_tweet_map, tweet_id_map = get_tweet_map()
	# print len(id_tweet_map)
	id_tweet_map = dict(sorted(id_tweet_map.items(), key=operator.itemgetter(0)))
	
	train_stance_feature_map = {}

	count = 1
	for key, tweet in id_tweet_map.iteritems():
		stance_feature_vector = getfeaturevector(key, tweet)
		
		train_stance_feature_map[key] = stance_feature_vector
		# print count
		count += 1

	return train_stance_feature_map

def getaccuracy(test_predicted, test_truth):
	count = 0
	for j in xrange(len(test_truth)):
		if test_truth[j] == test_predicted[j]:
			count += 1
	# print len(test_truth)
	# print count
	return float(float(count*100)/float(len(test_truth)))

def train_and_test():
	findfeatureproperties()
	id_stance_map = get_id_stance_map()

	train_stance_feature_map = getfeaturevectorforalltweets()
	
	stance_index = {'FAVOR': 0, 'AGAINST': 1, 'NONE': 2, 0: 'FAVOR', 1: 'AGAINST', 2: 'NONE'}

	id_tweet_map = get_tweet_map()

	tenfoldcrossvalidation(train_stance_feature_map, id_stance_map, stance_index, id_tweet_map)

# getfeaturevectorforalltweets()
train_and_test()