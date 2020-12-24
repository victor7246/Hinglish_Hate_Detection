#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle
import operator
import numpy as np
import preprocessing as pp

global char_n_grams_index, word_n_grams_index, stance_top_hashtags, stance_top_hi_tokens, stance_top_en_tokens, stance_top_rest_tokens

def addchargramfeatures(feature_vector, char_n_grams_index, char_n_grams):
	char_features = [0] * len(char_n_grams_index)
	for char_i_gram in char_n_grams:
		if char_i_gram in char_n_grams_index:
			char_features[char_n_grams_index[char_i_gram]] = 1
	feature_vector.extend(char_features)
	return feature_vector

def addwordfeatures(feature_vector, word_n_grams_index, word_n_grams):
	word_features = [0] * len(word_n_grams_index)

	for word_i_gram in word_n_grams:
		if word_i_gram in word_n_grams_index:
			word_features[word_n_grams_index[word_i_gram]] = 1
	feature_vector.extend(word_features)
	return feature_vector

def addtoptokenfeatures(feature_vector, top_hi_tokens, top_en_tokens, top_rest_tokens, tweet):
	for i in xrange(len(top_hi_tokens)):
		if top_hi_tokens[i].lower() in tweet.lower():
			feature_vector.append(1)
		else:
			feature_vector.append(0)
	for i in xrange(len(top_en_tokens)):
		if top_en_tokens[i].lower() in tweet.lower():
			feature_vector.append(1)
		else:
			feature_vector.append(0)
	for i in xrange(len(top_rest_tokens)):
		if top_rest_tokens[i].lower() in tweet.lower():
			feature_vector.append(1)
		else:
			feature_vector.append(0)
	return feature_vector

def buildstancefeaturevector(key, tweet):
	global char_n_grams_index, word_n_grams_index, stance_top_hashtags, stance_top_hi_tokens, stance_top_en_tokens, stance_top_rest_tokens

	emoticons, hashtags, mentions, urls, char_n_grams, word_n_grams = pp.preprocess(key, tweet)

	stance_feature_vector = []

	stance_feature_vector = addchargramfeatures(stance_feature_vector, char_n_grams_index, char_n_grams)
	stance_feature_vector = addwordfeatures(stance_feature_vector, word_n_grams_index, word_n_grams)
	stance_feature_vector = addtoptokenfeatures(stance_feature_vector, stance_top_hi_tokens, stance_top_en_tokens, stance_top_rest_tokens, tweet)
	return stance_feature_vector

# Build feature vector for a given tweet.
# 1. Char n-grams (n=1-3).
# 2. Word n-grams (n=1-3).
# 3. Target tokens.
def getfeaturevector(key, tweet):
	global char_n_grams_index, word_n_grams_index, stance_top_hashtags, stance_top_hi_tokens, stance_top_en_tokens, stance_top_rest_tokens

	fp = open('data.txt', 'r')
	data = []
	for i in xrange(pickle.load(fp)):
		data.append(pickle.load(fp))

	char_n_grams_index, word_n_grams_index, stance_top_hashtags, stance_top_hi_tokens, stance_top_en_tokens, stance_top_rest_tokens = data
	stance_feature_vector = buildstancefeaturevector(key, tweet)
	
	return stance_feature_vector