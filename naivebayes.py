#!/usr/bin/env python
from __future__ import division
import nltk
from nltk.corpus import PlaintextCorpusReader
import os.path 
import sys
import pdb
from random import choice
from numpy import *

def initializeWordList(directory):
	"""Open text corpus reader, read in corpus of files and get initial wordlist."""
	basepath=os.path.dirname(__file__)
	corpus_root=os.path.abspath(os.path.join(basepath,directory))
	return PlaintextCorpusReader(corpus_root, '.*')

def getWordsFromCorpus(wordlists, fileid, possibleauthors):
	"""Get all words from a corpus and return as a set."""
	return set([word.lower() for word in wordlists.words(fileid) if word.isalpha() and word.lower() not in possibleauthors])

def getAuthors(wordlists):
	"""Gets authors for each corpus file and returns as a dictionary ({name of file:author})."""
	authors={}
	for item in wordlists.fileids():
		authors[item]=wordlists.words(item)[0].lower()
	return authors

def assignAuthorIDs(wordlists, authors, numauthors):
	"""Creates a dictionary of numerical ID values associated to each author ({'a':0, 'b':1})."""
	docIDs={}
	allauthors=sorted([i.lower() for i in set(authors.values())])
	for i in range(numauthors):
		docIDs[allauthors[i]]=i
	return docIDs

def initializeArrays(allwords, numauthors):
	"""Initialize arrays of all words in wordlists. Return as dictionary({'cat':array})."""
	wordarrays={}
	for word in allwords:
		wordarrays[word]=ones((numauthors, 2), dtype=float) #initialize to 1 for Laplace priors 
	return wordarrays

def getProbAuthors(numauthors,wordarrays):
	"""Return an array, containing the probability that a sample comes from a certain."""
	probauthors=zeros((numauthors,1),dtype=float)
	randomkey=choice(wordarrays.keys())
	randomarray=wordarrays[randomkey]
	for row in xrange(randomarray[1].shape[0]):
		probauthors[row]=sum(randomarray[1])
	sumall=sum(probauthors[:,0])
	for row in xrange(probauthors[0].shape[0]):
		probauthors[row]/=sumall
	return probauthors

def processDocuments(wordarrays, authorsdict, docIDs, allwords, wordlists):
	"""Process all documents into arrays."""
	for doc in wordlists.fileids():
		author=authorsdict[doc]
		docid=docIDs[author]
		wordsindoc=set([word.lower() for word in wordlists.words(doc) if word.isalpha() and 
				word.lower() not in docIDs.keys()])
		for word in allwords:
			if word in wordsindoc:
				wordarrays[word][docid][1]+=1.
			else:
				wordarrays[word][docid][0]+=1.

	totalinstances=len(authorsdict.keys())+2*len(docIDs)
	for word in allwords: #normalize entries of arrays
		for x in xrange(wordarrays[word].shape[0]):
			for y in xrange(wordarrays[word].shape[1]):
				wordarrays[word][x][y]/=totalinstances

def predictOutcomeDoc(allwords, wordsdoc, probauthorsarray, wordarrays, docIDs): 
	"""Compute probabilities for a given doc--'allwords' contains all encountered words, 'wordarrays' contains the probability of a given word,
	'probauthorsarray' contains the probability of an author, and 'wordsdoc' is a list containing the words
	of the doc."""
	authorprobdict={}
	for author in docIDs.keys():
		#initialize all dictionary values for each author to 1
		authorprobdict[author]=1. 

	for author in docIDs.keys():
		docID=docIDs[author]
		probauthor=probauthorsarray[docID]
		for word in wordsdoc:
			if word in allwords:
				authorprobdict[author]*=(wordarrays[word][docID][1]/probauthor)
			else:
				authorprobdict[author]*=(wordarrays[word][docID][0]/probauthor)
		authorprobdict[author]*=probauthor

	return max(authorprobdict,key=authorprobdict.get)

def getAllPredictions(testwordlist, allwordstrain, possibleauthors, probauthorsarray, wordarrays, docIDs):
	"""Get predictions for all documents in test corpus."""
	doctoprediction={} 
	for fileid in testwordlist.fileids():
		doctestwords=getWordsFromCorpus(testwordlist, fileid, possibleauthors)
		prediction=predictOutcomeDoc(allwordstrain,doctestwords, probauthorsarray, wordarrays, docIDs)
		doctoprediction[fileid]=prediction
	return doctoprediction

def calculateSuccessRate(doctoprediction, testwordlists):
	"""Calculate success rate."""
	numtestdocs=0.
	numsuccesses=0.
	for fileid in testwordlists.fileids():
		if testwordlists.words(fileid)[0].lower()==doctoprediction[fileid]:
			numsuccesses+=1.
		numtestdocs+=1.
	print "Success:", int(numsuccesses), "out of", int(numtestdocs), "documents. Rate:", (numsuccesses/numtestdocs)*100,"%"
	
if __name__=='__main__': 
	trainwordlists=initializeWordList('traincorpus')
	testwordlists=initializeWordList('testcorpus')
	possibleauthors=set(trainwordlists.words(fileid)[0].lower() for fileid in trainwordlists.fileids())

	allwordstrain=set([word.lower() for word in testwordlists.words() if word.isalpha() and word.lower() not in possibleauthors]+
			[word.lower() for word in trainwordlists.words() if word.isalpha() and word.lower() not in possibleauthors])
	trainauthorsdict=getAuthors(trainwordlists)
	numauthors=len(set(trainauthorsdict.values()))
	docIDs=assignAuthorIDs(trainwordlists, trainauthorsdict, numauthors)
	wordarrays=initializeArrays(allwordstrain, numauthors)
	processDocuments(wordarrays, trainauthorsdict, docIDs, allwordstrain, trainwordlists)
	probauthorsarray=getProbAuthors(numauthors,wordarrays)
	testauthorsdict=getAuthors(testwordlists)
	doctoprediction=getAllPredictions(testwordlists, allwordstrain, possibleauthors, probauthorsarray, wordarrays, docIDs)	
	calculateSuccessRate(doctoprediction,testwordlists)

	