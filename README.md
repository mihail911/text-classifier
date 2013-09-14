Text-Classifier
=============================
***Background***

This is a text-classifier program that uses supervised learning to characterize the authors of a corpus of documents. It utilizes the Naive Bayes Classifier, a very simple machine learning algorithm, to make its predictions. More information about the classifier can be found here: http://en.wikipedia.org/wiki/Naive_Bayes_classifier.
Given that I have not tested the program on a very diverse or large training/testing dataset, I have yet to get good figures on the accuracy of the classifier as I have implemented it. 

***Use***

The program makes uses of the Python natural language toolkit as well as the ```numpy``` module. Instructions for installing both of them on various platforms can be found here: http://nltk.org/install.html.
Place the ```naivebayes.py``` file in a directory. In the same directory, place two subdirectories: ```traincorpus``` containing the training set to be used by the program and ```testcorpus``` containing the documents you wish to classify. The program as I have written it necessitates that all the documents in the training/testing corpuses have the author of each document as the first word of the document. The testing corpuses must have authors included so that the accuracy of the classifier can be calculated. 
Run ```./naivebayes.py``` and the classifier will execute. 