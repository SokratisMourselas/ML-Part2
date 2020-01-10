# Machine Learning - 1st Assignment, Part 2
This is a GitHub repository for Part 2 of the first assignment of the lecture Applied Machine Learning


# Instructions on how to run code from terminal:
1. Download all the files from the repository
2. In order for the code to run successfully, the folders must remain as they are when downloaded
3. Open a terminal in the folder where the file part2code_c1961726.py is
4. Run the command:
```
python3 part2code_c1961726.py
```
5. If all the necessary packages are installed the code must run successfully and analytical information will be available in the terminal window regarding the flow of the algorithm.

# Functionalities/Parameters:
The aim of the algorithm is to create a machine learning model to perform a sentiment analysis on IMDb reviews. The classification is either a review being positive (1) or negative (0).
The three features that will be used are:
1. Word frequency
2. Frequency of phrases built with two words (ngrams)
3. Word count of text

The algorithm will then decide the best number of features by predicting the outcome of the development dataset and for this number it will show metrics on how good the model predicts the test dataset.

# Comments
1. To run this code, python 3 must be intalled in the system, along with some basic packages such as:
pandas, numpy, nltk, sklearn.

# Documentation

Part of the code used in the algorithm has been provided by Cardiff University for the lecture Applied Machine Learning and it has been modified to be applied to this question.

## Documetation for the dataset

Large Movie Review Dataset v1.0

### Overview

This dataset contains movie reviews along with their associated binary
sentiment polarity labels. It is intended to serve as a benchmark for
sentiment classification. This document outlines how the dataset was
gathered, and how to use the files provided. 

### Dataset 

The core dataset contains 25,000 reviews split into train, development
and test sets. The overall distribution of labels is roughly balanced.

### Files

Each folder contains files with negative (neg) and positive (reviews).
One review per line.

### Reference paper

@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
