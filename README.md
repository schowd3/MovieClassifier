# AmazonClassifier
a A k-Nearest Neighbor Classifier to predict the sentiment for 15000 reviews for movies provided in the test file

Amazon Review Classification is done using data preprocessing, vectorization
and k nearest neighbors. Reviews are tokenized after removing number and
punctuations. The tokenized reviews are vectorized using terms frequency method.
K nearest neighbors are calculated using cosine similarity. The sentiment for testing
reviews are calculated using sentiments of K nearest neighbor neighbors for training
data.
