# ReviewClassifier

The goal of this project was to infer sentiment (or polarity) from free form review text submitted for range of products.

In this project, I have implement a k-Nearest Neighbor Classifier to predict the sentiment for 14999 reviews for movies provided in the test file (test_file.txt)

Explaination of the Approach:
Amazon Review Classification is done using data preprocessing, vectorization
and k nearest neighbor algorithm. 

Reviews are tokenized after removing number and punctuations. The tokenized reviews are vectorized using terms frequency method.
K nearest neighbors are calculated using cosine similarity. The sentiment for testing reviews are calculated using sentiments of K nearest neighbors.
