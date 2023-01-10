# Opinion-mining
Opinion mining: Build a system that can identify opinions and emotions expressed in text ( Positive, Negative, Neutral )


Download the Sentiment140 dataset: The Sentiment140 dataset is available as a CSV file at the following URL: http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip  You can download this file and extract it to obtain the dataset.


About Dataset:
This is the sentiment140 dataset. It contains 1,600,000 tweets extracted using the twitter api . The tweets have been annotated (0 = negative, 2 = neutral, 4 = positive) and they can be used to detect sentiment .

It contains the following 6 fields:

target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
ids: The id of the tweet (for example 2087)
date: the date of the tweet (for example: Sat May 16 23:58:44 UTC 2009)
flag: The query (lyx). If there is no query, then this value is NO_QUERY.
user: the user that tweeted (for example: robotickilldozr)
text: the text of the tweet (for example: Lyx is cool)

Acknowledgements:
The official link regarding the dataset with resources about how it was generated is 
http://help.sentiment140.com/for-students 


Pakages to install:
1. nltk
2. pandas
3. sklearn

Methology:
1. Collect a dataset of texts annotated with opinions and emotions.
2. Preprocess the data: lowercasing, tokenization, and removing stop words.
3. Extract features from the text. Some potential features for this task might include word n-grams.
4. Train a classifier on the extracted features: logistic regression, neural network.
5. Evaluate the classifier on a held-out test set. Measure the performance of the classifier using metrics such as precision, recall, and F1 score.
