# Opinion-mining
Opinion mining: Build a system that can identify opinions and emotions expressed in text ( Positive, Negative, Neutral )


Download the Sentiment140 dataset: The Sentiment140 dataset is available as a CSV file at the following URL: http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip  You can download this file and extract it to obtain the dataset.


About Dataset:
This is the sentiment140 dataset. It contains 1,600,000 tweets extracted using the twitter api . The tweets have been annotated (0 = negative, 2 = neutral, 4 = positive) and they can be used to detect sentiment .

It contains the following 6 fields:

target: the polarity of the tweet (0 = negative, 2 = neutral ( no neutral in this data) , 4 = positive)
ids: The id of the tweet (for example 2087)
date: the date of the tweet (for example: Sat May 16 23:58:44 UTC 2009)
flag: The query (lyx). If there is no query, then this value is NO_QUERY.
user: the user that tweeted (for example: robotickilldozr)
text: the text of the tweet (for example: Lyx is cool)

Acknowledgements:
The official link regarding the dataset with resources about how it was generated is 
http://help.sentiment140.com/for-students 



Methology:
1. Collect a dataset of texts annotated with opinions and emotions.
2. Preprocess the data: lowercasing, tokenization, removing stop words and punctuation. It's important to note, that I test some preprocessing options, with and without removing stop words and punctuation,with and without stemming, in order to understant the impact on sentiment analysis. 
4. Extract features from the text. Some potential features for this task might include word n-grams.
5. Train a classifier on the extracted features: logistic regression (part 1) in a part 2 I tried to use (Flair*) in a part 3 I tried to use Zero-shot Sentiment Prediction. 
 
7. Evaluate the classifier on a held-out test set. Measure the performance of the classifier using metrics such as precision, recall, and F1 score (relevant only for Logistic Regression). Evaluation of Flair and Zero-shot - I did seperatilly. 

Flair - FLAIR is an open-source natural language processing library for state-of-the-art text classification, language modeling, and named entity recognition tasks. One of the key features of FLAIR is its ability to perform sentiment analysis, which is the process of determining the emotional tone of a piece of text. Sentiment analysis is often used to gauge public opinion on a particular topic or to measure the success of a marketing campaign. The library allows to train custom models and fine-tune pre-trained models on specific tasks and languages. It also provides a simple API to use these models in different programming languages.

Zero-shot Sentiment Prediction - Zero-shot Sentiment Prediction is a natural language processing technique that uses pre-trained models to classify the sentiment of a given text without any fine-tuning on a specific dataset. This is achieved by training models on a large amount of text data, which allows the model to learn general sentiment-related features that can be applied to a wide range of texts. The zero-shot sentiment prediction model provided by Hugging Face utilizes a transformer-based architecture and has been pre-trained on a large dataset of text from various sources, including social media and online reviews. This allows the model to understand the nuances of natural language and to accurately classify text as having a positive, negative, or neutral sentiment. The zero-shot sentiment prediction model is useful for applications such as analyzing customer reviews, gauging public opinion on social media, and monitoring brand reputation. It can be used to quickly classify large amounts of text data and can be integrated into various NLP pipelines.
