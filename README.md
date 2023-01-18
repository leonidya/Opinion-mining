In this small project, my goal was to explore different models and techniques in order to achieve better results for sentiment analysis (also known as opinion mining). The task of sentiment analysis is to identify the opinions and emotions expressed in text, specifically in this case, determining whether the sentiment is positive or negative.

In the code provided, you will find the results of using different models, along with detailed comments and explanations. The code is divided into different sections:
1. Logistic regresion - model was trained and evaluated on the dataset, and the results of using this model are presented in the code.
2. NLTK: pretrained "SentimentIntensityAnalyzer"
3. Flair
4. Zero-shot Sentiment Prediction

------------------------------Logistic Regresion for Sentiment Analysis - Part 1 ---------------------------------------

Dataset: The Sentiment140 dataset is available as a CSV file at the following URL: http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip  You can download this file and extract it to obtain the dataset.

About Dataset:
This is the sentiment140 dataset. It contains 1,600,000 tweets extracted using the twitter api . The tweets have been annotated (0 = negative, 4 = positive) and they can be used to detect sentiment .

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
3. Extract features from the text. Some potential features for this task might include word n-grams.
4. Train a classifier on the extracted features: logistic regression (part 1). 
5. Evaluate the classifier on a held-out test set. Measure the performance of the classifier using metrics such as precision, recall, and F1 score.

Results:

On http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip it worked very well:
A precision was 0.78 which means that the model makes very few false positive predictions (i.e. it correctly identifies a large proportion of positive cases). Recall is the proportion of true positive predictions among all actual positive cases. A recall was 0.81 which means that the model is identifying most of the actual positive cases. F1 score is the harmonic mean of precision and recall, and it combines both metrics into a single number. A F1 was 0.80 which indicates that the model has a good balance of precision and recall. In this case, the AUC score is 0.79, which means that the model is performing well. It is correctly distinguishing the positive class from the negative class with a good level of accuracy. 

![image](https://user-images.githubusercontent.com/53173112/213147178-b8e2590c-a026-460c-b4e4-2aa7caf73d8a.png)

But then I used the same model on other data (from https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment - I drop neutral sentiment):
In my opnion, after some manual review of the twitter comments, sometimes it's hard to say if it's positive or not, they are more neutral. 

Accuracy: 0.7933454639979205
Precision: 0.49736716132120634
Recall: 0.8793906051629284
F1 score: 0.6353768536920961

![image](https://user-images.githubusercontent.com/53173112/213147017-00a5c762-48b3-4324-ac69-12ab1842f593.png)

According to the results, the model seems to be better on recognizing negative sentiment. 

This model may not be the best choice for this dataset. So, I tried other models/techniques to improve the results.

--------------------------------------------Flair part - 2 ---------------------------------------------------------

Flair - FLAIR is an open-source natural language processing library for state-of-the-art text classification, language modeling, and named entity recognition tasks. One of the key features of FLAIR is its ability to perform sentiment analysis, which is the process of determining the emotional tone of a piece of text. Sentiment analysis is often used to gauge public opinion on a particular topic or to measure the success of a marketing campaign. The library allows to train custom models and fine-tune pre-trained models on specific tasks and languages. It also provides a simple API to use these models in different programming languages.

-------------------------------------------Zero-shot Sentiment Prediction Part - 3 ---------------------------------

Zero-shot Sentiment Prediction - Zero-shot Sentiment Prediction is a natural language processing technique that uses pre-trained models to classify the sentiment of a given text without any fine-tuning on a specific dataset. This is achieved by training models on a large amount of text data, which allows the model to learn general sentiment-related features that can be applied to a wide range of texts. The zero-shot sentiment prediction model provided by Hugging Face utilizes a transformer-based architecture and has been pre-trained on a large dataset of text from various sources, including social media and online reviews. This allows the model to understand the nuances of natural language and to accurately classify text as having a positive, negative, or neutral sentiment. The zero-shot sentiment prediction model is useful for applications such as analyzing customer reviews, gauging public opinion on social media, and monitoring brand reputation. It can be used to quickly classify large amounts of text data and can be integrated into various NLP pipelines.
