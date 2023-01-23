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

--------------------------------------------NLTK: PART -2 --------------------------------------------------------

The Natural Language Toolkit (NLTK) is a Python library for working with human language data. One of the pre-trained models available in NLTK is the SentimentIntensityAnalyzer, which can be used to determine the sentiment (positive, negative, or neutral) of a given piece of text. It uses a combination of lexical heuristics and a pre-trained neural network to make its predictions. To use the SentimentIntensityAnalyzer, you will first need to install NLTK and then import the model. 

Precision(micro): 0.5298497267759563
Precision(macro): 0.5407337365543334
Precision(weighted): 0.7030271542285867
Recall: 0.5298497267759563
F1 score: 0.549293588567855

Confusion Matrix

[[4162 2806 2210]
[ 335 1562 1202]
[  70  260 2033]]
 
Short Summary:

It appears that the natural language processing (NLP) library NLTK is not accurately recognizing the sentiment of the text in the file. The main reason is not the model it's self, but the difficulty to make a decision ( NLTK gives an output of dictionary, for example: {'neg': 0.0, 'neu': 0.323, 'pos': 0.677, 'compound': 0.6369}
compound: a normalized score between -1 and 1, which indicates the overall sentiment of the text - sometimes we can have neutral 0.2, and neg 0.3, so is it neg or neutral? Now, we not always want to make specific desicion, for example in finance, when we want to make a sentiment analysis of the questions of analyst ask in earning calls. Maybe here we would like to have those 3 values in front of us, and then make a decision (hold, sell or buy). So this model will suit us well. But specifically in this project I want to recognize the sentiment of the text, where it will make a decision automatically - if it's positive, negative or neutral. Maybe because of the laziness, or maybe because this model doesn't fit, I would prefer to move to examine another models.
 
--------------------------------------------Flair part - 2 ---------------------------------------------------------

Flair - FLAIR is an open-source natural language processing library for state-of-the-art text classification, language modeling, and named entity recognition tasks. One of the key features of FLAIR is its ability to perform sentiment analysis, which is the process of determining the emotional tone of a piece of text. Sentiment analysis is often used to gauge public opinion on a particular topic or to measure the success of a marketing campaign. The library allows to train custom models and fine-tune pre-trained models on specific tasks and languages. It also provides a simple API to use these models in different programming languages.

On http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip it worked well: 
Accuracy: 0.700020625
Precision: 0.702104065282986
Recall: 0.69486625
F1 score: 0.6984664077906559
AUC score: 0.700020625
Confusion Matrix

[[564140 235860]
[244107 555893]]
 
![image](https://user-images.githubusercontent.com/53173112/213874540-c4ca0400-a717-4a48-8eef-b10bd998ec4f.png)
![image](https://user-images.githubusercontent.com/53173112/213874585-a15f1de2-ae6a-45e5-9840-84c5da34cc7a.png)

then I used Flair on other data (from https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment - I drop neutral sentiment):

Accuracy: 0.8680356988129279
Precision: 0.6328273244781784
Recall: 0.8468049090139653
F1 score: 0.7243438914027149
AUC score: 0.8601533806346793
Confusion Matrix 

[[8017 1161]
 [ 362 2001]]
 
![image](https://user-images.githubusercontent.com/53173112/213874687-d4f4630d-e564-4a0e-96ea-fd11c89f4d11.png)
![image](https://user-images.githubusercontent.com/53173112/213874716-122cb12f-737e-4885-9db6-f18c1429fe88.png)

Summary:

As we can see, the model (logistic regression)  that has been pre-trained on unique data gives better results, at the same time, on the new data (which is a little different, and as we would expect)  results which are not impressive. FLAIR, however, gave worse brings relatively good results on the first data set, and better results on the second data set. Apparently Flair, deals better with different types of data, the model is more holistic. However, when there is enough labeled data it is better to make your own model, it will bring much better results.

-------------------------------------------Zero-shot Sentiment Prediction Part - 3 ---------------------------------

Zero-shot Sentiment Prediction - Zero-shot Sentiment Prediction is a natural language processing technique that uses pre-trained models to classify the sentiment of a given text without any fine-tuning on a specific dataset. This is achieved by training models on a large amount of text data, which allows the model to learn general sentiment-related features that can be applied to a wide range of texts. The zero-shot sentiment prediction model provided by Hugging Face utilizes a transformer-based architecture and has been pre-trained on a large dataset of text from various sources, including social media and online reviews. This allows the model to understand the nuances of natural language and to accurately classify text as having a positive, negative, or neutral sentiment. The zero-shot sentiment prediction model is useful for applications such as analyzing customer reviews, gauging public opinion on social media, and monitoring brand reputation. It can be used to quickly classify large amounts of text data and can be integrated into various NLP pipelines.

On http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip it worked well: 

![image](https://user-images.githubusercontent.com/53173112/214004292-f350aca7-29c7-45b0-9a22-9992fca1b320.png)
![image](https://user-images.githubusercontent.com/53173112/214004374-d083f97d-2235-45c1-a2f5-43446d91d5a7.png)
![image](https://user-images.githubusercontent.com/53173112/214004424-81ef4990-9336-49b3-ada1-2f2c1505651f.png)
![image](https://user-images.githubusercontent.com/53173112/214004523-6c58223b-27f7-4ad1-bbf6-d8888449b94c.png)
It did better than flair 

But on data ( tweets about airlane companies) where it has also neutral label:
![image](https://user-images.githubusercontent.com/53173112/214004787-78388731-9b26-46c9-ac03-c3fe79d3a02b.png)
![image](https://user-images.githubusercontent.com/53173112/214004839-3293b67a-17e1-4e8f-a053-dc54262a71e5.png)
![image](https://user-images.githubusercontent.com/53173112/214004879-c72ce459-96e6-4095-96bb-bd2a67eda0d1.png)
(better to recognize negative sentiment)






