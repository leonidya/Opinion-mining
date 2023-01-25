In this project, the objective was to investigate various models and techniques with the aim of enhancing the outcomes of sentiment analysis, also referred to as opinion mining. The objective of sentiment analysis is to discern the emotions and opinions articulated in written text, specifically determining if the sentiment is positive or negative in this instance.

The code I provided is divided into different sections that cover various models and techniques for sentiment analysis.

1. **Logistic Regression** 
2. **Singular Value Decomposition (SVD)** in combination with **Logistic Regression**.
3. **SVD in combination with XGBoost**.

The next sections utilize **GloVe** (Global Vectors for Word Representation) which is a pre-trained word embedding model, to represent words as numerical vectors:

4. **pretrained GloVe - Logistic Regression model**.
5. **GloVe - XGBoost**.

The final sections use pretrained models:

6. **NLTK** (Natural Language Toolkit) library's pretrained "SentimentIntensityAnalyzer"
7. **Flair** 
8. **Zero-Shot Sentiment Prediction**
9. **Textblob library**

The code also includes detailed comments and explanations to guide the understanding of the methodology and results of each section.

## WHY I DID IT?
As we all know, Sentiment analysis is an important task in natural language processing because it allows for the automatic identification of opinions and emotions expressed in text. This can be useful in a wide range of applications such as social media analysis, customer feedback analysis, and opinion mining.

By exploring different models and techniques for sentiment analysis on the Sentiment140 dataset, I can gain insight into the performance of different approaches and understand which methods may be more effective for different types of data. Additionally, understanding the performance on a benchmark dataset such as Sentiment140, can give me an intuition about how well a model might perform on other datasets. By conducting this analysis, I can gain a deeper understanding of the capabilities and limitations of different models and techniques for sentiment analysis and make informed decisions about which methods to use for specific tasks and data types.

It's important to note, that that the performance of a sentiment analysis model can vary depending on the dataset it is trained on. For example, financial texts may have a different vocabulary and language structure compared to tweets, and thus a model trained on the Sentiment140 dataset may not perform as well on financial texts.

## Data

The Sentiment140 dataset is a large dataset of tweets that have been annotated with sentiment labels (0 = negative, 4 = positive). The dataset contains 1.6 million tweets and is available for download as a CSV file at the URL http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip. The dataset contains 6 fields:

**target**: the polarity of the tweet (0 = negative, 2 = neutral ( no neutral in this data) , 4 = positive)

**ids**: The id of the tweet (for example 2087)

**date**: the date of the tweet (for example: Sat May 16 23:58:44 UTC 2009)

**flag**: The query (lyx). If there is no query, then this value is NO_QUERY.

**user**: the user that tweeted (for example: robotickilldozr)

**text**: the text of the tweet (for example: Lyx is cool)

It was extracted using the twitter API and can be used for sentiment analysis. The official website for the dataset with resources about how it was generated is http://help.sentiment140.com/for-students.

Methology:
1. Collect a dataset of texts annotated with opinions and emotions.
2. Preprocess the data: lowercasing, tokenization, removing stop words and punctuation. It's important to note, that I test some preprocessing options, with and without removing stop words and punctuation,with and without stemming, in order to understant the impact on sentiment analysis. 
3. Extract features from the text. Some potential features for this task might include word n-grams.
4. Train a classifier on the extracted features: for example - logistic regression (part 1). 
5. Evaluate the classifier on a held-out test set. Measure the performance of the classifier using metrics such as precision, recall, F1 score, ctr.

```diff
## Part 1: + **Logistic Regresion for Sentiment Analysis **```

Results:

It looks like model performed well on the Sentiment140 dataset. The precision of 0.78 means that the model makes very few false positive predictions (i.e. it correctly identifies a large proportion of positive cases). Recall of 0.81 indicates that the model is identifying most of the actual positive cases. The F1 score of 0.80 is a harmonic mean of precision and recall, and it combines both metrics into a single number, it indicates that the model has a good balance of precision and recall.

Additionally, the AUC score of 0.79 indicates that the model is performing well. AUC (Area Under the Receiver Operating Characteristic Curve) is a metric used to evaluate binary classification models. It measures the ability of the model to distinguish between the positive and negative classes. A score of 0.79 means that the model is correctly distinguishing the positive class from the negative class with a good level of accuracy.

Overall, these results suggest that the Logistic Regresion has a good performance on the Sentiment140 dataset, with a good balance of precision and recall, and a high accuracy in classifying tweets as positive or negative.

![image](https://user-images.githubusercontent.com/53173112/213147178-b8e2590c-a026-460c-b4e4-2aa7caf73d8a.png)

**BUT** model's performance on the new dataset, which is sourced from https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment, was not as good as it was on the Sentiment140 dataset. The accuracy, precision, recall and F1 score values are lower than the previous one, and it seems that the model is better at recognizing negative sentiment than positive sentiment. This new dataset may have different characteristics than the Sentiment140 dataset, such as different vocabulary and language structure, which may have affected the model's performance. Additionally,  some comments in the new dataset are more neutral, which may have made it harder for the model to classify them as positive or negative.

Accuracy: 0.7933454639979205

Precision: 0.49736716132120634

Recall: 0.8793906051629284

F1 score: 0.6353768536920961

![image](https://user-images.githubusercontent.com/53173112/213147017-00a5c762-48b3-4324-ac69-12ab1842f593.png)

This model may not be the best choice for the new dataset. So, I tried other models/techniques to improve the results.

## Part 2: SVD - Logistic Regresion

In order to improve the performance of the model, I tried using SVD (Singular Value Decomposition) in combination with logistic regression. However, the results were not as good as the Logistic regression alone. SVD is a technique used to reduce the dimensionality of the feature space by removing noise or redundant features. While it can improve the accuracy of the model in some cases, it can also decrease the accuracy of the model by removing important information.


Results:
![image](https://user-images.githubusercontent.com/53173112/214265315-340e4879-8983-4f72-a93b-c5bd298d539b.png)
![image](https://user-images.githubusercontent.com/53173112/214265399-0cc63184-2d61-41fa-9620-594a417d6999.png)
![image](https://user-images.githubusercontent.com/53173112/214265458-5e3e593b-99ef-44b0-b0cf-c69df2fe909f.png)
![image](https://user-images.githubusercontent.com/53173112/214265605-f83497f8-b375-494c-b701-87aa03bd9b70.png)

Using SVD didn't improve the performance of the model as much as it was expected, but in some cases, it can be a useful technique in improving the accuracy of the model by reducing the dimensionality of the feature space. So, It looks like Logistic regresion is doing better, and yes, SVD can decrease the accuracy of the model by removing important information. BUT, in some cases, the use of SVD can improve the accuracy of the model by reducing the dimensionality of the feature space and removing noise or redundant features (it worth to check even if it takes time). 

-----------------------------------------SVD - Xgboost Part - 3 ---------------------------------------

Results:
![image](https://user-images.githubusercontent.com/53173112/214273179-d1a3df84-dfbd-4dd8-9007-c810025dd49d.png)
![image](https://user-images.githubusercontent.com/53173112/214273331-5440dedb-4a28-491c-8377-624bdf9013ee.png)
![image](https://user-images.githubusercontent.com/53173112/214273360-c47133c2-c066-410b-aa1f-4935828a926d.png)
![image](https://user-images.githubusercontent.com/53173112/214273422-18d549a3-1443-42af-9730-33c515c8583d.png)

Summary: it's look like it's doing better than SVD - Logistic regresion.


-------------------------------------------Pretrained Glove embedings with Logit model - 4---------------------------------------------------


results:

![image](https://user-images.githubusercontent.com/53173112/214481533-3ff1dd4c-6d41-43ae-99b1-8fbbe6324104.png)
![image](https://user-images.githubusercontent.com/53173112/214481581-01309141-f795-464f-b660-e383154f55ca.png)
![image](https://user-images.githubusercontent.com/53173112/214481617-2a444a3a-76ea-4c20-849e-5cd0bfb36738.png)
![image](https://user-images.githubusercontent.com/53173112/214481653-5175f4ec-6fd4-4eed-9660-052944169054.png)

Summary:
Pretrained Glove embedings with Logit model until now is giving the best results. 

-------------------------------------------Glove + XGBoost 5---------------------------------------------------
results:
![image](https://user-images.githubusercontent.com/53173112/214512569-6733d7af-9a80-4c86-9773-125a86d1bae8.png)
![image](https://user-images.githubusercontent.com/53173112/214512617-4c52bc3a-25d0-4afa-be3e-2807b22c3375.png)
![image](https://user-images.githubusercontent.com/53173112/214512654-73cba528-9eb5-4e72-bb5f-c71c2f5c01f9.png)
![image](https://user-images.githubusercontent.com/53173112/214512704-674457da-ddc9-4f69-9e7e-09dc6b942e52.png)
![image](https://user-images.githubusercontent.com/53173112/214512787-dcd83727-5cc1-4fae-8d0d-d7e091fc5a8a.png)
Summary:
Glove + XGBoost 5 - gives us little bit better results than Glove embedings with Logit model.

--------------------------------------------NLTK: PART - 5 --------------------------------------------------------

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
 
--------------------------------------------Flair part - 6 ---------------------------------------------------------

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

-------------------------------------------Zero-shot Sentiment Prediction Part - 7 ---------------------------------

Zero-shot Sentiment Prediction - Zero-shot Sentiment Prediction is a natural language processing technique that uses pre-trained models to classify the sentiment of a given text without any fine-tuning on a specific dataset. This is achieved by training models on a large amount of text data, which allows the model to learn general sentiment-related features that can be applied to a wide range of texts. The zero-shot sentiment prediction model provided by Hugging Face utilizes a transformer-based architecture and has been pre-trained on a large dataset of text from various sources, including social media and online reviews. This allows the model to understand the nuances of natural language and to accurately classify text as having a positive, negative, or neutral sentiment. The zero-shot sentiment prediction model is useful for applications such as analyzing customer reviews, gauging public opinion on social media, and monitoring brand reputation. It can be used to quickly classify large amounts of text data and can be integrated into various NLP pipelines.

On http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip it worked well: 

![image](https://user-images.githubusercontent.com/53173112/214004292-f350aca7-29c7-45b0-9a22-9992fca1b320.png)
![image](https://user-images.githubusercontent.com/53173112/214004374-d083f97d-2235-45c1-a2f5-43446d91d5a7.png)
![image](https://user-images.githubusercontent.com/53173112/214004424-81ef4990-9336-49b3-ada1-2f2c1505651f.png)
![image](https://user-images.githubusercontent.com/53173112/214004523-6c58223b-27f7-4ad1-bbf6-d8888449b94c.png)

It did better than flair - but it's hard to compare - I runned Flair on all the data, BUT Zero-shot Sentiment Prediction took a lot of time (~1.5days). So i decided to run in on a sample. 

But on data ( tweets about airlane companies) where it has also neutral label:
![image](https://user-images.githubusercontent.com/53173112/214004787-78388731-9b26-46c9-ac03-c3fe79d3a02b.png)
![image](https://user-images.githubusercontent.com/53173112/214004839-3293b67a-17e1-4e8f-a053-dc54262a71e5.png)
![image](https://user-images.githubusercontent.com/53173112/214004879-c72ce459-96e6-4095-96bb-bd2a67eda0d1.png)

(better to recognize negative sentiment)

Now let's check on the same data without neutral label in order to compare with Flair, results:  
![image](https://user-images.githubusercontent.com/53173112/214011636-120c2004-55f3-4767-b45c-0d66612d9d0b.png)

pos_label="negative":
Accuracy: 0.9188111948704618

Precision: 0.9840244332197815

Recall: 0.9127260841141861

F1 score: 0.9470352156463738

pos_label="positive":

Accuracy: 0.9188111948704618

Precision: 0.7354689564068693

Recall: 0.9424460431654677

F1 score: 0.826191801150065

![image](https://user-images.githubusercontent.com/53173112/214012023-cc7f5ab5-8ab5-4dbc-9d74-597cf9e47408.png)

![image](https://user-images.githubusercontent.com/53173112/214012081-e948e4f4-d3ff-42b5-9faa-43e68a97e7b9.png)

Summary: Hugging Face Zero-shot - is doing much better than flair (on both data sets), in recognizing Negative and Positive, with 0.93 AUC vs 0.86. About the model it's self, it's better on negative sentiment. 

--------------------------------------------------------------Textblob Part - 8 -----------------------------------------

Bad performance on my specific data, speccialy on tweets. In my opnion, it's the threeshold reason: 
data_3['scores_Textblob'] = data_3['scores_Textblob'].progress_apply(lambda x: "negative" if x < 0 else ("positive" if x >0.1 else "neutral")) no simple solution, I tried to play with different threesholds but it din't improve the results ( take a look to the code )

