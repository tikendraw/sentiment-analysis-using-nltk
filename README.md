# Sentiment Analysis Streamlit App

This is a sentiment analysis app that uses natural language processing to classify the sentiment of text input as either positive or negative.

## Dependencies
* pickle 
* nltk
* streamlit

## How to run the app

1. Clone the repository to your local machine.
2. Install the dependencies by running the command 
```
pip install -r   requirements.txt
```
run this if you don't have nltk already install

```
pip install nltk
nltk.download('stopwords')
nltk.download('twitter_samples')
nltk.download('punkt') # helps you tokenize words and sentences
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

3. Run the command 
```
streamlit run sentiment_app.py
```
The app will open in your default web browser.

## How to use the app
1. Input the text you want to analyze in the provided text area.
2. Click the "Analyze" button.
3. The app will classify the sentiment of the input text as either "positive" or "negative".
If the sentiment is positive, the app will display the output in green. If the sentiment is negative, the app will display the output in yellow.

## Files
* `sentiment_analysis_main.py`: This file contains the functions to clean and prepare the data for sentiment analysis.
* `saved_model/nltk_sentiment_classifier.pickle`: This file contains the trained classifier for sentiment analysis.
* `sentiment_app.py`: This file contains the code for the streamlit app.

## Authors
Tikendra Kumar Sahu (tikendraksahu1029@gmail.com)