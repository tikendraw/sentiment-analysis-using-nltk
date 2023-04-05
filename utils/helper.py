from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier

import re, string, random


def remove_noise(tweet_tokens, stop_words = ()):

	"""
		The remove_noise function takes a list of tweet tokens and removes noise from them. 
		Noise in this case refers to URLs, user mentions, stop words, and punctuation marks. 
		The function returns a list of cleaned tokens.
	"""
	
	cleaned_tokens = []

	for token, tag in pos_tag(tweet_tokens):
		token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
						'(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
		token = re.sub("(@[A-Za-z0-9_]+)","", token)

		if tag.startswith("NN"):
			pos = 'n'
		elif tag.startswith('VB'):
			pos = 'v'
		else:
			pos = 'a'

		lemmatizer = WordNetLemmatizer()
		token = lemmatizer.lemmatize(token, pos)

		if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
			cleaned_tokens.append(token.lower())
	return cleaned_tokens

def get_all_words(cleaned_tokens_list):

    """
    The get_all_words function takes a list of cleaned token lists and generates 
	a flattened list of all words.
    """

    for tokens in cleaned_tokens_list:
        yield from tokens

def get_tweets_for_model(cleaned_tokens_list):
    
	"""
	The get_tweets_for_model function takes a list of cleaned token lists 
	and generates a dictionary of word features for each tweet.
	"""

	for tweet_tokens in cleaned_tokens_list:
		yield dict([token, True] for token in tweet_tokens)





























def main():
	...

if __name__=="__main__":
	
	main()
