from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import  classify, NaiveBayesClassifier
import pickle
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

if __name__ == "__main__":


	stop_words = stopwords.words('english')

	# Loading the Data
	positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
	negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

	# Cleaning the Data
	positive_cleaned_tokens_list = [
	    remove_noise(tokens, stop_words) for tokens in positive_tweet_tokens
	]
	negative_cleaned_tokens_list = [
	    remove_noise(tokens, stop_words) for tokens in negative_tweet_tokens
	]

	# Prepare the data
	positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
	negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

	positive_dataset = [(tweet_dict, "Positive")
	                     for tweet_dict in positive_tokens_for_model]

	negative_dataset = [(tweet_dict, "Negative")
	                     for tweet_dict in negative_tokens_for_model]

	dataset = positive_dataset + negative_dataset

	# Shuffle
	random.shuffle(dataset)

	train_data = dataset[:9000]
	test_data = dataset[9000:]

	# Training
	print('*'*20)
	print("Training")
	classifier = NaiveBayesClassifier.train(train_data)
	print('*'*20)

	# Accuracy
	print("Accuracy is:", classify.accuracy(classifier, test_data))

	print(classifier.show_most_informative_features(10))

	custom_tweet = "I ordered just once from TerribleCo, they screwed up, never used the app again."

	custom_tokens = remove_noise(word_tokenize(custom_tweet))

	print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))
	print('*'*20)
	print("Saving the Model")

	with open('saved_model/nltk_sentiment_classifier.pickle', 'wb') as f:
		pickle.dump(classifier, f)
	
	print("Saved model at saved_model/nltk_sentiment_classifier.pickle")