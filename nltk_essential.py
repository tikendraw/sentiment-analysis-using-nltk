import nltk

def setup():

	print('Doing setup...')

	print('downloading stopwords')
	nltk.download('stopwords')
	
	print('downloading twitter_samples')
	nltk.download('twitter_samples')
	
	print('downloading punkt')
	nltk.download('punkt') # helps you tokenize words and sentences 
	
	print('downloading wordnet')
	nltk.download('wordnet') 
	
	print('downloading averaged_perceptron_tagger')
	nltk.download('averaged_perceptron_tagger')

	print('Done setup')

if __name__ == '__main__':
	setup()