import pickle
from nltk.tokenize import word_tokenize
from sentiment_analysis_main import remove_noise
import streamlit as st
import webbrowser

readme = 'https://github.com/tikendraw/sentiment-analysis-using-nltk/blob/main/README.md'
notebook_url = 'https://github.com/tikendraw/sentiment-analysis-using-nltk/blob/main/notebooks/sentiment_analysis_notebook.ipynb'
github_repo  = 'https://github.com/tikendraw/sentiment-analysis-using-nltk'



def header():
	col1, col2, col3 = st.columns([1,1,1])

	with col1:
		if st.button('Github'):
			webbrowser.open_new_tab(github_repo)
	with col2:
		if st.button('Notebook'):
			webbrowser.open_new_tab(notebook_url)
	with col3:
		if st.button('Readme'):
			webbrowser.open_new_tab(readme)



def predict(tweet:str, classifier):

	 
	tweet_tokens = remove_noise(word_tokenize(tweet))

	return classifier.classify(dict([token, True] for token in tweet_tokens))






def main():
	st.title("Sentiment Analysis using Natural Language Processing")
	
	header()

	st.subheader("Home")
	with st.form(key='nlpForm'):
		raw_text = st.text_area("Enter the text you want to analyze here: ")
		submit_button = st.form_submit_button(label='Analyze')

		#Loading the Model(here because it is going to take some time for input meanwhile we will load the model)
		with open('saved_model/nltk_sentiment_classifier.pickle', 'rb') as f:
			classifier = pickle.load(f)


	# layout
	st.info("Text Sentiment")

	if submit_button:
		token_sentiments = predict(raw_text, classifier)
		st.write(raw_text)

		if token_sentiments.lower().strip() == "positive":
			st.success(token_sentiments)  
		else: 
			st.warning(token_sentiments)




if __name__ == '__main__':
    main()












# def main():
# 	...

# if __name__=="__main__":
	
# 	main()
