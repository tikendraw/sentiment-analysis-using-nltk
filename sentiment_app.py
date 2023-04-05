import pickle
from nltk.tokenize import word_tokenize
from sentiment_analysis_main import remove_noise
import streamlit as st


def predict(tweet:str, classifier):

	 
	tweet_tokens = remove_noise(word_tokenize(tweet))

	return classifier.classify(dict([token, True] for token in tweet_tokens))






def main():
	st.title("Sentiment Analysis using Natural Language Processing")
	st.subheader("ISTE-782 Visual Analytics")

	menu = ["Home", "About"]
	choice = st.sidebar.selectbox("Menu", menu)

	if choice == "Home":
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

			if token_sentiments.lower().strip() == "positive":
				st.success(token_sentiments)  
			else: 
				st.warning(token_sentiments)

	else:
		st.subheader("About")
		st.write("Tikendraw")



if __name__ == '__main__':
    main()












# def main():
# 	...

# if __name__=="__main__":
	
# 	main()
