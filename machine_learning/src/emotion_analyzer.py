import pandas as pd
import numpy as np
import string
import pickle
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, classification_report


#paths and input initialization
input_path = 'input2.csv'
inputs = pd.read_csv(input_path, usecols=[0,1], names=['emotion','text'])
model_file = 'finalized_model.sav'

#variables initialization
stop_words = stopwords.words('english')


#lemmatizer
lemmatizer = WordNetLemmatizer()

if os.path.isfile(model_file):
	#load model
	model = pickle.load(open(model_file, 'rb'))

else:
#inputs preprocessing
	category_id_df = inputs['emotion'].drop_duplicates()

	inputs['text'] = inputs['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))

	inputs['text'] = inputs['text'].str.replace('[^\w\s]','')

	inputs['text'] = inputs['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))

	#rare words removal, top 10 least frequently used words in the corpus 
	freq = pd.Series(' '.join(inputs['text']).split()).value_counts()[-10:]
	#print(freq)
	freq = list(freq.index)

	inputs['text'] = inputs['text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

	inputs['text'] = inputs['text'].apply(lambda x: " ".join(lemmatizer.lemmatize(x) for x in x.split()))

	#TFIDF vectorizer
	tfifd = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), decode_error="replace")
	features = tfifd.fit_transform(inputs.text).toarray() 
	pickle.dump(tfifd,open("feature.pkl","wb"))
	labels = inputs['emotion']	
	
	N = 2

	for emotion in sorted(category_id_df.values):
		#print(emotion)
		features_chi2 = chi2(features, labels == emotion)
		indices = np.argsort(features_chi2[0])
		features_names = np.array(tfifd.get_feature_names())[indices]
		bigrams = [v for v in features_names if len(v.split(' ')) == 2]
		#print("Most correlated bigrams: {}".format('\n.'.join(bigrams[-N:])))

	X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, inputs.index, test_size=0.33, random_state=0)

	model = MultinomialNB()
	model.fit(X_train, y_train)
	pickle.dump(model, open(model_file, 'wb'))		
	y_pred = model.predict(X_test)
	print(classification_report(y_test, y_pred, target_names=inputs['emotion'].unique()))
	probabilities = pd.DataFrame(model.predict_proba(X_test), columns=model.classes_)
	#print(probabilities)


#preprocessing functions for new inputs
#converting all texts into lower case
def to_lower(text):
	n_text = text.lower()
	return n_text


#remove punctuation
def remove_punctuation(text):
	table = str.maketrans({key: None for key in string.punctuation})
	new_text = text.translate(table)
	return new_text 
	
#removal of stopwords
def remove_stop_words(text):
	new_text = " ".join(x for x in text.split() if x not in stop_words)
	return new_text


#Lemmatization - convert every word to its base form (running->run)
def lemmatize_(text):
	n_text = lemmatizer.lemmatize(text)
	return n_text


def vectorize(text):
	loaded_vec = pickle.load(open("feature.pkl", "rb"))
	n_features = []
	n_features = loaded_vec.transform([text])
	return n_features


def main():
	#checking for new sentence
	new_input = input("Enter a sentence: \n")
	n1 = to_lower(new_input)
	n2 = remove_punctuation(n1)
	n3 = remove_stop_words(n2)
	n4 = lemmatize_(n3)
	new_input_features = vectorize(n3)
	new_input_prob = pd.DataFrame(model.predict_proba(new_input_features), columns=model.classes_)
	print(new_input_prob)

main()
