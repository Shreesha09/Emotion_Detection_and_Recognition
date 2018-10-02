'''
#comparing multiple models
models = [RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0), LinearSVC(), MultinomialNB(), LogisticRegression(random_state=0)]

CV = 5
cv_df = pd.DataFrame(index=range(CV*len(models)))
entries = []
for model in models:
	model_name = model.__class__.__name__
	accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
	for fold_idx, accuracy in enumerate(accuracies):
		entries.append((model_name, fold_idx, accuracy))
	cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

print(cv_df.groupby('model_name').accuracy.mean())

'''
N = 2

	for emotion in sorted(category_id_df.values):
		#print(emotion)
		features_chi2 = chi2(features, labels == emotion)
		indices = np.argsort(features_chi2[0])
		features_names = np.array(tfifd.get_feature_names())[indices]
		bigrams = [v for v in features_names if len(v.split(' ')) == 2]
		#print("Most correlated bigrams: {}".format('\n.'.join(bigrams[-N:])))