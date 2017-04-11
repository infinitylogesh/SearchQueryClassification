import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt


# Converts the queries to words - Removes stopwords and special characters.
def queries_to_words(raw_review):
    letters_only = re.sub("[^a-zA-Z]", " ", raw_review)
    lower_Case = letters_only.lower()
    words = lower_Case.split(" ")
    words = [w for w in words if not w in stopwords.words("english")]
    return " ".join(words)

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 1.5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Train queries are read from the file
train = pd.read_csv("dataset/TrainingSet.tsv", header=0, delimiter="\t")
num_queries = train["query"].size
print num_queries
clean_train_reviews = []

for i in xrange(0, num_queries):
    if((i+1)%500 == 0):
        print "query %d of %d\n" % (i + 1, num_queries)
    clean_train_reviews.append(queries_to_words(train["query"][i]))

print "Creating the bag of words...\n"
# Bag of words is created for each queries - from (1,3) - 1 to trigrams
vectorizer = CountVectorizer(analyzer="word",ngram_range=(1,3), tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()
print vectorizer.get_feature_names()

# Training the Randomforestclassifier
forest = RandomForestClassifier(n_estimators = 100).fit(train_data_features, train["class"] )

# Test queries are read from the file
test = pd.read_csv("dataset/TestSet.tsv", header=0, delimiter="\t",quoting=3 )

print test.shape

# Create an empty list and append the clean reviews one by one
num_queries = len(test["query"])
clean_test_queries = []


print "Cleaning and parsing the test set queries...\n"
for i in xrange(0, num_queries):
    if( (i+1) % 1000 == 0 ):
        print "Queries %d of %d\n" % (i + 1, num_queries)
    clean_review = queries_to_words(test["query"][i])
    clean_test_queries.append(clean_review)

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_queries)
test_data_features = test_data_features.toarray()


# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)
print zip(vectorizer.get_feature_names(),forest.feature_importances_)
# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame(data={"id":test["query"], "class":result} )

scores = cross_val_score(forest,test_data_features, test["class"], cv=5)
#train_sizes, train_scores, valid_scores =  learning_curve(
#    RandomForestClassifier(n_estimators = 100), test_data_features, test["class"], cv=5, train_sizes=[50, 80, 110])
plot_learning_curve(forest,"First Plot",test_data_features, test["class"], ylim=(0.7, 1.01), cv=5)
#print train_sizes, train_scores, valid_scores
# Use pandas to write the comma-separated output file
plt.show()
output.to_csv( "Forest_EventsClassification_Bag_of_Words_model.csv", index=False, quoting=3 )

while(True):
    query = raw_input(">>")
    queryArray = []
    queryArray.append(queries_to_words(query))
    test_data_features = vectorizer.transform(queryArray)
    test_data_features = test_data_features.toarray()
    print(forest.predict(test_data_features))
