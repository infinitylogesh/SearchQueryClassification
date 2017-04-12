import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import plotOps
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB


def review_to_words(raw_review):
    letters_only = re.sub("[^a-zA-Z]", " ", raw_review)
    lower_Case = letters_only.lower()
    words = lower_Case.split(" ")
    words = [w for w in words if not w in stopwords.words("english")]
    return " ".join(words)


def all_words_to_Array(queries):
    temp = []
    for i in xrange(0,queries.size):
        if((i+1)%500 == 0):
            print "query %d of %d\n" % ( i+1, queries.size)
        temp.append(review_to_words(queries[i]))
    return temp

train = pd.read_csv("dataset/multiTrainingSet.tsv", header=0, delimiter="\t")
num_reviews = train["query"].size
print num_reviews
clean_train_reviews = all_words_to_Array(train["query"])

print "Creating the bag of words...\n"
vectorizer = CountVectorizer(analyzer="word",ngram_range=(1,4), tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()
print vectorizer.get_feature_names()

test = pd.read_csv("dataset/multiTestSet.tsv", header=0, delimiter="\t",quoting=3 )

# Verify that there are 25,000 rows and 2 columns
print test.shape

# Create an empty list and append the clean reviews one by one

clean_test_reviews = all_words_to_Array(test["query"])
print "Cleaning and parsing the test set queries...\n"


# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
lr = LogisticRegression(multi_class='multinomial',solver='sag')
lr.fit(train_data_features,train["class"])
predicted = lr.predict(test_data_features)

eventsFeatures =  zip(lr.coef_[0],vectorizer.get_feature_names())
workitemsFeatures =  zip(lr.coef_[1],vectorizer.get_feature_names())
knowledgeFeatures =  zip(lr.coef_[2],vectorizer.get_feature_names())
expertsFeatures =  zip(lr.coef_[3],vectorizer.get_feature_names())

plotOps.word_score_plot(sorted(eventsFeatures,key=lambda x:x[0],reverse=True),10)
plotOps.word_score_plot(sorted(workitemsFeatures,key=lambda x:x[0],reverse=True),10)


# plotOps.plot3D(test["query"],test_data_features,test["class"])
# plot3D(test["query"],test_data_features,predicted)

# print lr.score(test_data_features,test["class"])
# plotOps.plot_learning_curve(lr,"First Plot",test_data_features, test["class"], cv=5)


# while(True):
#     query = raw_input(">>")
#     queryArray = []
#     queryArray.append(review_to_words(query))
#     test_data_features = vectorizer.transform(queryArray)
#     test_data_features = test_data_features.toarray()
#     print(lr.predict(test_data_features))


# Use the random forest to make sentiment label predictions
#result = mnb.predict(test_data_features)
# Copy the results to a pandas dataframe with an "id" column and
# output = pd.DataFrame(data={"id":test["query"], "class":result} )
#
# scores = cross_val_score(mnb,test_data_features, test["class"], cv=5)
# print scores
# # Use pandas to write the comma-separated output file
#
# output.to_csv( "NB_EventsClassification_Bag_of_Words_model.csv", index=False, quoting=3 )
#

