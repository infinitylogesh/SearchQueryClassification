import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
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

def plot3D(texts,queries_matrix,Y):
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    X_reduced = PCA(n_components=3).fit_transform(queries_matrix)
    for i in xrange(0,Y.size):
        ax.scatter(X_reduced[:, 0][i], X_reduced[:, 1][i], X_reduced[:, 2][i],
           cmap=plt.cm.Paired)
        ax.text(X_reduced[:, 0][i], X_reduced[:, 1][i], X_reduced[:, 2][i] , '%s' % (texts[i]), size=5, zorder=1)
    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels(())
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels(())
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels(())
    plt.show()

train = pd.read_csv("dataset/TrainingSet.tsv", header=0, delimiter="\t")
num_reviews = train["query"].size
print num_reviews
clean_train_reviews = all_words_to_Array(train["query"])

print "Creating the bag of words...\n"
vectorizer = CountVectorizer(analyzer="word",ngram_range=(1,4), tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()
print vectorizer.get_feature_names()

test = pd.read_csv("dataset/TestSet.tsv", header=0, delimiter="\t",quoting=3 )

# Verify that there are 25,000 rows and 2 columns
print test.shape

# Create an empty list and append the clean reviews one by one

clean_test_reviews = all_words_to_Array(test["query"])
print "Cleaning and parsing the test set queries...\n"


# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english')
tfidfMatrix = tf.fit_transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
lr = LogisticRegression()
lr.fit(train_data_features,train["class"])
predicted = lr.predict(test_data_features)
plot3D(test["query"],tfidfMatrix.todense(),test["class"])
plot3D(test["query"],tfidfMatrix.todense(),predicted)




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
# while(True):
#     query = raw_input(">>")
#     queryArray = []
#     queryArray.append(review_to_words(query))
#     test_data_features = vectorizer.transform(queryArray)
#     test_data_features = test_data_features.toarray()
#     print(mnb.predict(test_data_features))
