import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB


def review_to_words(raw_review):
    letters_only = re.sub("[^a-zA-Z]", " ", raw_review)
    lower_Case = letters_only.lower()
    words = lower_Case.split(" ")
    words = [w for w in words if not w in stopwords.words("english")]
    return " ".join(words)

train = pd.read_csv("dataset/TrainingSet.tsv", header=0, delimiter="\t")
num_reviews = train["query"].size
print num_reviews
clean_train_reviews = []

for i in xrange(0,num_reviews):
    if((i+1)%500 == 0):
        print "query %d of %d\n" % ( i+1, num_reviews )
    clean_train_reviews.append(review_to_words(train["query"][i]))

print "Creating the bag of words...\n"
vectorizer = CountVectorizer(analyzer="word",ngram_range=(1,4), tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()
print vectorizer.get_feature_names()

mnb = MultinomialNB()
mnb.fit(train_data_features,train["class"])



test = pd.read_csv("dataset/TestSet.tsv", header=0, delimiter="\t",quoting=3 )

# Verify that there are 25,000 rows and 2 columns
print test.shape

# Create an empty list and append the clean reviews one by one
num_reviews = len(test["query"])
clean_test_reviews = []


print "Cleaning and parsing the test set queries...\n"
for i in xrange(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print "Queries %d of %d\n" % (i+1, num_reviews)
    clean_review = review_to_words( test["query"][i] )
    clean_test_reviews.append( clean_review )

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()


# Use the random forest to make sentiment label predictions
result = mnb.predict(test_data_features)
# Copy the results to a pandas dataframe with an "id" column and
output = pd.DataFrame(data={"id":test["query"], "class":result} )

scores = cross_val_score(mnb,test_data_features, test["class"], cv=5)
print scores
# Use pandas to write the comma-separated output file

output.to_csv( "NB_EventsClassification_Bag_of_Words_model.csv", index=False, quoting=3 )

while(True):
    query = raw_input(">>")
    queryArray = []
    queryArray.append(review_to_words(query))
    test_data_features = vectorizer.transform(queryArray)
    test_data_features = test_data_features.toarray()
    print(mnb.predict(test_data_features))
