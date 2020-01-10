import pandas as pd
import numpy as np
import nltk
import sklearn
import operator
import random
from nltk.util import ngrams
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


dataset_file_neg = pd.read_csv('./IMDb/train/imdb_train_neg.txt', sep='\n', header=None)
dataset_file_pos = pd.read_csv('./IMDb/train/imdb_train_pos.txt', sep='\n', header=None)
dataset_file_neg = np.ravel(dataset_file_neg)
dataset_file_pos = np.ravel(dataset_file_pos)

train_dataset = []
for pos_review in dataset_file_pos:
    train_dataset.append((pos_review, 1))
for neg_review in dataset_file_neg:
    train_dataset.append((neg_review, 0))

dataset_file_neg = pd.read_csv('./IMDb/dev/imdb_dev_neg.txt', sep='\n', header=None)
dataset_file_pos = pd.read_csv('./IMDb/dev/imdb_dev_pos.txt', sep='\n', header=None)
dataset_file_neg = np.ravel(dataset_file_neg)
dataset_file_pos = np.ravel(dataset_file_pos)

dev_dataset = []
for pos_review in dataset_file_pos:
    dev_dataset.append((pos_review, 1))
for neg_review in dataset_file_neg:
    dev_dataset.append((neg_review, 0))

dataset_file_neg = pd.read_csv('./IMDb/test/imdb_test_neg.txt', sep='\n', header=None)
dataset_file_pos = pd.read_csv('./IMDb/test/imdb_test_pos.txt', sep='\n', header=None)
dataset_file_neg = np.ravel(dataset_file_neg)
dataset_file_pos = np.ravel(dataset_file_pos)

test_dataset = []
for pos_review in dataset_file_pos:
    test_dataset.append((pos_review, 1))
for neg_review in dataset_file_neg:
    test_dataset.append((neg_review, 0))

print("DONE: Importing the training, development and test dataset of IMDb reviews")

print("\nTraining set size: " + str(len(train_dataset)))
print("Development set size: " + str(len(dev_dataset)))
print("Test set size: " + str(len(test_dataset)))

# Shuffling the three dataset
random.shuffle(train_dataset)
random.shuffle(dev_dataset)
random.shuffle(test_dataset)

lemmatizer = nltk.stem.WordNetLemmatizer()
stopwords = set(nltk.corpus.stopwords.words('english'))
stopwords.add(".")
stopwords.add(",")
stopwords.add("--")
stopwords.add("``")
stopwords.add("#")
stopwords.add("@")
stopwords.add(":")
stopwords.add("http")
stopwords.add("/")
stopwords.add(">")
stopwords.add("<")
stopwords.add("br")
stopwords.add("(")
stopwords.add(")")
stopwords.add("''")
print("\nDONE: Setting a list of stopwords to prevent false recognition of words or patterns\n")


def get_list_tokens(string):
    sentence_split = nltk.tokenize.sent_tokenize(string)
    list_tokens = []
    for sentence in sentence_split:
        list_tokens_sentence = nltk.tokenize.word_tokenize(sentence)
        for token in list_tokens_sentence:
            list_tokens.append(lemmatizer.lemmatize(token).lower())
    return list_tokens


def get_vector_text(list_vocab, string):
    vector_text = np.zeros(len(list_vocab) + 1)
    list_tokens_string = get_list_tokens(string)
    for i, word in enumerate(list_vocab):
        if word in list_tokens_string:
            vector_text[i] = list_tokens_string.count(word)
    #   Putting as last feature the word count
    vector_text[len(list_vocab)] = len(string.split())
    return vector_text


def get_vocabulary(training_set, num_features):  # Function to retrieve vocabulary
    dict_word_frequency = {}
    for instance in training_set:
        sentence_tokens = get_list_tokens(instance[0])
        # Aiming to find combination of two words that express sentiment
        n_grams = ngrams(sentence_tokens, 2)
        n_gram_list = [' '.join(grams) for grams in n_grams]
        for word in n_gram_list:
            if word in stopwords: continue
            if any(stopword in stopwords for stopword in word.split()): continue
            if word not in dict_word_frequency:
                dict_word_frequency[word] = 1
            else:
                dict_word_frequency[word] += 1
        # Aiming to find single words that express sentiment
        for word in sentence_tokens:
            if word in stopwords: continue
            if word not in dict_word_frequency:
                dict_word_frequency[word] = 1
            else:
                dict_word_frequency[word] += 1
    sorted_list = sorted(dict_word_frequency.items(), key=operator.itemgetter(1), reverse=True)[:num_features]
    vocabulary = []
    for word, frequency in sorted_list:
        vocabulary.append(word)
    return vocabulary


def train_classifier(training_set, vocabulary):
    X_train = []
    Y_train = []
    for instance in training_set:
        vector_instance = get_vector_text(vocabulary, instance[0])
        X_train.append(vector_instance)
        Y_train.append(instance[1])
    # We train the Logistic regression classifier
    log_reg = sklearn.linear_model.LogisticRegression(solver='liblinear')
    log_reg.fit(np.asarray(X_train), np.asarray(Y_train))
    return log_reg


Y_dev = []
for instance in dev_dataset:
    Y_dev.append(instance[1])
Y_dev_gold = np.asarray(Y_dev)

print("STARTED: Feature Selection Process")
print("\nThe goal of this part of the algorithm is to select the best number\n"
      "of features for the machine learning model.\n"
      "Please allow some time for the process to finish...\n")

print("No. of features    Precision     Recall      F1-score      Accuracy")
list_num_features = [50, 100, 250, 500, 1000, 1250]
best_accuracy_dev = 0.0
for num_features in list_num_features:
    # First, we extract the vocabulary from the training set and train our logistic regression classifier
    vocabulary = get_vocabulary(train_dataset, num_features)
    log_reg = train_classifier(train_dataset, vocabulary)
    # Then, we transform our dev set into vectors and make the prediction on this set
    X_dev = []
    for instance in dev_dataset:
        vector_instance = get_vector_text(vocabulary, instance[0])
        X_dev.append(vector_instance)
    X_dev = np.asarray(X_dev)
    Y_dev_predictions = log_reg.predict(X_dev)
    # Getting and printing the results of the classifier
    accuracy_dev = accuracy_score(Y_dev_gold, Y_dev_predictions)
    precision = precision_score(Y_dev_gold, Y_dev_predictions, average='macro')
    recall = recall_score(Y_dev_gold, Y_dev_predictions, average='macro')
    f1 = f1_score(Y_dev_gold, Y_dev_predictions, average='macro')

    print("       " + str(num_features) + "           " + str(round(precision, 3)) +
          "         " + str(round(recall, 3)) + "         " + str(round(f1, 3)) + "         " +
          str(round(accuracy_dev, 3)))
    # Seeking for the best accuracy
    if accuracy_dev >= best_accuracy_dev:
        best_accuracy_dev = accuracy_dev
        best_num_features = num_features
        best_vocabulary = vocabulary
        best_log_reg = log_reg
print("\nCOMPLETED: Feature Selection Process")
print("\n Best accuracy overall in the dev set is " + str(round(best_accuracy_dev, 3)) + " with " +
      str(best_num_features) + " features.")

print("\nSTARTED: Predicting the test dataset")
# Creating a vocabulary for the best number of feature found above
vocabulary = get_vocabulary(train_dataset, 1000)
log_reg = train_classifier(train_dataset, vocabulary)

X_test = []
Y_test = []
for instance in test_dataset:
    vector_instance = get_vector_text(vocabulary, instance[0])
    X_test.append(vector_instance)
    Y_test.append(instance[1])
X_test = np.asarray(X_test)
Y_test_gold = np.asarray(Y_test)

Y_test_predictions = log_reg.predict(X_test)

print("\nResults of classification report\n")
print(classification_report(Y_test_gold, Y_test_predictions))
print("\nConfusion Matrix")
print(confusion_matrix(Y_test_gold, Y_test_predictions))

precision = precision_score(Y_test_gold, Y_test_predictions, average='macro')
recall = recall_score(Y_test_gold, Y_test_predictions, average='macro')
f1 = f1_score(Y_test_gold, Y_test_predictions, average='macro')
accuracy = accuracy_score(Y_test_gold, Y_test_predictions)

print("\nResults of the metrics from predicting the outcome of the test dataset")
print("Precision: " + str(round(precision, 3)))
print("Recall: " + str(round(recall, 3)))
print("F1-Score: " + str(round(f1, 3)))
print("Accuracy: " + str(round(accuracy, 3)))

print("COMPLETED: Predicting the test dataset")
print("\nEND OF ALGORITHM")
