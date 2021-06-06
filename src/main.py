# Conrad Ibanez
'''
Code for ths project has been taken from the below references and modified.
References:
https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk
https://towardsdatascience.com/a-beginners-guide-to-sentiment-analysis-in-python-95e354ea84f6
https://www.datacamp.com/community/tutorials/simplifying-sentiment-analysis-python
https://www.nltk.org/book/ch06.html
'''

import matplotlib.pyplot as plt
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import re, string, random
import pandas as pd

import plotly.express as px

def read_test_data():
    # read the file
    data_df = pd.read_csv('../data/tweet-sentiment-extraction/test.csv')
    #print(data_df.size)
    return data_df

def read_train_data():
    # read the file
    data_df = pd.read_csv('../data/tweet-sentiment-extraction/train.csv')
    #print(data_df.size)
    return data_df

def graph_data(df, field, title):
    
    print('In graph_data')
    
    file_location = 'images/' + title + '.jpeg'

    # Product Scores
    fig = px.histogram(df, x=field)
    fig.update_traces(marker_color="blue")
    fig.update_layout(title_text=title)
    fig.write_image(file_location)
    
def create_word_cloud(data, stopwords, filename):
    # Create stopword list:
    #stopwords = set(STOPWORDS)
    #stopwords.update(["br", "href"])
    #stopwords=[]
    file_location = 'images/' + filename + '.png'
    full_text = " ".join(str(text) for text in data)
    wordcloud = WordCloud(stopwords=stopwords).generate(full_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(file_location)
    plt.show()


def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []
    #stop_words =['...','..']

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

def main():
    
    random.seed(10)
    stop_word =[]
    
    train_data = read_train_data()
    graph_data(train_data, 'sentiment', 'Training Data File Sentiment Count')
    create_word_cloud(train_data['selected_text'].fillna('').apply(str), stop_word, "TrainDataWordCloud")
    train_data['selected_text'] = train_data['selected_text'].fillna('').apply(str)
    #print('train_data', train_data.head())
    positive_data = train_data[(train_data['sentiment'] == 'positive')]
    #print('\n\npositive_data', positive_data.head())  
    positive_tweets = positive_data['selected_text'].values
    
    positive_tweet_tokens = []
    negative_tweet_tokens = []
    neutral_tweet_tokens = []
    
    for tweet in positive_tweets:
        positive_tweet_tokens.append(word_tokenize(tweet))
    #custom_tokens = word_tokenize(custom_tweet)
    #custom_tokens = remove_noise(word_tokenize(custom_tweet))

    negative_data = train_data[(train_data['sentiment'] == 'negative')] 
    #print('\n\nnegative_data', negative_data.head())
    negative_tweets = negative_data['selected_text'].values
    
    for tweet in negative_tweets:
        negative_tweet_tokens.append(word_tokenize(tweet))

    neutral_data = train_data[(train_data['sentiment'] == 'neutral')] 
    #print('\n\nneutral_data', neutral_data.head())  
    neutral_tweets = neutral_data['selected_text'].values  
    for tweet in neutral_tweets:
        neutral_tweet_tokens.append(word_tokenize(tweet))

    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = [] 
    neutral_cleaned_tokens_list = []
    
    stop_words = stopwords.words('english')
    
    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
    for tokens in neutral_tweet_tokens:
        neutral_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
        
    all_pos_words = get_all_words(positive_cleaned_tokens_list)
    freq_dist_pos = FreqDist(all_pos_words)
    print('\n\nPositive')
    print(freq_dist_pos.most_common(10))
    
    
    all_neg_words = get_all_words(negative_cleaned_tokens_list)
    freq_dist_neg = FreqDist(all_neg_words)
    print('\n\nNegative')
    print(freq_dist_neg.most_common(10))
    
    all_neut_words = get_all_words(neutral_cleaned_tokens_list)
    freq_dist_neut = FreqDist(all_neut_words)
    print('\n\nNeutral')
    print(freq_dist_neut.most_common(10))
    
    create_word_cloud(positive_cleaned_tokens_list, stop_word, "TrainPositiveWordCloud")
    create_word_cloud(negative_cleaned_tokens_list, stop_word, "TrainNegativeWordCloud")
    create_word_cloud(neutral_cleaned_tokens_list, stop_word, "TrainNeutralWordCloud")
    
    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)
    neutral_tokens_for_model = get_tweets_for_model(neutral_cleaned_tokens_list)

    positive_dataset = [(tweet_dict, "positive")
                         for tweet_dict in positive_tokens_for_model]

    negative_dataset = [(tweet_dict, "negative")
                         for tweet_dict in negative_tokens_for_model]
    
    neutral_dataset = [(tweet_dict, "neutral")
                         for tweet_dict in neutral_tokens_for_model]

    dataset = positive_dataset + negative_dataset + neutral_dataset
    
    train_size = int(len(dataset) * .20)
    print("\n\nTrain size:", train_size)
    random.shuffle(dataset)

    split_training_data = dataset[:train_size]
    split_testing_data = dataset[train_size:]
    
    #print("\n\nTraining data:", split_training_data[0])
    #print("\n\nTraining data:", split_training_data[1])
    
    #print("Test data:", split_testing_data[0])
    print("Test data:", split_testing_data[1])
#
    classifier = NaiveBayesClassifier.train(split_training_data)

    print("Accuracy is:", classify.accuracy(classifier, split_testing_data))

    print(classifier.show_most_informative_features(20))
    

    #classifier = NaiveBayesClassifier.train(data_set)
    #print("\n\nTrain size:", len(data_set))
    
    
    print("\n\nEvaluating Actual Test Data File")
    actual_data = read_test_data()
    graph_data(actual_data, 'sentiment', 'Test Data File Sentiment Count')
    actual_data['text'] = actual_data['text'].fillna('').apply(str)
    
    actual_positive_tweet_tokens = []
    actual_negative_tweet_tokens = []
    actual_neutral_tweet_tokens = []
    
    actual_positive_data = actual_data[(actual_data['sentiment'] == 'positive')] 
    actual_positive_tweets = actual_positive_data['text'].values
    for tweet in actual_positive_tweets:
        actual_positive_tweet_tokens.append(word_tokenize(tweet))
    
    actual_negative_data = actual_data[(actual_data['sentiment'] == 'negative')] 
    actual_negative_tweets = actual_negative_data['text'].values    
    for tweet in actual_negative_tweets:
        actual_negative_tweet_tokens.append(word_tokenize(tweet))
    
    
    actual_neutral_data = actual_data[(actual_data['sentiment'] == 'neutral')] 
    actual_neutral_tweets = actual_neutral_data['text'].values  
    for tweet in actual_neutral_tweets:
        actual_neutral_tweet_tokens.append(word_tokenize(tweet))
        
    actual_positive_cleaned_tokens_list = []
    actual_negative_cleaned_tokens_list = [] 
    actual_neutral_cleaned_tokens_list = []
    
    for tokens in actual_positive_tweet_tokens:
        actual_positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
    for tokens in actual_negative_tweet_tokens:
        actual_negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
    for tokens in actual_neutral_tweet_tokens:
        actual_neutral_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
        
        
    actual_positive_tokens_for_model = get_tweets_for_model(actual_positive_cleaned_tokens_list)
    actual_negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)
    actual_neutral_tokens_for_model = get_tweets_for_model(neutral_cleaned_tokens_list)
    
    actual_positive_dataset = [(tweet_dict, "positive")
                         for tweet_dict in actual_positive_tokens_for_model]

    actual_negative_dataset = [(tweet_dict, "negative")
                         for tweet_dict in actual_negative_tokens_for_model]
    
    actual_neutral_dataset = [(tweet_dict, "neutral")
                         for tweet_dict in actual_neutral_tokens_for_model]

    actual_dataset = actual_positive_dataset + actual_negative_dataset + actual_neutral_dataset
    
    # Use all data from training file
    full_classifier = NaiveBayesClassifier.train(dataset)

    print("Accuracy is:", classify.accuracy(full_classifier, actual_dataset))

    print(full_classifier.show_most_informative_features(20))
     
    actual_data['predicted'] = 'none'
    error_dataframe = pd.DataFrame()
    for index, row in actual_data.iterrows():
      custom_tweet = row['text']
      custom_tokens = word_tokenize(custom_tweet)
      custom_tokens = remove_noise(word_tokenize(custom_tweet), stop_words)
      prediction = full_classifier.classify(dict([token, True] for token in custom_tokens))
      actual_data.at[index,'predicted'] = prediction
      if row['sentiment'] != prediction:
         print('Error: Actual=', row['sentiment'], " Predicted=", prediction, ' tweet=', custom_tweet, "\n")
         error_dataframe = error_dataframe.append(row)
      
    
    error_dataframe.to_csv('error_list.csv')
    actual_data.to_csv('test_Predicted.csv')

    

    create_word_cloud(actual_positive_cleaned_tokens_list, stop_word, "TestPositiveWordCloud")
    create_word_cloud(actual_negative_cleaned_tokens_list, stop_word, "TestNegativeWordCloud")
    create_word_cloud(actual_neutral_cleaned_tokens_list, stop_word, "TestNeutralWordCloud")
    
    print('\n\nDisplaying Classification Report')
    print(classification_report(actual_data['sentiment'], actual_data['predicted']))
    
    print('\n\nConfusion matrix:', confusion_matrix(actual_data['sentiment'], actual_data['predicted']))

    print('Accuracy:', accuracy_score(actual_data['sentiment'], actual_data['predicted']))


if __name__ == '__main__':
    main()
    
