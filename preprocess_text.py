import json
import nltk
import re
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer #is based on The Porter Stemming Algorithm
from contractions import contractions_dict
from autocorrect import Speller
# import autocorrect
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


def to_lower(text):
    """
    Converting text to lower case as in, converting "Hello" to  "hello" or "HELLO" to "hello".
    """
    return ' '.join([w.lower() for w in nltk.word_tokenize(text)])

def strip_punctuation(text):
    """
    Removinmg puctuation
    """
    return ''.join(c for c in text if c not in punctuation)

def deEmojify(inputString):
    '''
    Removing emojis from text
    '''
    return inputString.encode('ascii', 'ignore').decode('ascii')
def prep_text():

    #reading the json comment file
    with open('memory_loss_comments.json') as json_file:
        lower_text = []
        data = json.load(json_file)
        for i in data["memory_loss"]:
            lower_text.append(to_lower(i["comments"])) #converting the comments in memory_loss dictionary to lower case



    #removing emojis, report, reply and number of likes
    cleaned_text = re.sub(r"reply", " " , str(lower_text))
    cleaned_text = re.sub(r"\n", " ", str(cleaned_text))
    cleaned_text = re.sub(r"report", " ", str(cleaned_text))
    cleaned_text = re.sub("[0-9]+\slikes", " ", str(cleaned_text))
    cleaned_text = re.sub("…", " ", str(cleaned_text))
    cleaned_text = deEmojify(cleaned_text)



    #removing digits, since they're not important
    cleaned_text = ''.join(c for c in cleaned_text if not c.isdigit())

    #removing punctuations
    cleaned_text = strip_punctuation(cleaned_text)


    #stopwords and tokenizing them
    stopword = nltk.corpus.stopwords.words("english")
    removing_stopwords = [word for word in cleaned_text.split() if word not in stopword]

    # removing_stopwords = removing_stopwords[:100]

    # for w in removing_stopwords:
    #     try:
    #         Speller(w)
    #     except:
    #         import ipdb; ipdb.set_trace()

    speller = Speller()
    correct_spelling = [speller(w) for w in removing_stopwords]
    # print (correct_spelling)

    #lemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in correct_spelling]
    # print(lemmatized_word)

    #stemming
    snowball_stemmer = SnowballStemmer("english")
    stemmed_word = [snowball_stemmer.stem(word) for word in lemmatized_word]
    return stemmed_word

    # x = re.findall("[^a-zA-Z]very$", ' '.join(c for c in removing_stopwords))
    # print(x)

