# import packages
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocessor(text):

    lemmatizer = WordNetLemmatizer()
    
    re_special_characters = re.compile('[/(){}\[\]\|@,;]')

    re_bad_characters = re.compile('[^0-9a-z #+_]')

    stopwords_set= set(stopwords.words('english'))

    text= text.lower()

    text= re_special_characters.sub(' ', text)

    text= re_bad_characters.sub('', text)

    text= ' '.join([lemmatizer.lemmatize(x) for x in text.split() if x and x not in stopwords_set])

    return text.strip()