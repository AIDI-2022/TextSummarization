import nltk
import pickle
import re
import numpy as np
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en import English
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def preprocessor(text):

  replace_by_space_re= re.compile('[/(){}\[\]\|@,;]')

  bad_symbols_re= re.compile('[^0-9a-z #+_]')

  stopwords_set= set(stopwords.words('english'))

  text= text.lower()

  text= replace_by_space_re.sub(' ', text)

  text= bad_symbols_re.sub('', text)

  text= ' '.join([lemmatizer.lemmatize(x) for x in text.split() if x and x not in stopwords_set])

  return text.strip()



def read_text_from_document(filepath):

    file = open(filepath,'r')
    text = file.read()
    file.close()
    datatxt = []
    for line in text.split('\n'):
        if line:
            line_dict = {}
            timestamp = re.findall(r'\d{2}:\d{2}:\d{2}',line)
            if timestamp:
                _timestamp = timestamp[0]
                line_dict['time_stamp'] = _timestamp
                line = line.strip(_timestamp)
            line_dict['sentence'] = line.strip().lstrip().rstrip()
            datatxt.append(line_dict)
    if datatxt:
        data = pd.DataFrame(datatxt)
        print(f'Input Unformatted Text:\n\n{data}\n\n')
        return ' '.join(data.sentence)

#clean_text = read_text_from_document(filepath)


tokenizer = English()
pipeline = tokenizer.create_pipe('sentencizer')
tokenizer.add_pipe(pipeline)
tokenized_sentences = tokenizer(clean_text.replace("\n",""))
sentences = [sent.string.strip() for sent in tokenized_sentences.sents]

print("Sentences are: \n",sentences)