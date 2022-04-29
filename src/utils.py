# import packages
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from models.tfidf_summarization import tfidf_summarizer

def extract_content(filepath):

    # read the text from the file
    file = open(filepath,'r')
    text = file.read()
    file.close()
    
    data = []
    for line in text.split('\n'):
        if line:
            line_dict = {}
            # regular expression to find the timestamp in each line
            timestamp = re.findall(r'\d{2}:\d{2}:\d{2}',line)
            
            if timestamp:
                _timestamp = timestamp[0]
                line_dict['time_stamp'] = _timestamp
                line = line.strip(_timestamp)
            
            line_dict['sentence'] = line.strip().lstrip().rstrip()
            data.append(line_dict)
    
    if data:
        df = pd.DataFrame(data)
        return df

def generate_summaries_on_xsum(model, xsum_test, n=10):
    
    data = xsum_test.head(n)
    documents = data['document']
    references = list(data['summary'])
    generated_summaries = []
    
    for doc in documents:
        summary = model(doc)
        generated_summaries.append(summary)
    
    return generated_summaries, references

def generate_summaries_tfidf(xsum_test, n=10):
    
    data = xsum_test.head(n)
    documents = data['document']
    references = list(data['summary'])
    generated_summaries = []
    
    for doc in documents:
        summary = tfidf_summarizer(doc)
        
        generated_summaries.append(summary)
    
    return generated_summaries, references