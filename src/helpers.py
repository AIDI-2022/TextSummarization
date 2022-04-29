from preprocessing import preprocessor
import spacy
from collections import Counter
from wordcloud import WordCloud
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import requests

def get_token(filepath):
    
    file = open(filepath,'r')
    token = file.read()
    file.close()
    
    return token

def get_chat_id(filepath):
    
    file = open(filepath,'r')
    chatid = file.read()
    file.close()
    
    return chatid

def split(a, n):
    
    k, m = divmod(len(a), n)
    
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

    
def send_picture(image_path, chat_id):
    
    TOKEN = get_token('data/token.txt')
    
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    
    data = {'chat_id' : chat_id}
    
    files = {'photo': open(image_path, 'rb')}
    
    r= requests.post(url, files=files, data=data)


def save_wordcloud(answers):
    
    answers = preprocessor(answers)
    
    answers = [text.lower().strip().lstrip() for text in answers.split()]

    answer_tokens = []
    
    nlp = spacy.load("en_core_web_sm")
    
    for answer in answers:
        
        doc = nlp(answer)
        
        with doc.retokenize() as retokenizer:
            
            for nc in doc.noun_chunks:
                
                if all(token.is_stop != True and token.is_punct != True and '-PRON-' not in token.lemma_ for token in nc) == True:
                    retokenizer.merge(doc[nc.start:nc.end], attrs={"LEMMA": str(doc[nc.start:nc.end])})
            
        answer_tokens.extend([tok.lemma_ for tok in doc if (not tok.is_punct) and (not tok.is_stop) and (not tok.is_digit) and (tok.lemma_ != '-PRON-')])
    
    word_cloud_dict=Counter(answer_tokens)

    
    wordcloud = WordCloud(background_color='white',width = 1000, height = 800).generate_from_frequencies(word_cloud_dict)
    
    fontP = FontProperties(weight='bold')
    
    fontP.set_size('xx-large')
    
    plt.figure(figsize=(40,20), facecolor='w')
    
    temp_name = 'outputs/wordcloud.jpg'
    
    plt.imshow(wordcloud)
    
    plt.axis("off")
    
    plt.tight_layout(pad=0)
    
    plt.savefig(temp_name, bbox_inches='tight')   
    
    return temp_name 