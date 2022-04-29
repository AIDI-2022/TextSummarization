from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en import English
import numpy as np

def tfidf_summarizer(text, max_sent_in_summary=3):
    
    tokenizer = English()
    tokenizer.add_pipe('sentencizer')
    
    tokenized_sentences = tokenizer(text.replace("\n",""))
    sentences = [str(sent).strip() for sent in tokenized_sentences.sents]
    
    sentence_organizer = {k:v for v,k in enumerate(sentences)}
    
    tf_idf_vectorizer = TfidfVectorizer(max_features=None,
                                        strip_accents='unicode',
                                        analyzer='word',
                                        token_pattern=r'\w{1,}',
                                        ngram_range=(1,3),
                                        use_idf=1,smooth_idf=1,
                                        sublinear_tf=1,
                                        stop_words='english')
    tf_idf_vectorizer.fit(sentences)
    
    sentence_vectors = tf_idf_vectorizer.transform(sentences)
    sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
    
    top_n_sentences = [sentences[idx] for idx in np.argsort(sentence_scores, axis=0)[::-1][:max_sent_in_summary]]
    
    mapped_top_n_sentences = [(sentence, sentence_organizer[sentence]) for sentence in top_n_sentences]
    mapped_top_n_sentences = sorted(mapped_top_n_sentences, key= lambda x: x[1])
    
    ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
    
    summary = " ".join(ordered_scored_sentences)
    
    return summary