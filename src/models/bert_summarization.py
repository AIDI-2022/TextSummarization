from summarizer import Summarizer

def bert_summarizer(text_data, min_len=60):
    
    # initialize the bert model
    bert_model = Summarizer()
    
    # generate the summary
    bert_summary = ''.join(bert_model(text_data, min_length=min_len))
    
    return bert_summary, bert_model