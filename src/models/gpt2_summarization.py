from summarizer import TransformerSummarizer

def gpt2_summarizer(text_data, min_len=60):
  
    # initialize the gpt2 model
    gpt2_model = TransformerSummarizer(transformer_type="GPT2", transformer_model_key="gpt2-medium")
    
    # generate the summary
    gpt2_summary = gpt2_model(text_data, min_length=min_len)

    return gpt2_summary, gpt2_model