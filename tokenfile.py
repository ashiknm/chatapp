from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("lidiya/bart-large-xsum-samsum")

model = AutoModelForSeq2SeqLM.from_pretrained("lidiya/bart-large-xsum-samsum")