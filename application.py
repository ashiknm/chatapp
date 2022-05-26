from flask import Flask, jsonify, request
from transformers import pipeline

app = Flask(__name__)

@app.route("/")
def hello():
    return jsonify({"about":"Hello World"})

# summarizer = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
# model = BertModel.from_pretrained("./test/saved_model/")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# tokenizer = AutoTokenizer.from_pretrained("lidiya/bart-large-xsum-samsum")

# summarizer = AutoModelForSeq2SeqLM.from_pretrained("lidiya/bart-large-xsum-samsum")

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# tokenizer = AutoTokenizer.from_pretrained("philschmid/bart-large-cnn-samsum")

# summarizer = AutoModelForSeq2SeqLM.from_pretrained("philschmid/bart-large-cnn-samsum",from_tf=True)
# model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
# tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')


@app.route("/summary",methods = ["POST"])
def summary():

    sentence = request.args.get('summarytext')
    processed  = summarizer(sentence) 
    return jsonify({
            "summaryText": processed[0]["summary_text"]
    })
  


    # ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
    # inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')

    # # Generate Summary
    # summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
    # print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])


if __name__ == '__main__':
    app.run(debug=True, threaded=True)