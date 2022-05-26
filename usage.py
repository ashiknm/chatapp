from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/")
def hello():
    return jsonify({"about":"Hello World"})

@app.route("/post", methods=['GET','POST'])
def index():
    if(request.method == 'POST'):
        some_json = request.get_json()
        return jsonify({'you sent' :some_json}),201
    else:
        return jsonify({"about":"hello world"})

@app.route("/multi/<int:num>",methods = ["GET"])
def get_multiply10(num):
    return jsonify({"result":num*10})

# summarizer = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
# model = BertModel.from_pretrained("./test/saved_model/")
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


summarizer = pipeline("summarization", model="lidiya/bart-large-xsum-samsum")

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# tokenizer = AutoTokenizer.from_pretrained("philschmid/bart-large-cnn-samsum")

# summarizer = AutoModelForSeq2SeqLM.from_pretrained("philschmid/bart-large-cnn-samsum",from_tf=True)
# model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
# tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')


@app.route("/summary",methods = ["POST"])
def summary():

    sentence = request.args.get('summarytext')
    # conversation = '''Jeff: Can I train a 🤗 Transformers model on Amazon SageMaker? 
    # Philipp: Sure you can use the new Hugging Face Deep Learning Container. 
    # Jeff: ok.
    # Jeff: and how can I get started? 
    # Jeff: where can I find documentation? 
    # Philipp: ok, ok you can find everything here. https://huggingface.co/blog/the-partnership-amazon-sagemaker-and-hugging-face                                           
    # '''
    
    conversation = '''Human 1: Hi!
Human 2: What is your favorite holiday?
Human 1: one where I get to meet lots of different people.
Human 2: What was the most number of people you have ever met during a holiday?
Human 1: Hard to keep a count. Maybe 25.
Human 2: Which holiday was that?
Human 1: I think it was Australia
Human 2: Do you still talk to the people you met?
Human 1: Not really. The interactions are usually short-lived but it's fascinating to learn where people are coming from and what matters to them
Human 2: Yea, me too. I feel like God often puts strangers in front of you, and gives you an opportunity to connect with them in that moment in deeply meaningful ways. Do you ever feel like you know things about strangers without them telling you?
Human 1: what do you mean?
Human 2: I think it's like a 6th sense, often seen as "cold readings" to people, but can be remarkably accurate. I once sat next to a man in a coffee and I felt a pain in my back. I asked the stranger if he had a pain. It turns out that he did in the exact spot, and said he pulled a muscle while dancing at a party. I had never met the man before and never saw him again.
Human 1: Wow! That's interesting, borderline spooky
Human 2: There's this practice called "Treasure Hunting" that's kind of a fun game you play in a public place. There's a book called "The Ultimate Treasure Hunt" that talks about it. You use your creativity to imagine people you will meet, and you write down a description, then you associate them with a positive message or encouraging word. Maybe you saw a teenage boy in a red hat at the shopping mall in your imagination, then while at the mall, you may find someone who matches that description. You show that you have a message for him and that you have a message for a boy in a red hat. You then give him a message of kindness or whatever was on your heart. You have no idea, sometimes you meet someone who is having a really hard day, and it brings them to tears to have a stranger show them love.
Human 1: So, do you do treasure hunting often?
Human 2: I did more when I was in grad school (and had more time). I would usually go with friends. For a while I would go to the farmers market in Santa Cruz every week and try to feel if there is something I am supposed to tell a stranger. Usually, they are vague hope-filled messages, but it's weird when I blurt out something oddly specific.                                        
    '''
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