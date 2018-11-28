#Code for creating a web application on loopback IP address inorder to analyse a tweet
#imorting pickle inorder to load the object from a file object
#importing re inorder to create expressions

import os
import pickle 
import re

#importing flask inorder to create a web application

from flash import Flask,request,jsonify

# Unpickle the trained classifier and write preprocessor method used
def tokenizer(text):
  return text.split(' ')
  
def preprocessor(text):
  """ Return a cleaned version of text"""
  text = re.sub('<[^>]*>', '', text)
  # Save emoticons for later appending
  emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
  # Remove any non-word character and append the emoticons,
  # removing the nose character for standarization. Convert to lower case
  text = (re.sub('[\W]+', ' ', text.lower()) + ' ' + ' '.join(emoticons).replace('-', ''))
  return text

tweet_classifier = pickle.load(open('../data/logisticRegression.pkl', 'rb'))
app = Flask(__name__, static_folder='static') #Construct an instance of Flask class for our webapp

@app.route('/') # URL '/' to be handled by index() route handler
def index():
    return app.send_static_file('html/index.html')
 
@app.route('/classify', methods=['POST']) #URL '/' to be handled by classify() route handler
def classify():
    text = request.form.get('text', None)
    assert text is not None
    prob_neg, prob_pos = tweet_classifier.predict_proba([text])[0] #Calculate the probability
    s = 'Positive' if prob_pos >= prob_neg else 'Negative'
    p = prob_pos if prob_pos >= prob_neg else prob_neg
    return jsonify({
        'sentiment': s,
        'probability': p
    })
app.run() #Launch built-in web server and run this Flask webapp



    
   
