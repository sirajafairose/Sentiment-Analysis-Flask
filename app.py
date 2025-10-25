from flask import Flask, request, render_template
import pickle, string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

app = Flask(__name__)

# Load saved model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = ""
    if request.method == 'POST':
        review = request.form['review']
        review_clean = ' '.join([word for word in review.lower().translate(str.maketrans('', '', string.punctuation)).split() if word not in ENGLISH_STOP_WORDS])
        vect = vectorizer.transform([review_clean])
        prediction = model.predict(vect)[0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
