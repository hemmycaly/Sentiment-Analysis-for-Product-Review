from flask import Flask, request, jsonify, render_template
import joblib
import spacy

app = Flask(__name__)

# Load the trained SVM model and TF-IDF vectorizer
svm_model = joblib.load('svm_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load spaCy NLP model for aspect extraction
nlp = spacy.load("en_core_web_sm")

# Define aspects to look for in the reviews
aspects = ['acting', 'direction', 'storyline', 'special effects']


# Function to identify aspects in a review
def identify_aspects(review):
    doc = nlp(review)
    review_aspects = []
    for token in doc:
        if token.lemma_ in aspects:
            review_aspects.append(token.lemma_)
    return review_aspects


# Function to get sentiment for each aspect
def aspect_sentiment(review, aspects):
    sentiments = {}
    for aspect in aspects:
        if aspect in review:
            # Find sentences containing the aspect
            aspect_sentence = [sent for sent in review.split('.') if aspect in sent]
            if aspect_sentence:
                joined_sentence = ' '.join(aspect_sentence)

                # Transform the sentence using the TF-IDF vectorizer
                transformed_sentence = tfidf_vectorizer.transform([joined_sentence])

                # Predict sentiment using the SVM model
                sentiment_prediction = svm_model.predict(transformed_sentence)
                sentiment_label = 'positive' if sentiment_prediction == 1 else 'negative'

                sentiments[aspect] = sentiment_label
    return sentiments


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    aspects_in_review = identify_aspects(review)
    sentiments = aspect_sentiment(review, aspects_in_review)

    # Predict overall sentiment
    transformed_review = tfidf_vectorizer.transform([review])
    overall_sentiment_prediction = svm_model.predict(transformed_review)
    overall_sentiment_label = 'positive' if overall_sentiment_prediction == 1 else 'negative'

    return jsonify({
        'review': review,
        'overall_sentiment': overall_sentiment_label,
        'aspects': sentiments
    })


if __name__ == '__main__':
    app.run(debug=True)
