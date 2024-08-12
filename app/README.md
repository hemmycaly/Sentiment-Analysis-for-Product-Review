# Sentiment Analysis App

## Overview
This application performs sentiment analysis on IMDb movie reviews using various machine learning models. It allows users to input movie reviews and get sentiment predictions based on trained models (Support Vector Machines (SVM)). The application demonstrates the capabilities of an optimized SVM model in classifying sentiments as positive or negative as well as performing aspect based analysis.

## Features
- Input movie reviews through a web interface.
- Perform sentiment analysis using pre-trained SVM model.
- Display sentiment predictions along with aspect based analysis.

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-flask-app.git
   cd sentiment-analysis-flask-app
   ```

2. **Create and activate a virtual environment:(OPTIONAL)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
   *Note:
   INSTALLING ANACONDA IS RECOMMENDED AS IT COMES PREINSTALLED WITH MOST OF THE REQUIRED LIBRARIES.*


3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the pre-trained models and dataset:**
   - Ensure you have the trained models saved in the SAME directory.
   ```bash
   python -m spacy download en_core_web_sm
   ```
## Usage
1. **Run the Flask application:**
 - Ensure you open a new terminal the SAME directory as app files.
   ```bash
   flask run
   ```

2. **Access the web application:**
   Open your web browser and navigate to `http://127.0.0.1:5000`.

3. **Input a movie review:**
   - Enter your movie review in the provided text box.
   - Click the "Analyze Sentiment" button to get the prediction.

4. **View the results:**
   - The application will display the predicted sentiment (positive or negative) along with the aspect based analysis of the review.
  
## Project Structure
```
sentiment-analysis-flask-app/
│
├── app.py                 # Main Flask application
├── requirements.txt       # List of dependencies
├── templates/
│   ├── index.html         # Main HTML page
│
├── svm_model.pkl          # Pre-trained SVM model
├── tfidf_vectorizer.pkl   # Pre-traine TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer object
│
└── data/
    └── imdb_reviews.csv   # IMDb dataset (for reference)
```

## Dependencies
- Flask==3.0.3
- Werkzeug==3.0.3
- transformers==4.42.3
- torch==2.3.1
- pandas==1.5.3
- spacy==3.7.5
- gunicorn==20.1.0
- joblib==1.4.0

## Future Work
- Integrate more advanced models for better performance.
- Implement additional preprocessing steps to handle more complex linguistic features.
- Expand the application to handle multiple languages and more diverse datasets.
- Improve the UI/UX for a better user experience.



## Contact
If you have any questions or suggestions, feel free to contact me.

---
