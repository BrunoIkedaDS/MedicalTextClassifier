# This script will be served by the server application as the utility functions.
import re
from nltk.corpus import stopwords
from joblib import load

# Text Preprocessing and TfidfVectorizer.
Stopwords = stopwords.words('english')
vectorizer = load('models/vectorizer.joblib')
def preprocess(text):
    """This function preprocesses the text"""

    # Convert to lowercase
    text = text.lower()

    # Removing punctuation and html tags
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'<.*?>+', '', text)

    # Removing multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Finally, removing stopwords
    text = " ".join([word for word in text.split() if word not in Stopwords])

    text_vectorized = vectorizer.transform([text])

    return text_vectorized
