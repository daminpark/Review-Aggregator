import nltk
import ssl

# Workaround for SSL certificate verification failure on Mac
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("Downloading NLTK data...")
nltk.download('brown')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('conll2000') # Required for noun_phrases
nltk.download('movie_reviews')
print("Download complete.")
