#Removing extra spaces and newlines from data
import re
import string
import unicodedata
import contractions
import nltk
from nltk.corpus import stopwords


text = "  This   is  a   test   text   with    extra   spaces.   \n\n"
clean_text = re.sub(r'\s+', ' ', text).strip()
print(clean_text)  # "This is a test text with extra spaces."

#convert to lowercase
text = "HELLO World!"
print(text.lower())  # "hello world!"

#Remove punctiaction and special characters
text_punc = "Wow!!! This product is amazing!!! 🤯🔥"

clean_text_punc = text_punc.translate(str.maketrans('', '', string.punctuation))
print(clean_text_punc)  # "Wow This product is amazing"

# Remove numbers
text = "My order number is 123456 and I paid 50 dollars."
clean_text = re.sub(r'\d+', '', text)
print(clean_text)  # "My order number is  and I paid  dollars."

"""Advanced Text extraction techniques"""
# Standardizing common terms
common_terms = {
    "Dr.": "Doctor",
    "NYC": "New York",
    "N.Y.": "New York",
    "USD": "dollars",
    "$": "dollars"
}

def standardize_terms(text, replacements):
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text
 #Example run through
text = "Dr. Smith is based in NYC and charges 100 USD."
clean_text = standardize_terms(text, common_terms)
print(clean_text)


#Expanding Contractions
text = "I'll be there, but I don't know if he's coming."
expanded_text = contractions.fix(text)
print(expanded_text)


#Intresting case: handaling stop words
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)
#Example
text = "This is an example of text with some stopwords."
clean_text = remove_stopwords(text)
print(clean_text) # Only gives out example text stopwords