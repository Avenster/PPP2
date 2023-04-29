import pickle
import string
import re
import pandas as pd
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Creating a function to process the data
def text_process(text):
    pattern = r'https?://\S+|www\.\S+'
    text = re.sub(pattern, '', text)
    text = text.lower()
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'ð', '', text)
    text = re.sub(r'rt', '', text)

    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

with open('hsdc.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Hate Speech Detection')

# Creating the text input field
text = st.text_input('Enter your text here:')

# Creating the prediction button
if st.button('Predict'):
    s = pd.Series([text])
    y_pred = model.predict(s)[0]
    if y_pred == "Hate_Speech":
        st.write('This is a hate speech.')
    elif y_pred == "Offensive_Speech":
        st.write('This is an offensive speech.')
    else:
        st.write('This is a safe speech.')
