import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from pony.orm import *
from models import db
from datetime import date

st.set_page_config(page_title="Sentiment", page_icon="🎬")

# инициализируем элемент класса Mystem() для последующей лемматизации  текста на русском языке

label_names = {1: 'pos', 0: 'neg'}

with open('data/pkl_object/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
    
with open('data/pkl_object/logit.pkl', 'rb') as f:
    logit = pickle.load(f)
    
day = date.today()    
    
# функция записи данных в таблицу 'Sentiment'
@db_session
def add_sentiment(day, review):
    db.Sentiment_New(date=day, review=review)
    # commit() will be done automatically
    # database session cache will be cleared automatically
    # database connection will be returned to the pool    
         
@st.cache(allow_output_mutation=True)

# функция трансформации текста договора для предсказания
def review_lower(review):
    
    return pd.Series(review.lower())

# функция предсказания
def ModelPredictProba(review, label_names, vectorizer, logit):
    
    # применение векторайзера
    review = vectorizer.transform(review)
    label_pred = label_names[logit.predict_proba(review).argmax()]
    score = logit.predict_proba(review).max()
    
    return label_pred, score 
    
st.write(
    """
## 🎬 Классификатор комментариев (отзывов) к фильмам
"""
)

st.write('###')

st.write('**Оставьте свой комментарий (отзыв)**')

review_text = st.text_area(label='Оставьте свой комментарий (отзыв)', label_visibility="hidden")

if review_text:              
    st.success("Комментарий (отзыв) принят", icon="✅")
    result = st.button('Классифицировать')
    if result:
        review = review_lower(review_text)
        label_pred, score = ModelPredictProba(review, label_names, vectorizer, logit)
        add_sentiment(day, review_text)
        st.success("Комментарий (отзыв) классифицирован успешно", icon="✅")
        
        st.write(f"""**Результаты:**
               
    ✔️ Предсказанный статус:  {label_pred}
    
    ✔️ Вероятность уверенности алгоритма:  {round(score, 2)}%
       
        """)

else:
    st.error("Вы ничего не ввели", icon="❌")
    
st.markdown("<h5 style='text-align: center; color: blac;'> ©️ Designed by Ruslan Latipov </h5>", unsafe_allow_html=True)