import joblib
from django.http import JsonResponse,HttpResponse
import numpy as np # linear algebra
import pandas as pd #data processing
import os
import re
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
import nltk
import sklearn

pipeline = joblib.load('./pipeline.sav')

def index(request):
    return HttpResponse("<h1>Fake news API</h1>")

def newsApi(request):
    print("TEST:  ",request.GET.get('query'))
    news = remove_punctuation_stopwords_lemma(str(request.GET.get('query')))
    print([news])
    pred = pipeline.predict([news])
    print(pred)
    dic = {1:'real',0:'fake'}
    return JsonResponse(dic[pred[0]], safe=False)




def remove_punctuation_stopwords_lemma(sentence):
    filter_sentence = ''
    lemmatizer=WordNetLemmatizer()
    sentence = re.sub(r'[^\w\s]','',sentence)
    words = nltk.word_tokenize(sentence) #tokenization
    words = [w for w in words if not w in stopwords.words('english')]
    for word in words:
        filter_sentence = filter_sentence + ' ' + str(lemmatizer.lemmatize(word)).lower()
    return filter_sentence