from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import joblib as jl
import pickle

# Create your views here.

sc = jl.load('static/Standard_scaler.joblib')
model = jl.load('static/House_Price_precdictor.joblib')
mail_model = pickle.load(open('static/model.pkl', 'rb'))
vector = pickle.load(open('static/vectorizer.pkl', 'rb'))

def index(request):
    data = pd.read_csv('static/House_Rent_Dataset.csv')
    cities = data['City'].unique()
    bhks = sorted(data['BHK'].unique())
    bathrooms = sorted(data['Bathroom'].unique())

    context = {
        "cities" : cities,
        "bhks" : bhks,
        "bathrooms" : bathrooms,   
    }

    return render(request, 'index.html', context)

@csrf_exempt
def predict(request):    

    df = pd.read_csv('static/sending_format.csv')

    df.drop(columns=['Unnamed: 0'], inplace=True)
    cols = list(df.columns)
    df.columns = [i for i in range(21)]

    if request.method == "POST":
        req_data = request.POST

        df.iloc[0, cols.index(req_data['city'])] = 1
        df.iloc[0, cols.index(req_data['radio'])] = 1
        df.iloc[0, cols.index(req_data['radio1'])] = 1
        df.iloc[0, cols.index(req_data['radio2'])] = 1
        df.iloc[0, cols.index(req_data['radio3'])] = 1
        df.iloc[0, cols.index("BHK")] = int(req_data['BHK'])
        df.iloc[0, cols.index("Size")] = int(req_data['size'])
        df.iloc[0, cols.index("Bathroom")] = int(req_data['bathroom'])


    df.iloc[:, 18:] = sc.transform(df.iloc[:, 18:])

    pred = model.predict([df.iloc[0]])

    return render(request, 'index.html', {'pred': int(pred[0])})


def mail_checker(request):

    return render(request, 'mail.html', { 'mail': None })

@csrf_exempt
def predict_mail(request):

    if request.method == "POST":
        message = request.POST['mail']

    processed_msg = vector.transform([message])

    prediction = mail_model.predict(processed_msg)

    pred_msg = "Not Spam"
    print(prediction[0])
    spam = False

    if prediction[0] == 1: 
        pred_msg = "Spam"
        spam = True

    return render(request, 'mail.html', {'mail': pred_msg, 'spam': spam })