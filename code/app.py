from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
import joblib
import pickle
import pathlib
import csv
from datetime import datetime
from forex_python.converter import CurrencyRates
import requests

app = Flask(__name__, template_folder = './template/public',static_folder='./template/public')

model_disease = joblib.load('./model/model.pkl')
model_insurance = joblib.load('./model/insurance_model.pkl')

def convert_currency(amount, to_currency, from_currency='USD', api_key='64dd72977amsh213af99dbf31037p1734eejsndf1ee1db9184'):
    endpoint = "https://currency-exchange.p.rapidapi.com/exchange"
    headers = {
        "X-RapidAPI-Host": "currency-exchange.p.rapidapi.com",
        "X-RapidAPI-Key": api_key,
    }
    params = {
        "from": from_currency,
        "to": to_currency,
        "q": amount,
    }
    print(params)
    if from_currency == to_currency:
        converted_amount = amount
    else:
        try:
            response = requests.get(endpoint, headers=headers, params=params)
            print(response)
            if response.status_code == 200:
                converted_amount = float(response.text) * amount
                # print(f"{amount} {from_currency} is equal to {converted_amount:.2f} {to_currency}")
            else:
                print(f"Error: {response.status_code}, {response.text}")
        except Exception as e:
            print(f"Error: {e}")

    return converted_amount

def read_region_from_csv(file_path):
    unique_regions = set()

    with open(file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        # Assuming the first row contains column headers
        if 'region' in csv_reader.fieldnames:
            for row in csv_reader:
                unique_regions.add(row['region'])

    return sorted(list(unique_regions))  # Convert set to list and sort

def read_child_from_csv(file_path):
    unique_chilren = set()

    with open(file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        # Assuming the first row contains column headers
        if 'children' in csv_reader.fieldnames:
            for row in csv_reader:
                unique_chilren.add(row['children'])
    re = sorted(list(unique_chilren))
    re2 = re[:-1]
    re2.append(str(re[-1])+'+')

    return re2

# def get_common_symptoms(file_path):
#     df = pd.read_csv(file_path)
#     common_symptoms = {}

#     for disease in df['prognosis'].unique():
#         disease_df = df[df['prognosis'] == disease]
#         symptom_counts = disease_df.sum(axis=0)
#         common_symptoms[disease] = set(symptom_counts[symptom_counts > len(disease_df) * 0.5].index)

#     return common_symptoms

def convert_insurance_price(insurance_class, to_currency):
    bins = [1000, 22500, 43500, 65000]
    price_range = [bins[insurance_class], bins[insurance_class+1]]

    lower_end = convert_currency(price_range[0], to_currency)
    higher_end = convert_currency(price_range[1], to_currency)
    
    yearly = (lower_end//1, higher_end//1)
    quarterly = (lower_end//4, higher_end//4)
    monthly = (lower_end//12, higher_end//12)
    return yearly, quarterly, monthly

@app.route("/", methods = ['GET'])
def home():
    file_path = './data/disease_insurance_price_final.csv'
    regions = read_region_from_csv(file_path)
    children = read_child_from_csv(file_path)
    return render_template('index.html', regions=regions, children=children)

@app.route("/disease", methods=['POST'])
def predict_disease():
    data = request.get_json()
    symptomData = data.get('symptomData')
    disease_model = joblib.load('./model/disease_model.pkl')
    column_names = ['itching', 'skin_rash', 'continuous_sneezing', 'stomach_pain', 
                    'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'bloody_stool', 'neck_pain', 'cramps', 'bruising', 'obesity', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'movement_stiffness', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'shivering_or_chills', 'unprotected_blood/sex_transfusion', 'pain/irritation_in_anal_region', 'swollen_painful_joints']
    symptomData = pd.DataFrame([symptomData], columns=column_names)
    prediction = disease_model.predict(symptomData)

    # Get label
    predicted_not_in_insurance = ['(vertigo) Paroymsal  Positional Vertigo', 'Chicken pox', 
            'Common Cold', 'Diabetes ', 'Dimorphic hemmorhoids(piles)', 
            'GERD', 'Hepatitis C', 'Hepatitis E', 'Hypertension ', 
            'Osteoarthristis', 'Peptic ulcer diseae', 'Pneumonia', 'hepatitis A']
    mapped = ['Paroxysmal Positional Vertigo (Vertigo)', 'Chickenpox',
            'Allergy', 'Diabetes', 'Dimorphic hemorrhoids (piles)',
            'GERD (Gastroesophageal Reflux Disease)', 'Hepatitis A', 'Hepatitis A', 'Heart attack',
            'Arthritis', 'GERD (Gastroesophageal Reflux Disease)', 'Typhoid', 'Hepatitis A']
    label = prediction[0]
    if label in predicted_not_in_insurance:
        idx = predicted_not_in_insurance.index(label)
        label = mapped[idx]

    encoder = joblib.load('./model/disease_encoder.pkl')
    label = encoder.transform(pd.Series(label))
    label = label[0]

    # Saving the district and country of user and his prognosis
    userLocation = data.get('userLocation')
    district = userLocation.get('district')
    country = userLocation.get('country')

    # Append the district to the CSV file
    prediction_result = prediction[0]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    data_to_be_saved = {
        "PredictionResult": [prediction_result],
        "Timestamp": [timestamp],
        "District": [district],  
        "Country": [country],
    }

    disease_log = pd.DataFrame(data_to_be_saved)

    csv_file_path = "./data/predictions_log.csv"
    disease_log.to_csv(csv_file_path, mode="a", header=not pd.io.common.file_exists(csv_file_path), index=False)
    ###########################

    result = {'message': f'Disease is {prediction[0]}',
                'disease': prediction[0],
                'disease_label': int(label)}
    return jsonify(result)

@app.route("/predict", methods=['POST'])
def predict_insurance():
    if request.method == 'POST':
        data = request.get_json()
        for key, value in data.items():
            try:
                data[key] = float(value)
            except (ValueError, TypeError):
                pass
    
        exercise = data.get('exercise')
        junk_food = data.get('junkFood')
        smoking = data.get('smoking')
        alcohol = data.get('alcohol')
        sedentary = data.get('sedentary')
        stress = data.get('stress')
        drug = data.get('drug')
        age = data.get('age')
        sex = data.get('sex')
        weight = data.get('weight')
        height = data.get('height')
        children = data.get('children')
        region = data.get('region')
        disease = data.get('disease')

        bmi = weight/((height/100)**2)

        children = int(str(children)[0])
        
        northwest, southeast, southwest = 0, 0, 0
        if region == 'northwest':
            northwest = 1
        elif region == 'southeast':
            southeast = 1
        elif region == 'southwest':
            southwest = 1
        
        age_bin = [((float('-inf'),25), 3), ((25,40), 0), ((40,60), 1), ((60,float('inf')), 2)]
        for b, l in age_bin:
            if age > b[0] and age <= b[1]:
                age_group = l
                break

        
        input_df = pd.DataFrame({'age': [age],
                                 'sex': [sex],
                                 'bmi': [bmi],
                                 'children': [children],
                                 'smoker': [smoking],
                                 'disease': [disease],  ###############
                                 'sed_pd': [sedentary],
                                 'junk_food_consumption': [junk_food],
                                 'alcohol_consumption': [alcohol],
                                 'exercise_routine': [exercise],
                                 'substance_use': [drug],
                                 'stress_level': [stress],
                                 'northwest': [northwest],
                                 'southeast': [southeast],
                                 'southwest': [southwest],
                                 'age_group': [age_group]
                                 })

        scaler = joblib.load('./model/scaler.pkl')
        insurance_model = joblib.load('./model/insurance_model.pkl')

        input_df = scaler.transform(input_df)
        prediction = insurance_model.predict(input_df)       

        to_currency = data.get('to_currency')
        print(to_currency)
        yearly, quarterly, monthly = convert_insurance_price(prediction[0], to_currency)
        insurance_plan = ['Basic', 'Priority', 'Premium'][prediction[0]]

        result = {'message': f'Insurance type is {int(prediction[0])}',
                  'type': int(prediction[0]),
                  'text11_header': f'Monthly {insurance_plan} Plan',
                  'text11': f'The pricing ranges from {monthly[0]} to {monthly[1]} {to_currency}',
                  'text12_header': f'Quarter {insurance_plan} Plan',
                  'text12': f'The pricing ranges from {quarterly[0]} to {quarterly[1]} {to_currency}',
                  'text13_header': f'Yearly {insurance_plan} Plan',
                  'text13': f'The pricing ranges from {yearly[0]} to {yearly[1]} {to_currency}',
                  }
        return jsonify(result)
    else:
        return jsonify({'error': 'Invalid request method'})
    
        
if __name__ == "__main__":
    app.run(host='127.0.0.1',port='8000',debug=True)