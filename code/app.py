from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
import joblib
import pickle
import pathlib


app = Flask(__name__, template_folder = './template/public')

model_disease = joblib.load('./model/model.pkl')
model_insurance = joblib.load('./model/insurance_model.pkl')

@app.route("/", methods = ['GET'])
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict_insurance():
    data = request.get_json()
    # symptom = data.get('symptom')
    exercise = data.get('exerciseFrequency')
    print('='*50)
    print(exercise)


    # symptoms_list = ['itching', 'skin_rash', 'continuous_sneezing', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 
    #              'muscle_wasting', 'burning_micturition', 'spotting_ urination', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 
    #              'mood_swings', 'weight_loss', 'restlessness', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 
    #              'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 
    #              'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 
    #              'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'swelling_of_stomach', 'swelled_lymph_nodes', 
    #              'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'congestion', 
    #              'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'bloody_stool', 'neck_pain', 
    #              'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 
    #              'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'drying_and_tingling_lips', 
    #              'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 
    #              'movement_stiffness', 'spinning_movements', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 
    #              'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 
    #              'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 
    #              'belly_pain', 'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes', 'increased_appetite', 
    #              'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 
    #              'coma', 'stomach_bleeding', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 
    #              'shivering_or_chills', 'unprotected_blood/sex_transfusion', 'pain/irritation_in_anal_region']
    # input = pd.DataFrame(data={
    #     'Diabetes' : [diabetes],
    #     'Cholesterol' : [cholesterol],
    #     'Systolic BP' : [bp],
    #     'Continent' : [continent],
    #     'Sleep Hours Per Day' : [sleep],
    #     'Exercise Hours Per Week' : [exercise],
    #     'Triglycerides' : [triglycerides],
    #     'Previous Heart Problems' : [heartprob],
    #     'Obesity' : [obesity],
    #     'Age' : [age],
    #     'Death Rate' : [deathrate],
    #     'Alcohol Consumption' : [alcohol]
    # })
    
    # input = input.replace('', None)

    # imputer = pickle.load(open('./preprocessor/imputer.pkl', 'rb'))
    # transformer = pickle.load(open('./preprocessor/transformer.pkl', 'rb'))

    # fmed = ['Cholesterol', 'Systolic BP', 'Sleep Hours Per Day', 'Exercise Hours Per Week', 
    #         'Triglycerides', 'Age', 'Death Rate']
    # fmode = ['Diabetes', 'Continent', 'Previous Heart Problems', 'Obesity', 
    #          'Alcohol Consumption']

    # input = imputer.transform(input)
    # input = pd.DataFrame(input, columns=fmed+fmode)
    # input = transformer.transform(input)

    # prediction = model.predict(input)[0]
    # message = ''
    # if prediction == 0:
    #     message = 'You are not at risk of Heart Attack'
        
    # if prediction == 1:
    #     message = 'You are at risk of Heart Attack'
    # ###result
    # result = {
    #     'message': message
    # }
    # print(message)
    # return jsonify(result), 200

    return print('exercise')
        
if __name__ == "__main__":
    app.run(host='127.0.0.1',port='8000',debug=True)