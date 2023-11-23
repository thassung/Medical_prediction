import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, ClientsideFunction

import numpy as np
import pandas as pd
import datetime
from datetime import datetime as dt
import pathlib
import pickle

import dash_bootstrap_components as dbc

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Disease Prognosis Dashboard"

server = app.server
app.config.suppress_callback_exceptions = True

# Path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()   

# Read data
# df = pd.read_csv(DATA_PATH.joinpath("disease prognosis.csv")) 
# df.dropna(axis=1, inplace=True)

# symptoms_list = df.drop(["prognosis"], axis=1).columns.tolist()
symptoms_list = ['itching', 'skin_rash', 'continuous_sneezing', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 
                 'muscle_wasting', 'burning_micturition', 'spotting_ urination', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 
                 'mood_swings', 'weight_loss', 'restlessness', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 
                 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 
                 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 
                 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'swelling_of_stomach', 'swelled_lymph_nodes', 
                 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'congestion', 
                 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'bloody_stool', 'neck_pain', 
                 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 
                 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'drying_and_tingling_lips', 
                 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 
                 'movement_stiffness', 'spinning_movements', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 
                 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 
                 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 
                 'belly_pain', 'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes', 'increased_appetite', 
                 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 
                 'coma', 'stomach_bleeding', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 
                 'shivering_or_chills', 'unprotected_blood/sex_transfusion', 'pain/irritation_in_anal_region']

def description_card():
    """

    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5("Disease Prognosis"),
            html.H3("Welcome to the Disease Prognosis Dashboard"),
            html.Div(
                id="intro",
                children="Explore your symptoms and potential diseases. Select your ongoing symptoms and contact to the nearby hospital if needed.",
            ),
        ],
    )


def generate_control_card():
    """

    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[
            html.P("Enter Your Gender"),
            dcc.Dropdown(
                id="gender-select",
                options=[{"label": 'Male', "value": 0},
                         {"label": 'Female', "value": 1}],
                value=0,
            ),
            html.Br(),
            html.P("Enter Your Height (cm)"),            
            dcc.Slider(120, 200, 1,
               value=160, marks=None,
               tooltip={"placement": "bottom", "always_visible": True},
               id='height-slider'
            ),
            html.Br(),
            html.P("Enter Your Weight (kg)"),
            dcc.Slider(0, 200, 1, 
               value=60, marks=None,
               tooltip={"placement": "bottom", "always_visible": True},
               id='weight-slider'
            ),
            html.Br(),
            html.Br(),
            html.P("Select Your Symptoms"),
            dcc.Dropdown(
                id="symptom-select",
                options=[{"label": i, "value": i} for i in symptoms_list],
                value=symptoms_list[:],   # initial selected value = all
                multi=True,
            ),
            html.Br(),
            html.Div(
                id="reset-btn-outer",
                children=html.Button(id="reset-btn", children="Reset", n_clicks=0),
            ),
        ],
    )

def generate_column_text(predicted_diseases, symptoms):
    """
    disease is str ex. 'Allergy'
    predicted_diseases is list of three most probable diseases
    symptoms is list of selected symptom input from user ex. ['itching', 'shivering_or_chills']
    """
    # common_symptom_df = pd.read_csv('./Dataset/Common symptoms of each disease.csv')
    # common_symptoms = common_symptom_df.loc[disease][common_symptom_df.loc[disease] == 1].index.tolist()
    # 
    # matched, found, expected = [], [], []
    # for s in set(symptoms + common_symptoms):
    #     if s in common_symptoms and s in symptoms:
    #         matched.append(s)
    #     elif s in symptoms and s not in common_symptoms:
    #         found.append(s)
    #     elif s not in symptoms and s in common_symptoms:
    #         expected.append(s)
    # return matched, found, expected

    common_symptom_df = pd.read_csv('code\Dataset\Common symptoms of each disease.csv', index_col='prognosis')
    col = set(symptoms)
    for disease in predicted_diseases:
        col.update(common_symptom_df.loc[disease][common_symptom_df.loc[disease] == 1].index.tolist())
    col = list(col)
    indexes = ['Your Symptoms']
    indexes.extend(predicted_diseases)
    symptom_hm_df = pd.DataFrame(0, columns=col, index=indexes)
    for idx in predicted_diseases:
        for c in col:
            symptom_hm_df.loc[idx][c] = common_symptom_df.loc[idx][c]
    for s in symptoms:
        symptom_hm_df.loc['Your Symptoms'][s] = 1
    symptom_hm_df = symptom_hm_df.reset_index()

    for c in symptom_hm_df.columns:
        new_c = c.split('_')
        new_c = ' '.join(new_c)
        symptom_hm_df.rename(columns={c: new_c.capitalize()}, inplace=True)

    return html.Div(
                        [
                            dash_table.DataTable(
                                data=symptom_hm_df.to_dict("records"),
                                columns=[{"id": x, "name": x} for x in symptom_hm_df.columns],
                                style_table={'overflowX': 'scroll'},
                                style_cell={'fontSize':20, 'font-family':'sans-serif'},
                            )
                        ]
                    )

app.layout = html.Div(
    id="app-container",
    children=[
        # Banner
        html.Div(
            id="banner",
            className="banner",
            children=[html.Img(src=app.get_asset_url("plotly_logo.png"))],
        ),
        # Left column
        html.Div(
            id="left-column",
            className="four columns",
            children=[description_card(), generate_control_card()]
            + [
                html.Div(
                    ["initial child"], id="output-clientside", style={"display": "none"}
                )
            ],
        ),
        # Right column
        html.Div(
            id="right-column",
            className="eight columns",
            children=[
                # Patient Volume Heatmap
                html.Div(
                    id="disease_symptom_card",
                    children=[
                        html.B("Disease Symptoms"),
                        html.Hr(),
                        html.Div(id='disease_column_text'),
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("disease_column_text", "children"),
    [
        Input("symptom-select", "value"),
        Input("reset-btn", "n_clicks"),
    ],
)
def update_column_text(symptoms, reset_click):
    symptoms_df = pd.DataFrame({col: [0] for col in symptoms_list})
    for s in symptoms:
        symptoms_df[s] = 1
    model = pickle.load(open('code\model\model.pkl', 'rb'))
    # predicted_disease = model.predict(symptoms_df)
    top_classes = model.predict_proba(symptoms_df)
    top_classes = np.argsort(top_classes, axis=1)[:, -3:][:, ::-1]
    labels = model.classes_

    predicted_diseases = []

    for c in top_classes:
        label = labels[c]
        predicted_diseases.append(label)
    
    predicted_diseases = predicted_diseases[0]

    reset = False
    # Find which one has been triggered
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "reset-btn":
            reset = True

    # Return to original hm(no colored annotation) by resetting
    return generate_column_text(predicted_diseases, symptoms)

# app.clientside_callback(
#     ClientsideFunction(namespace="clientside", function_name="resize"),
#     Output("output-clientside", "children"),
#     [Input("wait_time_table", "children")] + wait_time_inputs + score_inputs,
# )

@app.callback(
    Output("insurance_column", "children"),
    [
        Input("gender-select", "value"),
        Input("height-slider", "value"),
        Input("weight-slider", "value"),
        Input("symptom-select", "value"),
        Input("reset-btn", "n_clicks"),
    ],
)
def insurance_pick(gender, height, weight, symptoms, reset_click):

    col = ['age', 'sex', 'bmi', 'children', 'smoker', 'disease', 'sed_pd',
       'junk_food_consumption', 'alcohol_consumption', 'exercise_routine',
       'substance_use', 'stress_level', 'northwest', 'southeast', 'southwest',
       'age_group']
    
    age = forms.getdata('age')
    # ...

    input = [age, sex, weight, height, children, smoker, disease, sed_pd,
       junk_food_consumption, alcohol_consumption, exercise_routine,
       substance_use, stress_level, region]
    
    # df = pd.DataFrame({input:input})
    df['bmi'] = df['weight'] / np.square((df['height']/100))
    df[['nortwest', 'southeast', 'southwest']] = 0
    df[region] = 1
    df['age_group'] = pd.cut(df['age'], bins=[0,20,40,60,100], labels=[0,1,2,3])
    df = df[col]
    
    model = pickle.load(open('code\model\insurance_model.pkl', 'rb'))
    # predicted_disease = model.predict(symptoms_df)
    pred_insu = model.predict(df)
    
    # Return to original hm(no colored annotation) by resetting
    return pred_insu

# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)
