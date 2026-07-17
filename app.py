# app.py

import os
import gradio as gr
import joblib

# Load the trained Decision Tree model at startup
deployed_dt = joblib.load('diabetes_prediction_model.pkl')

# --- CODE BLOCK: PREDICTION LOGIC FOR 5 FEATURES ---
def predict_diabetes(pregnancies, glucose, insulin, bmi, age):
    # The model expects a 2D array matching the exact order of x_train
    input_data = [[pregnancies, glucose, insulin, bmi, age]]
    prediction = deployed_dt.predict(input_data)
    
    # Interpret the binary outcome (typically 1 for positive, 0 for negative)
    if prediction[0] == 1:
        return "Prediction: High Risk of Diabetes (Positive)"
    else:
        return "Prediction: Low Risk of Diabetes (Negative)"
# ---------------------------------------------------

# --- CODE BLOCK: GRADIO INTERFACE SETUP ---
interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies (Number of times pregnant)"),
        gr.Number(label="Glucose (Plasma glucose concentration)"),
        gr.Number(label="Insulin (2-Hour serum insulin)"),
        gr.Number(label="BMI (Body mass index)"),
        gr.Number(label="Age (Years)")
    ],
    outputs=gr.Text(label="Assessment Result"),

    title="🩺 Diabetes Prediction System",

    description="""
    ## Diabetes Prediction using Decision Tree Machine Learning

    Enter the patient's medical metrics to predict diabetes risk.

    **Developer Details**
    - **Name:** Rohit
    - **Contact:** 8708261681
    - **Email:** kashyaprohit03456@gmail.com
    """,

    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan",
        neutral_hue="slate",
    ),

    css="""
    .gradio-container{
        background-image: url('https://images.unsplash.com/photo-1576091160550-2173dba999ef?auto=format&fit=crop&w=1600&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    .gradio-container::before{
        content:"";
        position:fixed;
        inset:0;
        background:rgba(255,255,255,0.82);
        z-index:-1;
    }

    .block{
        border-radius:18px !important;
        box-shadow:0 8px 25px rgba(0,0,0,0.15);
    }

    h1{
        text-align:center;
        color:#0B5ED7;
        font-weight:800;
    }
    """,
)
# ------------------------------------------

if __name__ == "__main__":
    # Render network configuration
    interface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
