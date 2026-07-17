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

  theme=gr.themes.Base(
    primary_hue="emerald",
    secondary_hue="teal",
    neutral_hue="slate",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Poppins"), "sans-serif"],
),

css="""
/* ===========================
   Background
=========================== */

body{
    background:
    radial-gradient(circle at top left,#0f766e 0%,transparent 35%),
    radial-gradient(circle at bottom right,#14532d 0%,transparent 35%),
    linear-gradient(135deg,#081c24,#0b1726,#111827);
    overflow-x:hidden;
}

/* Floating animated blobs */

body::before{
    content:"";
    position:fixed;
    width:500px;
    height:500px;
    border-radius:50%;
    background:rgba(16,185,129,.18);
    filter:blur(90px);
    top:-150px;
    left:-120px;
    animation:float1 14s ease-in-out infinite;
}

body::after{
    content:"";
    position:fixed;
    width:450px;
    height:450px;
    border-radius:50%;
    background:rgba(20,184,166,.18);
    filter:blur(90px);
    right:-120px;
    bottom:-120px;
    animation:float2 18s ease-in-out infinite;
}

@keyframes float1{
0%{transform:translateY(0);}
50%{transform:translateY(40px);}
100%{transform:translateY(0);}
}

@keyframes float2{
0%{transform:translateY(0);}
50%{transform:translateY(-40px);}
100%{transform:translateY(0);}
}

/* ===========================
   Main Container
=========================== */

.gradio-container{
    max-width:900px !important;
    margin:auto;
    padding-top:30px;
}

/* Glass Card */

.block{
    background:rgba(255,255,255,0.08)!important;
    backdrop-filter:blur(18px);
    border:1px solid rgba(255,255,255,0.15);
    border-radius:22px!important;
    box-shadow:
        0 15px 35px rgba(0,0,0,.35),
        inset 0 0 0 1px rgba(255,255,255,.05);
    transition:.35s;
}

.block:hover{
    transform:translateY(-3px);
    box-shadow:0 20px 40px rgba(0,0,0,.45);
}

/* ===========================
   Title
=========================== */

h1{
    text-align:center;
    color:white!important;
    font-size:40px!important;
    font-weight:800!important;
    letter-spacing:.5px;
}

h2,h3,p,label{
    color:#e5e7eb!important;
}

/* ===========================
   Inputs
=========================== */

input{
    background:rgba(255,255,255,.08)!important;
    border:1px solid rgba(255,255,255,.15)!important;
    color:white!important;
    border-radius:12px!important;
}

input:focus{
    border:1px solid #34d399!important;
    box-shadow:0 0 15px rgba(52,211,153,.4)!important;
}

/* ===========================
   Button
=========================== */

button{
    background:linear-gradient(90deg,#10b981,#14b8a6)!important;
    color:white!important;
    font-weight:700!important;
    border:none!important;
    border-radius:14px!important;
    transition:.3s;
}

button:hover{
    transform:scale(1.03);
    box-shadow:0 10px 25px rgba(16,185,129,.35);
}

/* ===========================
   Output Box
=========================== */

.output-text{
    background:rgba(255,255,255,.08)!important;
    border-radius:15px!important;
}
"""
)

# ------------------------------------------

if __name__ == "__main__":
    # Render network configuration
    interface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
