import streamlit as st
import pickle
import numpy as np
import sklearn
from streamlit_option_menu import option_menu
import pymongo

# MongoDB connection
client = pymongo.MongoClient("mongodb+srv://thalakumara619:kumara@datascienceproject.9eegkw6.mongodb.net/?retryWrites=true&w=majority&appName=DataScienceProjec")
db = client["Predictive_Maintenance"]
coll = db["datas"]

def predict_failure(Air_temp, Process_temp, Rot_speed, Torque, Tool_wear, Type):
    # Encode categorical variable
    if Type == "H":
        H, L, M = (1, 0, 0)
    elif Type == "L":
        H, L, M = (0, 1, 0)
    else:  # M
        H, L, M = (0, 0, 1)

    # Load model
    with open(r"RandomForest_Model.pkl", "rb") as f:
        model_reg = pickle.load(f)

    user_data = np.array([[Air_temp, Process_temp, Rot_speed, Torque,
                           Tool_wear, H, L, M]])
    
    y_pred = model_reg.predict(user_data)

    # Extract prediction
    Target = int(y_pred[0][0])
    No_Failure = int(y_pred[0][1])
    Heat_Dissipation_Failure = int(y_pred[0][2])
    Overstrain_Failure = int(y_pred[0][3])
    Power_Failure = int(y_pred[0][4])
    Tool_Wear_Failure = int(y_pred[0][5])
    Random_Failures = int(y_pred[0][6])

    # Save to DB
    user_data_dict = {
        'Air temperature [K]': Air_temp,
        'Process temperature [K]': Process_temp,
        'Rotational speed [rpm]': Rot_speed,
        'Torque [Nm]': Torque,
        'Tool wear [min]': Tool_wear,
        'H': H, 'L': L, 'M': M,
        'Target': Target,
        'No Failure': No_Failure,
        'Heat Dissipation Failure': Heat_Dissipation_Failure,
        'Overstrain Failure': Overstrain_Failure,
        'Power Failure': Power_Failure,
        'Tool Wear Failure': Tool_Wear_Failure,
        'Random Failures': Random_Failures
    }
    coll.insert_one(user_data_dict)

    st.success("✅ User Data Saved Successfully!")

    # Return results
    return y_pred


# Streamlit page settings
st.set_page_config(layout="wide")
st.title(":blue[**Predictive Maintenance for Manufacturing Equipment**]")

# Sidebar Menu
with st.sidebar:
    option = option_menu('MENU', options=["Home", "Predict Failure"], 
                         icons=["house", "activity"], menu_icon="cast", default_index=0)

# Home Page
if option == "Home":
    st.header("🏭 Welcome to the Predictive Maintenance App")
    st.markdown("""
    This application helps in predicting **machine failures** in a manufacturing setup.  
    It uses a trained **Random Forest Multi-label Classification Model** to predict different failure types:
    - 🔥 Heat Dissipation Failure  
    - 💪 Overstrain Failure  
    - ⚡ Power Failure  
    - 🛠️ Tool Wear Failure  
    - 🎲 Random Failure  
    
    The goal is to minimize downtime and improve productivity by detecting failures early.
    """)
    
    # Add image (place a file named machine.jpg in your project folder)
    st.image("machine.jpg", caption="Smart Manufacturing with Predictive Maintenance", use_container_width=True)

# Prediction Page
elif option == "Predict Failure":
    st.header("🔎 Predict Machine Failure")
    st.write(" ")

    col1, col2 = st.columns(2)

    with col1:
        Air_temperature = st.number_input("**Enter the Value for Air temperature [K]**")
        Process_temperature = st.number_input("**Enter the Value for Process temperature [K]**")
        Rotational_speed = st.number_input("**Enter the Value for Rotational speed [rpm]**")

    with col2:
        Torque = st.number_input("**Enter the Value for Torque [Nm]**")
        Tool_wear = st.number_input("**Enter the Value for Tool wear [min]**")
        Type = st.selectbox("Enter the Type:", ["H", "L", "M"])

    button = st.button(":violet[***PREDICT THE FAILURE***]", use_container_width=True)

    if button:
        y_predicted = predict_failure(Air_temperature, Process_temperature,
                                      Rotational_speed, Torque, Tool_wear, Type)

        if y_predicted[0][0] == 1:  # Failure detected
            st.error("⚠️ **Failure Detected!**")
            st.subheader("Predicted Failure Types:")

            if y_predicted[0][2] == 1:
                st.write("🔥 Heat Dissipation Failure")

            if y_predicted[0][3] == 1:
                st.write("💪 Overstrain Failure")

            if y_predicted[0][4] == 1:
                st.write("⚡ Power Failure")

            if y_predicted[0][5] == 1:
                st.write("🛠️ Tool Wear Failure")

            if y_predicted[0][6] == 1:
                st.write("🎲 Random Failure")
                
        else:
            st.success("✅ No Failure Occurred")
   