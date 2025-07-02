# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 21:36:57 2025

@author: lalit
"""

import numpy as np
import pickle
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Thyroid Status Predictor",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load the saved model
Loaded_model = pickle.load(open('best_model.sav', 'rb'))

# Function for prediction
def disease_detection(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    input_data_reshape = input_data_as_numpy_array.reshape(1, -1)
    prediction = Loaded_model.predict(input_data_reshape)

    if prediction[0] == 0:
        return "✅ The individual is likely normal (no thyroid disease detected)."
    else:
        return "⚠️ The individual may have a thyroid disorder. Please consult a healthcare professional."

# Home Page
def home():
    st.title("Welcome to Thyroid prediction App")
    st.markdown("## 🔍 What does this app do?")
    st.write("""
        This application helps determine whether a person has a thyroid disorder based on input features.  
        It uses a pre-trained machine learning model to predict the result.
    """)
    st.info("👉 Use the sidebar to navigate to the prediction page.")

# Prediction Page
def thyroid_prediction():
    st.title("🩺 Thyroid Status Checker")
    st.markdown("### Please enter your details:")

    # Inputs
    age = st.text_input("👤 Age:")
    TSH = st.text_input("🧪 TSH Value:")
    T3 = st.text_input("🧪 T3 Value:")
    T4 = st.text_input("🧪 T4 Value:")

    diagnose = None

    if st.button("🔍 Predict Thyroid Status"):
        try:
            with st.spinner("Analyzing your thyroid status... 🧠"):
                diagnose = disease_detection([age, TSH, T3, T4])

            if "normal" in diagnose.lower():
                st.markdown("""
                    <div style='background-color:#666ef1;color:black; padding: 15px; border-radius: 10px; border-left: 5px solid #2196f3;'>
                   <h4>🟢 Result: Normal</h4>
                  <p>✅ You are not showing signs of thyroid disease based on the input values.</p>
                  <p>🧘 Stay healthy! Keep monitoring your levels regularly.</p>
                  
                 </div>

                """, unsafe_allow_html=True)

                with st.expander("💡 Tips to Maintain a Healthy Thyroid"):
                    st.write("""
                    - Eat iodine-rich foods (like fish, dairy, eggs).  
                    - Manage stress — yoga & sleep help.  
                    - Stay active and hydrated.  
                    - Get routine thyroid tests if symptoms arise.
                    """)
            else:
                st.markdown("""
                    <div style='background-color:#f14b4b; color:black; padding: 15px; border-radius: 10px; border-left: 5px solid red;'>
                        <h4>🔴 Result: Possible Thyroid Disorder</h4>
                        <p>⚠️ Your values indicate a potential thyroid issue.</p>
                        <p>👩‍⚕️ Please consult an endocrinologist or physician for proper diagnosis.</p>
                        <h4>🧠 Interpretation:</h4>
    <p>Based on the entered values, there is a high likelihood of thyroid dysfunction. This might indicate <strong>hypothyroidism</strong> (underactive thyroid) or <strong>hyperthyroidism</strong> (overactive thyroid), depending on which hormone is imbalanced.</p>
    <h4>🩺 Recommended Actions:</h4>
    <ul>
        <li>📅 Book an appointment with a certified <strong>endocrinologist</strong>.</li>
        <li>🧪 Request blood tests including: TSH, Free T3, Free T4, Anti-TPO antibodies.</li>
        <li>🧭 Ask for an ultrasound if nodules or swelling are observed in the thyroid region.</li>
    </ul>
    <h4>🥗 Dietary Tips:</h4>
    <ul>
        <li>✅ Include iodine-rich foods: seafood, eggs, dairy.</li>
        <li>🧂 Avoid excessive soy, cruciferous vegetables (like cabbage) if hypothyroid.</li>
        <li>🚰 Stay well hydrated, maintain balanced meals with selenium and zinc.</li>
    </ul>
                        <h4>💊 General Treatment Options:</h4>
    <ul>
        <li>🔹 For Hypothyroidism: <em>Levothyroxine (T4 hormone replacement)</em></li>
        <li>🔹 For Hyperthyroidism: <em>Antithyroid medications, radioactive iodine therapy, or surgery (rare cases)</em></li>
    </ul>
    </div>
                
                """, unsafe_allow_html=True)

                st.warning("⚠️ Note: This is a machine learning prediction, not a medical diagnosis.")

                with st.expander("🧠 Learn About Thyroid Disorders"):
                    st.write("""
                    - **Hyperthyroidism:** Overactive thyroid (fast heartbeat, weight loss).  
                    - **Hypothyroidism:** Underactive thyroid (fatigue, weight gain).  
                    - Both require blood tests and medical consultation for confirmation.
                    """)
        except ValueError:
            st.error("❌ Please enter valid numerical values for all fields.")

# Footer
def footer():
    st.markdown("""
        <style>
            .footer {
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                background-color: #f0f2f6;
                color: #4b4b4b;
                text-align: center;
                padding: 10px;
                font-size: 14px;
                border-top: 1px solid #e0e0e0;
            }
        </style>
        <div class="footer">
            🚀 Developed by Lalit | 💡 Powered by Streamlit | 📅 2025
        </div>
    """, unsafe_allow_html=True)

# Main Function
def main():
    # Professional Sidebar
    with st.sidebar:
        st.title("🧭 Navigation")
        
        page = st.radio("Select a Page", ["Home", "Thyroid Prediction"])


        st.markdown("### 🌐 Useful Links")
        st.markdown("[📂 GitHub Repo](https://github.com/yourusername)")
        st.markdown("[📧 Email](mailto:lalitsinghs420@gmail.com)")
        st.markdown("[📄 Docs](https://your-docs-link.com)")


    # Route to selected page
    if page == "Home":
        home()
    elif page == "Thyroid Prediction":
        thyroid_prediction()

    footer()  # Show footer always

# Run the App
if __name__ == "__main__":
    main()
