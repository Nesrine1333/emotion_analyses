# Core Pkgs
import streamlit as st 
import altair as alt
import plotly.express as px 
import altair as alt
# EDA Pkgs
import pandas as pd 
import numpy as np 
from datetime import datetime

import joblib 

#from track_utils import create_page_visited_table,add_page_visited_details,view_all_page_visited_details,add_prediction_details,view_all_prediction_details,create_emotionclf_table
emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}

pipe_lr = joblib.load(open("model/emotion_classifier_pipe.pkl", "rb"))  




#function to predict sentiment
def predict_emotion(text):
    results = pipe_lr.predict([text])
    return results[0]

def get_prediction_proba(text):
    results = pipe_lr.predict_proba([text])
    return results

def main():
    st.title("Emotion Detection")
    menue =["Home","Monitor","About"]
    choice=st.sidebar.selectbox("Menu",menue)
    if choice=="Home": 
        st.subheader("Home-Emotion Detection")
        with st.form(key='emotion_clf_form'):
            raw_text=st.text_area("Type Here")
            submit_text=st.form_submit_button(label='Submit')
        if submit_text:
            col1,col2  = st.columns(2)
			# Apply Fxn Here
            prediction = predict_emotion(raw_text)
            probability = get_prediction_proba(raw_text)
         #   add_prediction_details(raw_text,prediction,np.max(probability),datetime.now())
            
            with col1:
                st.success("Original Text")
                st.write(raw_text)
                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction,emoji_icon))
                st.write(f"Confidence: {probability.max():.2f}")
				
            
            with col2:
                st.success("Prediction Probability")
                #st.write(probability)
                proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
              #  st.write(proba_df.T)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions","probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
                st.altair_chart(fig,use_container_width=True)
                
    elif choice=="Monitor ":
        st.subheader("Monitor App")

if __name__ == '__main__':
    main()