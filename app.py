import streamlit as st
import os
#os.environ['DISPLAY'] = ':1'
import webbrowser as web
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
import seaborn as sns
#import pyautogui
import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
import plotly.graph_objects as go
import altair as alt
import database as db
import streamlit_authenticator as stauth
from streamlit_option_menu import  option_menu 
from streamlit_lottie import st_lottie
import json
import requests  
import sklearn


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

st.set_page_config(page_title="Community Attrition Management", page_icon="::chart_with_upwards_trend:", layout="wide")


# users=db.fetch_all_users() 
# usernames=[user["key"] for user in users]
# names=[user["name"] for user in users]
# hashed_passwords=[user["password"] for user in users]
  
# authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
#     "Eclaireurs_prediction", "abcdef", cookie_expiry_days=30)

# lottie_log = load_lottiefile("login.json") 
# lottie_go = load_lottiefile("go.json") 




# name, authentication_status, username = authenticator.login("Login", 'sidebar')

# if authentication_status == False:
#     st.error("Username/password is incorrect")

# if authentication_status == None:
#     st_lottie(
#     lottie_log,
#     speed=1,
#     reverse=True,
#     loop=True,
#     quality="high", 
#     height=300,
#     width=300,
#     key=None,
#     )
#     st.sidebar.warning("Please enter your username and password",icon="⚠️")

# if authentication_status:
     

authenticator.logout("Logout", "sidebar")
st.sidebar.title(f"Welcome {name}")
st.sidebar.image('Capture14.png', width=100)

st.sidebar.markdown('# Eclaireurs')
st.sidebar.markdown('Est une application  développée pour fournir des informations liées aux  déplacements des Français.')

with st.sidebar.expander("More Information",expanded=False):
    st.write("""
            Les utilisateurs vont être impliqués dès l’installation d’une application mobile qui va
            leur offrir une expérience numérique et des mesures sur leurs déplacements quotidiens . 
            L’application collecte en temps réel les données de tous les utilisateurs pour les stocker dans une plateforme data.
            Ces données seront traitées et  valorisées surtout de par leur exploitation.                         """)



df2= pd.read_csv('data_date.csv')
category_tr = df2.groupby(by='year')['token'].count().to_frame().reset_index()

fig1 = px.bar(category_tr, x='year', y='token',
            
            )
fig1.update_layout(showlegend=False,
                title="Number of users per Year",
                title_x=0.5,
                xaxis_title='Year',
                yaxis_title='Number of users',  
                )
st.sidebar.plotly_chart(fig1, use_container_width=True)

#pour enlever made in streamlit, parmas , ligne...
# --- HIDE STREAMLIT STYLE ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """






def main():
    st.title(":chart_with_upwards_trend:   Community Attrition Management")
    
    "---"

    df = pd.read_csv('token_data.csv')
    df.date_premiere_cap=pd.to_datetime(df.date_premiere_cap,format='%Y/%m/%d').dt.date
    
    selected = option_menu(
    menu_title=None,
    options=["Data", "Modeling"],
    icons=["bar-chart-fill", "bi-graph-up-arrow"],  
    orientation="horizontal",
    )
    
    
    if selected == "Data":
        info1,info2=st.columns(2)
    
            
        df2= pd.read_csv('target.csv')
        df_target = df2[['token', 'profil_utilisateur']].groupby('profil_utilisateur').count() / len(df2)
        colours = ['#9ed5d6','#24abb0']
        fig_target = go.Figure(data=[go.Pie(labels=["Active","Inactive"],
                                            values=df_target['token']
                                        ,marker= {'colors': colours,'line':dict(color='#D33682', width=0.7)},hole=.5)])
        fig_target.update_layout(showlegend=False,
                                height=200,
                                margin={'l': 20, 'r': 60, 't': 0, 'b': 0})
        fig_target.update_traces(textposition='inside', textinfo='label+percent')
        with info1:
            st.markdown("##### Our users")
            st.plotly_chart(fig_target, use_container_width=True)
        
        with info2:
            st.markdown('##### Info about the Data')
            show_data = st.checkbox('Click to show  Data')
            if show_data   :
                st.dataframe(df.head())
            
            stat = st.checkbox('Click to show Statistical Info ')
            if stat:
                st.write(df.describe())
        
        st.info('Active users: They are people who have less than 5 months of inactivity', icon="ℹ️")

        "---"
        st.subheader("Distribution of inactive users")
        
        viz_year, comparatif_viz = st.tabs(['Choose a year', 'Compare the years'])
        with viz_year:
            df["date_premiere_cap"]=pd.to_datetime(df.date_premiere_cap,format='%Y/%m/%d')
            df["year"]=df["date_premiere_cap"].dt.year
            df["month"]=df["date_premiere_cap"].dt.month
            contact_options = [2019,2020,2021,2022]
            contact_selected = st.selectbox("Select a Year", contact_options,)
            df_filter=df[df["year"]==contact_selected]
            inform = f"The chart for {contact_selected} :"
            fig = px.line(df[df["year"]==contact_selected], x="date_premiere_cap", y="profil_utilisateur", title=inform)

            fig2 = px.line(df_filter, x='date_premiere_cap', y='profil_utilisateur',
                    
                        )
            fig2.update_layout(showlegend=False,
                            title=inform,
                            title_x=0.5,
                            xaxis_title='Date',
                            yaxis_title='Number of users',  
                        )
            st.plotly_chart(fig2, use_container_width=True)
        
        with comparatif_viz:
            category_tr2 = df.groupby(by=['year','month'])['profil_utilisateur'].sum().to_frame().reset_index()
            fig2 = px.line(category_tr2, x='month', y='profil_utilisateur',color="year"
                    
                        )
            fig2.update_layout(showlegend=True,
                            title="Number of users per Year",
                            title_x=0.5,
                            xaxis_title='Year',
                            yaxis_title='Number of users',  
                        )
            st.plotly_chart(fig2, use_container_width=True)
        
        "---"
        
        
    if selected=="Modeling":
    
        st.subheader('Model')
        df_select = df 

        col1, col2 = st.columns(2)
        "---"
        with col1:
            start_date = st.date_input('Start Date',min_value= datetime.date(2021,9,27),max_value=datetime.date(2022,11,29),value=datetime.date(2021,9,27))

        with col2:    
            end_date = st.date_input('End Date',min_value=datetime.date(2021,9,27),max_value=datetime.date(2023,1,31),value=datetime.date(2023,1,31))
        
    
        if st.checkbox('Period Selected to predict'):
            if(start_date != None or end_date != None):
                if(start_date < end_date):

                    train=df[df.date_premiere_cap < start_date]
                    test=df[df.date_premiere_cap.between(start_date,end_date)]
                    
                    
                    train=train.set_index("date_premiere_cap")
                    test=test.set_index("date_premiere_cap")
                    scaler =  StandardScaler()
                    X_train = scaler.fit_transform(train.drop('profil_utilisateur',axis=1))
                    X_test = scaler.transform(test.drop('profil_utilisateur',axis=1))
                    y_train = scaler.fit_transform(train["profil_utilisateur"].values.reshape(-1, 1)).reshape(-1, )
                    y_test = scaler.transform(test["profil_utilisateur"].values.reshape(-1, 1)).reshape(-1, )
                
                    model = SARIMAX(y_train,X_train, order=(3, 0, 0))
                    model.fit()
                
                    predictions = model.fit().predict(start=len(y_train), end=len(y_train)+len(y_test)-1, exog=X_test)
                    forecast_test = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(-1, )
                    test["Forecast_ARIMAX"] = forecast_test
                    
                    line_fig = px.line(
                    test.reset_index(),
                    x="date_premiere_cap",
                    y=['profil_utilisateur',"Forecast_ARIMAX"],
                    title="Actual users vs Forecasted users",
                    labels={
                        "date_premiere_cap": " Date",
                        "profil_utilisateur": "Users",
                        
                    }
                    )

            
                    line_fig.update_layout(
                    xaxis=dict(showgrid=True),
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="right",
                        x=0.99
                    ),
                    title_x=0.5,
                    height=600
                    )

                    st.plotly_chart(line_fig, use_container_width=True) 
            
                else:
                    st.warning("Invalid Date  - Re-enter Dates")
            "---"
            if st.button('Predict'):
                r2=format(round(r2_score(test["profil_utilisateur"],test["Forecast_ARIMAX"])*100,2),",")
                actual_volume = format(test["profil_utilisateur"].sum(), ",")
                predicted_volume = format(round(test["Forecast_ARIMAX"].sum()), ",")
                val_abs=round((abs(int(actual_volume)-int(predicted_volume))/int(actual_volume))*100,2)
                metric_col1, metric_col2, metric_col3,metric_col4= st.columns(4)
                
                personnes="Users"
                metric_col1.metric('Actual Volume',f"{actual_volume} {personnes}")

    
                metric_col2.metric('Forecasted Volume',f"{predicted_volume}"+" Users")
                
                metric_col3.metric('R2',f"{r2}"+"%")
                
                metric_col4.metric('Error',f"{val_abs}"+"%")
            

                """---"""
                
                
                    
                
                
    




        


            
        
    if __name__ == '__main__':
        main() 