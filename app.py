# import
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import pyautogui
import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import database as db
import streamlit_authenticator as stauth
from streamlit_option_menu import  option_menu 
from streamlit_lottie import st_lottie
import json
import requests  # pip install requests
import sklearn



#lottie
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)



# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Community Attrition Management", page_icon="::chart_with_upwards_trend:", layout="wide")


# user authentication

users=db.fetch_all_users() 
usernames=[user["key"] for user in users]
names=[user["name"] for user in users]
hashed_passwords=[user["password"] for user in users]
  
authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
    "Eclaireurs_prediction", "abcdef", cookie_expiry_days=30)

lottie_log = load_lottiefile("login.json")  # replace link to local lottie file
lottie_go = load_lottiefile("go.json") 




name, authentication_status, username = authenticator.login("Login", 'sidebar')

if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st_lottie(
    lottie_log,
    speed=1,
    reverse=True,
    loop=True,
    quality="high", # medium ; high
    #renderer="canvas", # canvas,svg
    height=300,
    width=300,
    key=None,
    )
    st.sidebar.warning("Please enter your username and password",icon="⚠️")

if authentication_status:
     
    
#ajouter le titre 

    # Add a sidebar to the web page. 
   # st.markdown('---')
    
    authenticator.logout("Logout", "sidebar")
    st.sidebar.title(f"Welcome {name}")
    # Sidebar Configuration
    #st.sidebar.image('https://is4-ssl.mzstatic.com/image/thumb/Purple125/v4/10/ba/d3/10bad386-6c5f-cf95-c227-6690d321a298/AppIcon-0-0-1x_U007emarketing-0-0-0-5-0-0-sRGB-0-0-0-GLES2_U002c0-512MB-85-220-0-0.png/256x256bb.jpg', width=200)
    st.sidebar.image('Capture14.png', width=100)
   
    st.sidebar.markdown('# Eclaireurs')
    st.sidebar.markdown('Est une application  développée pour fournir des informations liées aux  déplacements des Français.')

    with st.sidebar.expander("More Information",expanded=False):
        st.write("""
                Les utilisateurs vont être impliqués dès l’installation d’une application mobile qui va
                leur offrir une expérience numérique et des mesures sur leurs déplacements quotidiens . 
                L’application collecte en temps réel les données de tous les utilisateurs pour les stocker dans une plateforme data.
                Ces données seront traitées et  valorisées surtout de par leur exploitation.                         """)


    # Graph (Bar Chart in Sidebar)
    df2= pd.read_csv('data_date.csv')
    # Plot 
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
    # hide_st_style = """
    #             <style>
    #             #MainMenu {visibility: hidden;}
    #             footer {visibility: hidden;}
    #             header {visibility: hidden;}
    #             </style>
    #             """
    # st.markdown(hide_st_style, unsafe_allow_html=True)





    def main():
        st.title(":chart_with_upwards_trend:   Community Attrition Management")
        
        "---"
        #ajouter data 
        df = pd.read_csv('token_data.csv')
        df.date_premiere_cap=pd.to_datetime(df.date_premiere_cap,format='%Y/%m/%d').dt.date
        
        # --- NAVIGATION MENU ---
        selected = option_menu(
        menu_title=None,
        options=["Data", "Modeling"],
        icons=["bar-chart-fill", "bi-graph-up-arrow"],  # https://icons.getbootstrap.com/
        orientation="horizontal",
        )
       
       
        if selected == "Data":
            info1,info2=st.columns(2)
             # Graph (Pie Chart )
             
            df2= pd.read_csv('target.csv')
            df_target = df2[['token', 'profil_utilisateur']].groupby('profil_utilisateur').count() / len(df2)
            colours = ['#9ed5d6','#24abb0']
            #contour='line':dict(color='#D33682', width=)
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
                
                # Display statistical information on the dataset.
                #st.subheader('Statistical Info about the Data')
                stat = st.checkbox('Click to show Statistical Info ')
                if stat:
                    st.write(df.describe())
            
            st.info('Active users: They are people who have less than 5 months of inactivity', icon="ℹ️")

            "---"
        #  La distribution des utilisateurs inactifs
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

                #st.plotly_chart(fig, use_container_width=True)
                # Plot 
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
        
        
        #Partie de prediction 
        # Selection for a specific time frame.
            st.subheader('Model')
            df_select = df 

            col1, col2 = st.columns(2)
            "---"
            with col1:
                #st.write('Select a Start Date to predict ')
                start_date = st.date_input('Start Date',min_value= datetime.date(2021,9,27),max_value=datetime.date(2022,11,29),value=datetime.date(2021,9,27))

            with col2:    
                #st.write('Select an End Date to predict ')
                end_date = st.date_input('End Date',min_value=datetime.date(2021,9,27),max_value=datetime.date(2023,1,31),value=datetime.date(2023,1,31))
            
            #Model
        
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
                    
                        # Prédisez les données de test en utilisant le modèle entraîné
                        predictions = model.fit().predict(start=len(y_train), end=len(y_train)+len(y_test)-1, exog=X_test)
                        # #inverse_transform train
                        forecast_test = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(-1, )
                        test["Forecast_ARIMAX"] = forecast_test
                        #st.line_chart(test[['profil_utilisateur',"Forecast_ARIMAX"]],width=0, height=0, use_container_width=True)
                        
                    #   # create a Plotly line plot
                        line_fig = px.line(
                        test.reset_index(),
                        x="date_premiere_cap",
                        y=['profil_utilisateur',"Forecast_ARIMAX"],
                        title="Actual users vs Forecasted users",
                        labels={
                            "date_premiere_cap": " Date",
                            "profil_utilisateur": "Users",
                            #"variable": "Legend"
                        }
                        )

                        # Plotly graph configs
                        #legend_names = {"sales": "Actual Sales", "sales_forecast": "Forecasted Sales"}
                
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

                        # passing in the Plotly graph object to Streamlit
                        st.plotly_chart(line_fig, use_container_width=True) 
                
                    else:
                        st.warning("Invalid Date  - Re-enter Dates")
                "---"
                if st.button('Predict'):
                    r2=format(round(r2_score(test["profil_utilisateur"],test["Forecast_ARIMAX"])*100,2),",")
                    #st.metric(value=r2,label="% of precesion")
                    actual_volume = format(test["profil_utilisateur"].sum(), ",")
                    predicted_volume = format(round(test["Forecast_ARIMAX"].sum()), ",")
                    val_abs=round((abs(int(actual_volume)-int(predicted_volume))/int(actual_volume))*100,2)
                    metric_col1, metric_col2, metric_col3,metric_col4= st.columns(4)
                    
                    personnes="Users"
                    #metric_col1.markdown('##### Actual Volume:\n')
                    metric_col1.metric('Actual Volume',f"{actual_volume} {personnes}")

        
                    #metric_col2.markdown('##### Forecasted Volume:\n')
                    metric_col2.metric('Forecasted Volume',f"{predicted_volume}"+" Users")
                    
                    #metric_col3.markdown('##### R2:\n')
                    metric_col3.metric('R2',f"{r2}"+"%")
                    
                    #metric_col4.markdown('##### Error:\n')
                    metric_col4.metric('Error',f"{val_abs}"+"%")
                

                    # line break to separate metrics from plot
                    """---"""
                    
                    #st.write(r2,'% of Precision ')
                    #st.warning('This is a warning', icon="⚠️")
                    
                
                
    




        


            
        
    if __name__ == '__main__':
        main() 