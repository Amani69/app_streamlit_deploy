import streamlit_authenticator as stauth
import database as db

names = ["Amani","amani"]
usernames = ["Amani","amina"]
passwords = ["123","abc"]

hashed_passwords = stauth.Hasher(passwords).generate()


for(username,name,hashed_password) in zip(usernames,names,hashed_passwords):
    db.insert_user(username,name,hashed_password)