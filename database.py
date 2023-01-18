import os
from deta import Deta
from dotenv import load_dotenv


load_dotenv(".env")
DETA_KEY=os.getenv("DETA_KEY")




DETA_KEY="a0rnlw2m_JLMUiJBqspjrnhGV3R5tmZvFsxJCPLpk"
#initialiser
deta=Deta(DETA_KEY)
#CREATE/CONNECT a database
db=deta.Base("user_db")
def insert_user(username,name,password):
    """successful user creation or error"""
    return db.put({"key":username,"name":name,"password":password})
#insert_user("Amani","ELLAFI Amani","123")

def fetch_all_users():
    
    """return all users """
    res=db.fetch()
    return res.items
def update_user(username,updates):
    return db.update(updates,username)

def delete_user(username):
    return db.delete(username)