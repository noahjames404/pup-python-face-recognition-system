import requests 
import os
from pprint import pprint
from supabase_py import create_client, Client
from queue import Queue
from typing import NamedTuple
from cachetools import cached, TTLCache
from datetime import timedelta
import traceback

class StudentResponse(NamedTuple):
    registration_no:str
    name:str
    grade:int
    section:str
    guardian:str
    phone:str

class API():
    def __init__(self):
        self.database_url:str = os.environ.get("SUPABASE_URL")
        self.key:str = os.environ.get("SUPABASE_KEY")
        self.supabase:Client = create_client(self.database_url,self.key) 

    #todo replace App.NotifyContact with API.NotifyContact
    def NotifyContact(self,send_to:str,name:str,body:str):
        req = requests.post(
            "https://facedetection-1-s8245812.deta.app/sms/send",
            json={
                "send_to": send_to,
                "body": body,
                "name": name,
            }
        ) 
    
    @cached(cache=TTLCache(maxsize=100,ttl=300))
    def FetchStudentById(self,id:str):
        if id is None:
            print("Error Handled: Invalid ID type of `None`")
            return None
        
        try:
            res = self.supabase.table("user").select("*").eq("registration_no",id).execute() 
            if len(res["data"]) != 0:
                 return StudentResponse(
                     res["data"][0]["registration_no"],
                     res["data"][0]["name"],
                     res["data"][0]["grade"],
                     res["data"][0]["section"],
                     res["data"][0]["guardian"],
                     res["data"][0]["phone"]
                 )
            else:
                return None
        except:
            print("Error Handled"+traceback.format_exc())

        

# setup
# print(os.environ.get("SUPABASE_URL"))
# print(os.environ.get("SUPABASE_KEY"))

# api = API()
# pprint(api.FetchStudentById("1011"))