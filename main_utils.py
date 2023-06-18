import datetime
import re 

def GetDateTime():
    # Get the current date and time
    current_datetime = datetime.datetime.now()

    # Remove the microsecond component
    current_datetime = current_datetime.replace(microsecond=0)

    return current_datetime

def ExtractStudentID(label:str)->str: 
    print(label)
    match = re.search(r"\d+", label)
    if match:
        return match.group(0) 
    else:
        return None
    
#print()