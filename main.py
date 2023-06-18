from main_ai import App 
from main_gui import GUIBuilder, STANDBY, VERIFY, DETECTED
from main_utils import GetDateTime, ExtractStudentID
from main_api import API, StudentResponse
from pprint import pprint

print("App: Initializing")

gui = GUIBuilder()
api = API()
app = App()
 
root = gui.Build() 
app_thread = app.Execute()

root.protocol("WM_DELETE_WINDOW", lambda:app.queue_frame.put(None))

state = STANDBY

try:
    while True:
        info = app.queue_frame.get()

        if info is None: 
            root.destroy()
            app.queue_stop = True
            print("App: Closing Main Thread")
            break 

        gui.UpdateFrame(info[1])
        gui.UpdateTime(GetDateTime())

        if info[0] is None:
            gui.UpdateState(STANDBY)
            state = STANDBY
        elif state == DETECTED:
            gui.UpdateState(DETECTED)
        elif info[0] is not None:
            gui.UpdateState(VERIFY)
            res = api.FetchStudentById(ExtractStudentID(info[0]))
            if(res is not None):
                state = DETECTED
                gui.UpdateInfo(
                    res.name,
                    f"{res.grade} {res.section}",
                    res.guardian,
                    res.phone 
                )  
    
        root.update_idletasks()
        root.update()
except Exception as e:
    app.queue_stop = True
    pprint(e)

app_thread.join()

print("App: Resource Clean Up")
app.CleanUp()

#todo fix image not showing
#todo integrate with db