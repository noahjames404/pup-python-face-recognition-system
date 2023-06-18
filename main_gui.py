
import time 

from tkinter import *
from tkinter import ttk
from tkinter import PanedWindow, Label, font 
from typing import Literal
from PIL import ImageTk, Image

#start modify
app_name = "Face Recognition System"
app_intro = "Welcome Back!"
app_guidelines = "Please stand still in front of the camera"

frame_size = "500x400" 
color_bg = "#024BAA"
color_accent = "#004AAF"
color_fg = "#FCFDFC" 

font_family = "Verdana"
font_size = 12
font_style = "normal"
#end modify


#constants
STANDBY = "standby"
DETECTED = "detected" 
VERIFY = "verify"
 

class GUIBuilder: 
    def __init__(self):
        self.state = STANDBY

    def AttachLabel(self,panel:PanedWindow, label:StringVar, bgc=color_bg,fgc=color_fg) -> Label:
        l = Label(panel,text=label,bg=bgc,fg=fgc,font=self.Font())
        panel.add(l)
        return l

    def AttachPanel(self,container:PanedWindow, bgc="red") -> PanedWindow:
        panel = PanedWindow(bg=bgc)
        container.add(panel)
        return panel

    def Build(self) -> Tk:
        root=Tk()
        root.title(app_name)
        root.geometry(frame_size)    
        root.configure(bg=color_bg)

        pmain = PanedWindow(orient=VERTICAL,bg=color_bg) 
        pmain.pack(fill=BOTH,padx=10,pady=40)

        self.Header(pmain)
        self.Body(pmain)  

        self.root = root
        return Tk
    
    def Run(self):
        self.root.mainloop()
    
    def Font(self,fsize=font_size,fweight=font_style) -> font.Font:
        return font.Font(family=font_family, size=fsize, weight=fweight)

    def Header(self,container:PanedWindow) -> PanedWindow :
        panel = self.AttachPanel(container,bgc=color_bg)
        h1 = self.AttachLabel(panel, app_name)
        h1["font"] = self.Font(9)

        h2 = self.AttachLabel(panel, app_intro) 
        h2["font"] = self.Font(20,"bold")

        self.AttachLabel(panel, app_guidelines) 
        panel["orient"] = VERTICAL 

    def Body(self,container:PanedWindow) -> PanedWindow :
        panel = self.AttachPanel(container,bgc=color_bg) 
        panel["orient"] = HORIZONTAL
        panel.pack(fill=BOTH) 

        self.Footer(container)
        

        pcam = self.AttachPanel(panel,bgc="#fff") 
        pcam.pack(side=LEFT, fill=BOTH, expand=1,padx=(10,0),pady=(0,10)) 

        pdetails = self.AttachPanel(panel,bgc=color_bg)  
        pdetails["orient"] = VERTICAL
        pdetails.pack(side=TOP,padx=10,pady=(0,10))

        canvas = self.LoadImage(pdetails,"./assets/img/logo.png") 
        canvas.pack(pady=(0,10))


        pinfo = self.AttachPanel(pdetails,"#fff")
        pinfo["orient"] = VERTICAL
        pinfo.pack(padx=(10,10),fill=X) 
 
        self.info_status = self.FormatLabelInfo(pinfo,"Stand By")
        self.info_status.configure(font=self.Font(15,"bold"))  

        self.info_student = self.FormatLabelInfo(pinfo,"----")
         
        self.FormatLabelInfo(pinfo,"GRADE & SECTION")
        
        self.info_gs = self.FormatLabelInfo(pinfo,"----")
        self.FormatLabelInfo(pinfo,"GUARDIAN")
        self.info_guardian = self.FormatLabelInfo(pinfo,"----")
        self.info_contact = self.FormatLabelInfo(pinfo,"----") 

    def Footer(self,container:PanedWindow):
        pfooter = self.AttachPanel(container,bgc=color_bg)
        pfooter.pack(side=BOTTOM,fill=BOTH,padx=10,pady=(0,10))

        pfooter = self.AttachLabel(pfooter,"00:00:00",bgc=color_bg)
        pfooter.pack(fill=BOTH,side=LEFT)

 
    def FormatLabelInfo(self,panel:PanedWindow,text) -> Label:
        label = self.AttachLabel(panel,text)
        label["bg"] = color_fg
        label["fg"] = "#333" 
        label.pack(fill=X)
        return label
    
    def LoadImage(self,container:PanedWindow,asset_path:str) -> Canvas:
        canvas = Canvas(container,bg=color_bg,highlightthickness=0,width=250,height=300)
        
        logo_size = (250,350)

        image = Image.open(asset_path) 
        image = image.resize(logo_size)
        photo = ImageTk.PhotoImage(image)
        canvas.create_image(logo_size[0]/2,logo_size[1]/2-35,image=photo)
        canvas.image = photo
        return canvas

    
    def UpdateState(self,state:Literal["standby","verify","detected"]):
        self.state = state
        if(state == STANDBY):
            self.info_status["text"] = "Stand By"
            self.info_status["fg"] = "#f1c40f"
            self.info_student["text"] = "----"
            self.info_gs["text"] = "----"
            self.info_guardian["text"] = "----"
            self.info_contact["text"] = "----"
        if(state == DETECTED):
            self.info_status["text"] = "Welcome Back!"
            self.info_status["fg"] = "#2ecc71"
        if(state == VERIFY):
            self.info_status["text"] = "Verifying..."
            self.info_status["fg"] = "#3498db"


    
 

gui = GUIBuilder()

root = gui.Build()   
gui.UpdateState(VERIFY)

gui.Run()







