
from tkinter import *
from tkinter import ttk
from tkinter import PanedWindow, Label, font

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

class GUIBuilder:
    def AttachLabel(self,panel:PanedWindow, label:StringVar, bgc=color_bg,fgc=color_fg) -> Label:
        l = Label(panel,text=label,bg=bgc,fg=fgc,font=self.Font())
        panel.add(l)
        return l

    def AttachPanel(self,container:PanedWindow, bgc="red") -> PanedWindow:
        panel = PanedWindow(bg=bgc)
        container.add(panel)
        return panel

    def Build(self):
        root=Tk()
        root.title(app_name)
        root.geometry(frame_size)   

        pmain = PanedWindow(orient=VERTICAL,bg=color_bg)
        pmain.pack(fill=BOTH)

        self.Header(pmain)
        self.Body(pmain) 
        
        root.mainloop();
    
    def Font(self,fsize=font_size,fweight=font_style) -> font.Font:
        return font.Font(family=font_family, size=fsize, weight=fweight)

    def Header(self,container:PanedWindow) -> PanedWindow :
        panel = self.AttachPanel(container,bgc=color_bg)
        self.AttachLabel(panel, app_name) 
        self.AttachLabel(panel, app_intro) 
        self.AttachLabel(panel, app_guidelines) 
        panel["orient"] = VERTICAL 

    def Body(self,container:PanedWindow) -> PanedWindow :
        panel = self.AttachPanel(container,bgc=color_bg) 
        panel["orient"] = HORIZONTAL
        panel.pack(fill=BOTH) 

        pcam = self.AttachPanel(panel,bgc=color_bg) 
        pcam.pack(side=LEFT, fill=BOTH, expand=1)
        pcam_label = self.AttachLabel(pcam,"PCAM") 

        pdetails = self.AttachPanel(panel,bgc=color_bg) 
        pdetails["orient"] = VERTICAL
        pdetails.pack(side=LEFT, expand=1)
        pdetails_label = self.AttachLabel(pdetails,"Stand By") 
        pdetails_label = self.AttachLabel(pdetails,"Henry Sy.") 
        pdetails_label = self.AttachLabel(pdetails,"0935-380-2346") 
        pdetails_label = self.AttachLabel(pdetails,"grade & section") 
        pdetails_label = self.AttachLabel(pdetails,"4-Archimedes") 
        pdetails_label = self.AttachLabel(pdetails,"guardian") 
        pdetails_label = self.AttachLabel(pdetails,"rose marie") 
 

gui = GUIBuilder()
gui.Build();







