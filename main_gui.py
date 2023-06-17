
from tkinter import *
from tkinter import ttk

root=Tk()
root.title("Pythong Guides")
root.geometry("500x400")

def AttachLabel(panel:PanedWindow, label:StringVar) -> Label:
     l = Label(panel,text=label)
     panel.add(l)
     return l

def AttachPanel(container:PanedWindow, bg="red") -> PanedWindow:
     panel = PanedWindow(bg=bg)
     container.add(panel)
     return panel

pmain = PanedWindow(orient=VERTICAL)
pmain.pack(fill=BOTH)

pheading = AttachPanel(pmain) 

pheading["orient"] = VERTICAL 
AttachLabel(pheading, "HEADING") 
AttachLabel(pheading, "HEADING") 
AttachLabel(pheading, "HEADING") 


pbody = AttachPanel(pmain, bg="green") 
pbody["orient"] = HORIZONTAL
pbody.pack(fill=BOTH) 

pcam = AttachPanel(pbody) 
pcam.pack(side=LEFT, fill=BOTH, expand=1)
pcam_label = AttachLabel(pcam,"PCAM") 

pdetails = AttachPanel(pbody) 
pdetails.pack(side=LEFT, fill=BOTH, expand=1)
pdetails_label = AttachLabel(pdetails,"details")

pbody.pack_propagate(0)
pbody.paneconfig(pcam,weight=70)

root.mainloop();

