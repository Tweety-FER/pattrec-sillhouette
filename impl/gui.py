import sys
from Tkinter import *
from tkFileDialog import askopenfilename
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import segmentation
import numpy as np
import matplotlib.pyplot as plt

graph = None

def get_classification():
    max_frames = 40 if v.get() == 'Kamera' else -1
    source = 0 if v.get() == 'Kamera' else v.get()
    return segmentation.classify(source, max_frames = max_frames, display = True)

def get_frames():
    if segmentation.g_frames != None:
        return segmentation.g_frames
    max_frames = 40 if v.get() == 'Kamera' else -1
    source = 0 if v.get() == 'Kamera' else v.get()
    return segmentation.process_frames(source, max_frames = max_frames)

def klasificiraj():
    frames = get_classification()
    print frames
    v_class.set('Akcija je: ' + segmentation.most_common_frame(frames).upper())

def camera():
    v.set('Kamera')

def prikaziVektore():
    global label, graph
    vectors = get_frames()

    rng = np.arange(1,9)

    f = Figure(dpi=100, figsize=(4,4))
    ax = f.add_subplot(111)

    for vector in vectors:
        ax.plot(rng, vector)
    ax.grid()
    ax.set_ylim(-1,1)
    
    canvas.delete('all')

    if graph != None:
        graph.get_tk_widget().destroy()

    graph = FigureCanvasTkAgg(f, canvas)
    graph.get_tk_widget().pack(fill = X)
    graph.show()

def prikaziKuteve():
    return

def fileChooser():
    Tk().withdraw()
    filename = askopenfilename()
    v.set(filename)
    return

mGui = Tk()
mGui.geometry("700x600")
mGui.title("Raspoznavanje uzoraka")

v_class = StringVar()
classification = Label(mGui, textvariable=v_class).pack(fill = X)

canvas = Canvas(mGui, height=400)
canvas.pack(fill=X);
photo = PhotoImage(file="image.gif")
canvas.create_image(0, 0, anchor=NW, image=photo)
#label = Label(canvas, image=photo).pack(fill = X)

v = StringVar()
#label2= Label(mGui, bg = "white", textvariable=v).pack(fill = X)
entry = Entry(mGui, textvariable=v).pack( fill = X)
button0 = Button(text = "Odaberi Video", command = fileChooser).pack( fill = X)
buttonV = Button(text = "Koristi Kameru", command = camera).pack( fill = X)
button1 = Button(text = "Klasificiraj", command = klasificiraj).pack( fill = X)
button2 = Button(text = "Prikazi vektore", command = prikaziVektore).pack(fill = X)
button3 = Button(text = "Prikazi kuteve", command = prikaziKuteve).pack(fill = X)
mGui.mainloop()