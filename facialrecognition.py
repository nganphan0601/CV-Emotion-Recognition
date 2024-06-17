import tkinter as tk 
from tkinter import filedialog, ttk
from PIL import Image, ImageOps, ImageTk, ImageFilter, ImageGrab  
import numpy as np 
import cv2 

root = tk.Tk()
root.geometry("1000x600")
root.title("Let's Generate Some Images")
root.config(bg="Grey")

file_path = ""
current_image = None

def add_image():
    global file_path, current_image
    file_path = filedialog.askopenfilename(initialdir='/Users/gurmanhunjan/gurman/test') #modify to your path
    image = Image.open(file_path)
    width, height = int(image.width / 2), int(image.height / 2)
    image = image.resize((width, height), Image.LANCZOS)
    canvas.config(width=image.width, height=image.height)
    current_image = image
    image = ImageTk.PhotoImage(image)
    canvas.image = image
    canvas.create_image(0, 0, image=image, anchor="nw")

def clear_canvas():
    canvas.delete("all")

def save_image():
    x0 = canvas.winfo_rootx()
    y0 = canvas.winfo_rooty()
    x1 = x0 + canvas.winfo_width()
    y1 = y0 + canvas.winfo_height()

    im = ImageGrab.grab((x0, y0, x1, y1))
    im.save('saveimage.png')
    im.show()

def apply_filter(filter_name):
    if filter_name == "Size down":
        size_down()
    elif filter_name == "Size up":
        size_up()
    elif filter_name == "Rotate":
        rotate()
    elif filter_name == "Facial Recognition":
        faces()
    

def size_down():
    global current_image
    if file_path and current_image:
        width, height = int(current_image.width / 4), int(current_image.height / 4)
        resized_image = current_image.resize((width, height), Image.LANCZOS)
        current_image = resized_image
        display_image(current_image)

def size_up():
    global current_image
    if file_path and current_image:
        image = Image.open(file_path)
        width, height = int(current_image.width * 2), int(current_image.height * 2)
        resized_image = image.resize((width, height), Image.LANCZOS)
        current_image = resized_image
        display_image(current_image)



def faces():
    global current_image
    if file_path and current_image:
        image = np.asarray(current_image)
        face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_classifier.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_color = image[y:y + h, x:x + w]
            sharpened_face = cv2.detailEnhance(roi_color, sigma_s=10, sigma_r=0.15)
            image[y:y + h, x:x + w] = sharpened_face
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        display_image(image)



def display_image(image):
    photo = ImageTk.PhotoImage(image)
    canvas.config(width=image.width, height=image.height)
    canvas.image = photo
    canvas.create_image(0, 0, image=photo, anchor="nw")

left_frame = tk.Frame(root, width=200, height=600, bg="white")
left_frame.pack(side="left", fill="y")

canvas = tk.Canvas(root, width=750, height=600)
canvas.pack()

first_button = tk.Button(left_frame, text="Add Image", command=add_image, bg="white")
first_button.pack(pady=15)

clear_button = tk.Button(left_frame, text="Clear", command=clear_canvas, bg="#FF9797")
clear_button.pack(pady=10)

save_button = tk.Button(left_frame, text="Save Image", command=save_image, bg="lightblue")
save_button.pack(pady=15)

filter_label = tk.Label(left_frame, text="Image Interpolation", bg="white")
filter_label.pack()
filter_combobox = ttk.Combobox(left_frame, values=["Size down", "Size up" ,"Facial Recognition"])
filter_combobox.pack()

filter_combobox.bind("<<ComboboxSelected>>", lambda event: apply_filter(filter_combobox.get()))

root.mainloop()
