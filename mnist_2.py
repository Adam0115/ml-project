# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Preprocessing (reshape for CNN and normalize)
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 3. Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train model
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)

# 5. Predict test set
y_pred = model.predict(x_test)

# 6. Find wrong prediction indices
wrong_indices = [i for i in range(len(x_test)) 
                 if np.argmax(y_test[i]) != np.argmax(y_pred[i])]

# State variables
index = 0
mode = "all"  # "all" or "wrong"

def get_current_indices():
    return range(len(x_test)) if mode == "all" else wrong_indices

def show_image(idx):
    global img_tk
    img = (x_test[idx].reshape(28,28) * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img = img.resize((200, 200))
    img_tk = ImageTk.PhotoImage(img)
    label_img.config(image=img_tk)

    true_label = np.argmax(y_test[idx])
    pred_label = np.argmax(y_pred[idx])

    # Title color hint: correct ¡÷ green, wrong ¡÷ red
    if true_label == pred_label:
        label_text.config(text=f"T:{true_label}, P:{pred_label}", fg="green")
    else:
        label_text.config(text=f"T:{true_label}, P:{pred_label}", fg="red")

def next_image(event=None):
    global index
    indices = get_current_indices()
    if indices:
        index = (index + 1) % len(indices)
        show_image(indices[index])

def prev_image(event=None):
    global index
    indices = get_current_indices()
    if indices:
        index = (index - 1) % len(indices)
        show_image(indices[index])

def switch_mode(new_mode):
    global mode, index
    mode = new_mode
    index = 0
    indices = get_current_indices()
    if indices:
        show_image(indices[index])
    else:
        label_text.config(text="No wrong cases found!", fg="black")
        label_img.config(image="")

# Create Tkinter window
root = tk.Tk()
root.title("MNIST Browser (Left/Right arrows to flip, mode switch)")

label_img = Label(root)
label_img.pack()

label_text = Label(root, font=("Arial", 16))
label_text.pack()

# Mode switch buttons
btn_all = Button(root, text="Show All", command=lambda: switch_mode("all"))
btn_all.pack(side="left", padx=10)

btn_wrong = Button(root, text="Show Wrong Only", command=lambda: switch_mode("wrong"))
btn_wrong.pack(side="right", padx=10)

# Bind keyboard events
root.bind("<Right>", next_image)
root.bind("<Left>", prev_image)

# Show the first image
switch_mode("all")

root.mainloop()

# Execute after GUI window is closed
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

# Statistics
total_samples = len(x_test)
wrong_count = len(wrong_indices)
correct_count = total_samples - wrong_count
accuracy = correct_count / total_samples * 100
error_rate = wrong_count / total_samples * 100

print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")
print(f"Total: {total_samples} | Correct: {correct_count} | Wrong: {wrong_count} | Accuracy: {accuracy:.2f}% | Error Rate: {error_rate:.2f}%")

# Confusion matrix (displayed after GUI is closed)
y_true = np.argmax(y_test, axis=1)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
disp.plot(cmap=plt.cm.Blues)
plt.title("MNIST Confusion Matrix")
plt.show()
