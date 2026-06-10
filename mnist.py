import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import numpy as np
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Preprocessing
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(f"Train samples: {len(x_train)}, Test samples: {len(x_test)}")

# 3. Build model (MLP)
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train model and record history
history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=2
)

# 5. Predict test set
y_pred = model.predict(x_test)

# 6. Find wrong prediction indices
wrong_indices = [i for i in range(len(x_test)) 
                 if np.argmax(y_test[i]) != np.argmax(y_pred[i])]

# GUI state variables
index = 0
mode = "all"

def get_current_indices():
    return range(len(x_test)) if mode == "all" else wrong_indices

def show_image(idx):
    global img_tk
    img = (x_test[idx] * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img = img.resize((200, 200))
    img_tk = ImageTk.PhotoImage(img)
    label_img.config(image=img_tk)

    true_label = np.argmax(y_test[idx])
    pred_label = np.argmax(y_pred[idx])

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

# Tkinter window
root = tk.Tk()
root.title("MNIST Browser")

label_img = Label(root)
label_img.pack()

label_text = Label(root, font=("Arial", 16))
label_text.pack()

btn_all = Button(root, text="Show All", command=lambda: switch_mode("all"))
btn_all.pack(side="left", padx=10)

btn_wrong = Button(root, text="Show Wrong Only", command=lambda: switch_mode("wrong"))
btn_wrong.pack(side="right", padx=10)

root.bind("<Right>", next_image)
root.bind("<Left>", prev_image)

switch_mode("all")

# Run Tkinter main loop
root.mainloop()

# ---- After closing Tkinter, show analysis ----
def show_confusion_matrix():
    y_true = np.argmax(y_test, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title("MNIST Confusion Matrix")
    plt.show(block=False)

def plot_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Accuracy vs Epochs')
    axes[0].legend()

    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Loss vs Epochs')
    axes[1].legend()

    fig.tight_layout()
    plt.show(block=False)

# Call analysis after GUI closes
show_confusion_matrix()
plot_training_curves()

# Evaluate test set and print detailed stats
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
y_true = np.argmax(y_test, axis=1)
y_pred_classes = np.argmax(y_pred, axis=1)
total_samples = len(y_true)
correct_count = np.sum(y_true == y_pred_classes)
wrong_count = total_samples - correct_count
accuracy = correct_count / total_samples * 100
error_rate = wrong_count / total_samples * 100

print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")
print(f"Total: {total_samples} | Correct: {correct_count} | Wrong: {wrong_count} | Accuracy: {accuracy:.2f}% | Error Rate: {error_rate:.2f}%")
