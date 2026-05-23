import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import numpy as np
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import threading
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. 載入資料
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 前處理
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 3. 建立模型
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. 訓練模型
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# 5. 預測測試集
y_pred = model.predict(x_test)

# 6. 找出錯誤案例索引
wrong_indices = [i for i in range(len(x_test)) 
                 if np.argmax(y_test[i]) != np.argmax(y_pred[i])]

# 狀態變數
index = 0
mode = "all"  # 可選 "all" 或 "wrong"

def get_current_indices():
    return range(len(x_test)) if mode == "all" else wrong_indices

def show_image(idx):
    global img_tk
    img = (x_test[idx] * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img = img.resize((200, 200))  # 放大顯示
    img_tk = ImageTk.PhotoImage(img)
    label_img.config(image=img_tk)

    true_label = np.argmax(y_test[idx])
    pred_label = np.argmax(y_pred[idx])

    # 標題顏色提示：正確 → 綠色，錯誤 → 紅色
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

# 建立 Tkinter 視窗
root = tk.Tk()
root.title("MNIST Browser (← → to flip, mode switch)")

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

# ✅ 新增：在另一個 thread 顯示混淆矩陣
def show_confusion_matrix():
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

    total_samples = len(x_test)
    wrong_count = len(wrong_indices)
    correct_count = total_samples - wrong_count
    accuracy = correct_count / total_samples * 100
    error_rate = wrong_count / total_samples * 100

    print(f"Total: {total_samples} | Correct: {correct_count} | Wrong: {wrong_count} | Accuracy: {accuracy:.2f}% | Error Rate: {error_rate:.2f}%")

    y_true = np.argmax(y_test, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("MNIST Confusion Matrix")
    plt.show()

threading.Thread(target=show_confusion_matrix).start()

# Tkinter 主迴圈
root.mainloop()
