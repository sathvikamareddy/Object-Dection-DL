import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import threading

# ===============================
# LOAD MODEL
# ===============================
try:
    model = YOLO("yolov8n.pt")
except Exception as e:
    print("Error loading model:", e)
    exit()

# ===============================
# GLOBAL VARIABLES
# ===============================
cap = None
running = False

# ===============================
# CREATE MAIN WINDOW
# ===============================
root = tk.Tk()
root.title("Object Detection Using Deep Learning")
root.geometry("1000x750")
root.configure(bg="#f5f5f5")

# ===============================
# UI COMPONENTS
# ===============================
title = tk.Label(root, text="Object Detection Using Deep Learning",
                 font=("Arial", 20, "bold"), bg="#f5f5f5")
title.pack(pady=10)

display_label = tk.Label(root, bg="black")
display_label.pack(pady=20)

result_label = tk.Label(root, text="Detected Objects:",
                        font=("Arial", 14), bg="#f5f5f5")
result_label.pack(pady=5)


# ===============================
# IMAGE DISPLAY FUNCTION
# ===============================
def show_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((800, 500))
    imgtk = ImageTk.PhotoImage(image=img)
    display_label.imgtk = imgtk
    display_label.config(image=imgtk)


# ===============================
# UPDATE RESULT TEXT
# ===============================
def update_result(objects):
    if objects:
        result_label.config(
            text="Detected Objects:\n" + "\n".join(objects))
    else:
        result_label.config(text="Detected Objects: None")


# ===============================
# UPLOAD IMAGE FUNCTION
# ===============================
def upload_image():
    global running
    running = False

    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    image = cv2.imread(file_path)

    results = model(image)[0]
    detected_objects = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        detected_objects.append(f"{label} ({conf:.2f})")

        cv2.rectangle(image, (x1, y1), (x2, y2),
                      (0, 255, 0), 2)
        cv2.putText(image, f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    update_result(detected_objects)
    show_image(image)


# ===============================
# WEBCAM FUNCTION
# ===============================
def start_camera():
    global cap, running

    if running:
        return  # already running

    running = True
    cap = cv2.VideoCapture(0)

    def run_camera():
        global running, cap

        while running:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)[0]
            detected_objects = []

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                detected_objects.append(label)

                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

            update_result(list(set(detected_objects)))
            show_image(frame)

        if cap:
            cap.release()

    threading.Thread(target=run_camera, daemon=True).start()


# ===============================
# CLEAR DISPLAY
# ===============================
def clear_display():
    global running, cap
    running = False

    if cap:
        cap.release()
        cap = None

    display_label.config(image="")
    result_label.config(text="Detected Objects:")


# ===============================
# EXIT APPLICATION
# ===============================
def exit_app():
    global running, cap
    running = False

    if cap:
        cap.release()

    root.destroy()


# ===============================
# BUTTONS
# ===============================
button_frame = tk.Frame(root, bg="#f5f5f5")
button_frame.pack(pady=20)

upload_btn = tk.Button(button_frame, text="Upload Image",
                       font=("Arial", 12), bg="#2e8b57",
                       fg="white", width=15,
                       command=upload_image)
upload_btn.grid(row=0, column=0, padx=20, pady=10)

camera_btn = tk.Button(button_frame, text="Start Webcam",
                       font=("Arial", 12), bg="#2e8b57",
                       fg="white", width=15,
                       command=start_camera)
camera_btn.grid(row=0, column=1, padx=20, pady=10)

clear_btn = tk.Button(button_frame, text="Clear",
                      font=("Arial", 12), bg="#a9a9a9",
                      fg="white", width=15,
                      command=clear_display)
clear_btn.grid(row=1, column=0, padx=20, pady=10)

exit_btn = tk.Button(button_frame, text="Exit",
                     font=("Arial", 12), bg="#b22222",
                     fg="white", width=15,
                     command=exit_app)
exit_btn.grid(row=1, column=1, padx=20, pady=10)

# ===============================
# RUN APP
# ===============================
root.mainloop()f
