import os
import threading
import time
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter as tk
from ultralytics import YOLO
from datetime import datetime
import torch

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# ---------------- CONFIG ----------------
STREAMS_FILE = r'C:\Users\rao\Work\scripts\intrusion dettection in multi camera\list.streams.txt'
MAX_CAMERAS = 16
FPS_DISPLAY = 10
RECONNECT_DELAY = 5
LOG_FILE = 'intrusion_log.txt'
DETECTION_CONF_THRESHOLD = 0.3
MODEL_NAME = r'C:\Users\rao\Work\scripts\intrusion detection in multi camera\yolo11n.pt'
CAM_WIDTH = 640
CAM_HEIGHT = 480
BLINK_INTERVAL = 0.5  # seconds for alert blink

# ---------------- DEVICE CHECK ----------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ---------------- GLOBALS ----------------
streams = []
running = True
cam_state = {}
cam_detected = {}  # Person detection state per camera
blink_state = {}   # Blink toggle per camera
last_blink_time = {}  # Timestamp of last blink toggle
state_lock = threading.Lock()
yolo_model = YOLO(MODEL_NAME)

# ---------------- LOG FUNCTION ----------------
def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, 'a') as f:
        f.write(f"[{ts}] {msg}\n")

# ---------------- CAMERA THREAD ----------------
class CaptureThread(threading.Thread):
    def __init__(self, idx, url):
        super().__init__(daemon=True)
        self.idx = idx
        self.url = url
        self.cap = None

    def open_capture(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.url)
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
            except:
                pass

    def close_capture(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None

    def run(self):
        global running
        while running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    self.open_capture()
                    if self.cap is None or not self.cap.isOpened():
                        with state_lock:
                            cam_state[self.idx] = None
                            cam_detected[self.idx] = False
                        log(f"Camera {self.idx}: failed to open")
                        time.sleep(RECONNECT_DELAY)
                        continue

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    with state_lock:
                        cam_state[self.idx] = None
                        cam_detected[self.idx] = False
                    log(f"Camera {self.idx}: read failed")
                    self.close_capture()
                    time.sleep(RECONNECT_DELAY)
                    continue

                # Person detection only
                results = yolo_model.predict(frame, device=device, verbose=False,
                                             conf=DETECTION_CONF_THRESHOLD, classes=[0])
                frame = results[0].plot(conf=False)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Update global states
                with state_lock:
                    cam_state[self.idx] = frame
                    detected = any(yolo_model.names[int(box.cls[0])] == 'person'
                                   for r in results for box in r.boxes)
                    cam_detected[self.idx] = detected
                    if detected:
                        log(f"Camera {self.idx}: PERSON detected")

                time.sleep(1.0 / FPS_DISPLAY)
            except Exception as e:
                with state_lock:
                    cam_state[self.idx] = None
                    cam_detected[self.idx] = False
                log(f"Camera {self.idx}: Exception {e}")
                self.close_capture()
                time.sleep(RECONNECT_DELAY)

        self.close_capture()

# ---------------- GUI ----------------
class MultiCamGUI:
    def __init__(self, root, ncam):
        self.root = root
        self.ncam = ncam
        self.cells = []
        self.photo_refs = [None]*ncam
        self._calculate_grid()
        self.no_feed_img = self._create_no_feed_image()
        self._build()

    def _calculate_grid(self):
        # Dynamically calculate rows and columns for camera count
        self.rows = int(np.ceil(np.sqrt(self.ncam)))
        self.cols = int(np.ceil(self.ncam / self.rows))
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        # Maximize cell size to fill screen
        self.cell_width = screen_width // self.cols
        self.cell_height = screen_height // self.rows

    def _create_no_feed_image(self):
        img = Image.new('RGB', (self.cell_width, self.cell_height), color='black')
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0,0), "No Feed", font=font)
        w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        draw.text(((self.cell_width - w)/2, (self.cell_height - h)/2), "No Feed", fill='white', font=font)
        return ImageTk.PhotoImage(img)

    def _build(self):
        self.root.title("Multi-Camera Person Detection")
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill='both', expand=True)

        for r in range(self.rows):
            for c in range(self.cols):
                idx = r*self.cols + c
                frame = tk.Frame(main_frame, width=self.cell_width, height=self.cell_height, bg='black', relief='sunken')
                frame.grid(row=r, column=c, padx=1, pady=1)
                frame.grid_propagate(False)

                lbl = tk.Label(frame, image=self.no_feed_img, bg='black')
                lbl.pack(fill='both', expand=True)

                if idx < self.ncam:
                    self.cells.append(lbl)

        # Make rows and columns expand evenly
        for r in range(self.rows):
            main_frame.rowconfigure(r, weight=1)
        for c in range(self.cols):
            main_frame.columnconfigure(c, weight=1)

        self.root.after(int(1000/FPS_DISPLAY), self._refresh)

    def _refresh(self):
        now = time.time()
        with state_lock:
            for idx in range(self.ncam):
                frame = cam_state.get(idx)
                lbl = self.cells[idx]

                # Handle blinking alert
                if cam_detected.get(idx, False):
                    last_time = last_blink_time.get(idx, 0)
                    if now - last_time > BLINK_INTERVAL:
                        blink_state[idx] = not blink_state.get(idx, False)
                        last_blink_time[idx] = now
                else:
                    blink_state[idx] = False

                if frame is not None:
                    img = Image.fromarray(frame)
                    img = img.resize((self.cell_width, self.cell_height))

                    # Draw blinking alert text
                    if blink_state.get(idx, False):
                        draw = ImageDraw.Draw(img)
                        try:
                            font = ImageFont.truetype("arial.ttf", 30)
                        except:
                            font = ImageFont.load_default()
                        text = "INTRUSION DETECTED"
                        text_w, text_h = draw.textsize(text, font=font)
                        padding = 10
                        x = img.width - text_w - padding
                        y = padding
                        draw.text((x, y), text, fill='red', font=font)

                    imgtk = ImageTk.PhotoImage(img)
                    self.photo_refs[idx] = imgtk
                    lbl.configure(image=imgtk)
                else:
                    lbl.configure(image=self.no_feed_img)

        self.root.after(int(1000/FPS_DISPLAY), self._refresh)

# ---------------- HELPERS ----------------
def read_streams(file_path):
    if not os.path.exists(file_path):
        log(f"Streams file not found: {file_path}")
        return []
    with open(file_path, 'r') as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]

# ---------------- MAIN ----------------
def main():
    global streams, running
    streams = read_streams(STREAMS_FILE)
    if not streams:
        print("No streams found in", STREAMS_FILE)
        return

    streams = streams[:MAX_CAMERAS]
    ncam = len(streams)

    with state_lock:
        for i in range(ncam):
            cam_state[i] = None
            cam_detected[i] = False
            blink_state[i] = False
            last_blink_time[i] = 0

    threads = [CaptureThread(i, url) for i, url in enumerate(streams)]
    for t in threads:
        t.start()

    root = tk.Tk()
    root.state('zoomed')  # maximize window
    gui = MultiCamGUI(root, ncam)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass

    running = False
    time.sleep(0.5)

if __name__ == "__main__":
    main()
