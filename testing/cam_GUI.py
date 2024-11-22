import datetime
import os
import time
import tkinter as tk

from PIL import ImageTk, Image
from libcamera import controls
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder

if not os.path.exists("recordings"):
    os.mkdir("recordings")


def init_camera(camera):
    if "AfMode" in camera.camera_controls:
        print("afmode=continuous!! ", controls.AfModeEnum.Continuous)
        camera.set_controls({"AfMode": controls.AfModeEnum.Continuous})  # Enable continuous autofocus

class CameraApp:
    def __init__(self, master, camera_index):
        self.master = master
        self.camera = Picamera2(camera_index)
        init_camera(self.camera)
        self.preview_started = False
        self.is_recording = False

        # Configure the main window
        self.master.title("Raspberry Pi Camera")
        self.master.geometry('800x600')
        
        # Duration Label
        self.duration_label = tk.Label(master, text="Duration: 00:00:00")
        self.duration_label.pack(side=tk.BOTTOM, padx=20, pady=20)

        # Start/Stop Recording Button
        self.button = tk.Button(master, text="Start Recording", command=self.toggle_recording)
        self.button.pack(side=tk.BOTTOM, pady=0)

        # Dropdown Menu for Video Resolution
        self.resolution_var = tk.StringVar(master)
        self.resolution_var.set("1920x1080")
        self.resolutions = ["2560x1440","1920x1440","1920x1080", "1280x720", "1024x768", "720x480", "640x480", "640x360"]
        self.dropdown = tk.OptionMenu(master, self.resolution_var, *self.resolutions, command=self.change_resolution)
        self.dropdown.pack(side=tk.BOTTOM, pady=10)

        # Preview Frame
        self.preview_label = tk.Label(master)
        self.preview_label.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        self.master.update_idletasks()
        self.start_preview()

    def start_preview(self):
        self.camera_config = self.camera.create_preview_configuration(main={"size": (1920, 1080)})
        self.camera.configure(self.camera_config)
        init_camera(self.camera)
        self.camera.startt()
        self.update_preview()

    def update_preview(self):
        image = self.camera.capture_array()
        pil_image = Image.fromarray(image)

        label_width = self.preview_label.winfo_width()
        label_height = self.preview_label.winfo_height()

        image_aspect_ratio = pil_image.width / pil_image.height
        label_aspect_ratio = label_width / label_height

        if label_aspect_ratio > image_aspect_ratio:
            new_height = label_height
            new_width = int(new_height * image_aspect_ratio)
        else:
            new_width = label_width
            new_height = int(new_width / image_aspect_ratio)

        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)

        photo = ImageTk.PhotoImage(image=pil_image)
    
        self.preview_label.configure(image=photo)
        self.preview_label.image = photo
        
        self.update_duration()
        
        self.master.after(60, self.update_preview)

    def toggle_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.camera.stop_recording()
            self.button.config(text="Start Recording")
        else:
            self.is_recording = True
            self.recording_start_time = time.time()
            self.camera.start_recording(output=f"recordings/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.h264", encoder=H264Encoder(50000000))
            self.button.config(text="Stop Recording")

    def change_resolution(self, value):
        width, height = map(int, value.split('x'))
        self.camera.stop()
        self.camera_config = self.camera.create_video_configuration(main={"size": (width, height)})
        self.camera.configure(self.camera_config)
        init_camera(self.camera)
        self.camera.startt()
        
    def update_duration(self):
        if self.is_recording:
            elapsed_time = int(time.time() - self.recording_start_time)
        else:
            elapsed_time = 0
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.duration_label.config(text=f"Duration: {hours:02}:{minutes:02}:{seconds:02}")
        self.master.after(500, self.update_duration)
            

def choose_camera():
    root = tk.Tk()
    root.title("Select Camera")

    def on_camera_selected(index):
        root.destroy()
        main(index)

    camera_names = ["Camera 0","Camera 1"]
    for index, name in enumerate(camera_names):
        btn = tk.Button(root, text=name, command=lambda idx=index: on_camera_selected(idx))
        btn.pack(pady=10, padx=100)

    root.mainloop()

# Main function to run the application
def main(camera_index=0):
    root = tk.Tk()
    app = CameraApp(root, camera_index)
    root.mainloop()

if __name__ == "__main__":
    choose_camera()
