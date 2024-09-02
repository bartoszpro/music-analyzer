import tkinter as tk
from tkinter import filedialog
from pygame import mixer
import librosa
import numpy as np
import time


def format_time(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}:{seconds:02}"


class MusicAnalyzerApp:
    def __init__(self, root):
        self.load_button = None
        self.file_label = None
        self.progress_bar = None
        self.progress_rect = None
        self.time_label = None
        self.volume_label = None
        self.volume_slider = None
        self.analyze_button = None
        self.advanced_button = None
        self.advanced_frame = None
        self.sample_rate_menu = None
        self.root = root
        self.root.title("Music Analyzer")
        self.root.geometry("284x466")
        self.root.resizable(False, False)

        mixer.init()

        self.music_file = None
        self.song_length = 0

        self.advanced_mode = False
        self.sample_rate = tk.IntVar(value=22050)

        self.labels = {}
        self.advanced_labels = {}

        self.setup_ui()

    def setup_ui(self):
        self.load_button = tk.Button(self.root, text="Load MP3 File", command=self.load_file)
        self.file_label = tk.Label(self.root, text="No file loaded")

        self.progress_bar = tk.Canvas(self.root, width=280, height=20, bg='gray')
        self.progress_rect = self.progress_bar.create_rectangle(0, 0, 0, 20, fill='blue')

        self.time_label = tk.Label(self.root, text="0:00 / 0:00")

        button_frame = tk.Frame(self.root)
        self.setup_buttons(button_frame)

        volume_frame = tk.Frame(self.root)
        self.volume_label = tk.Label(volume_frame, text="Volume")
        self.volume_slider = tk.Scale(volume_frame, from_=0, to=1, resolution=0.1, orient=tk.HORIZONTAL,
                                      command=self.set_volume)
        self.volume_slider.set(0.5)

        self.analyze_button = tk.Button(self.root, text="Analyze", command=self.analyze_audio_features)

        self.advanced_button = tk.Button(self.root, text="Advanced", command=self.toggle_advanced)

        self.advanced_frame = tk.Frame(self.root)
        sample_rate_frame = tk.Frame(self.advanced_frame)
        tk.Label(sample_rate_frame, text="Sample Rate:").pack(side=tk.LEFT)
        sample_rates = [22050, 32000, 44100, 48000, 96000]
        self.sample_rate_menu = tk.OptionMenu(sample_rate_frame, self.sample_rate, *sample_rates)

        self.load_button.pack(pady=10)
        self.file_label.pack(pady=10)
        self.progress_bar.pack(pady=(5, 0))
        self.time_label.pack(pady=(0, 5))
        button_frame.pack(pady=10)
        volume_frame.pack(pady=10)
        self.volume_label.pack(pady=(5, 5))
        self.volume_slider.pack()
        self.analyze_button.pack(pady=10)

        for text in ["BPM", "Key"]:
            self.labels[text] = tk.Label(self.root, text=f"{text}: N/A")
            self.labels[text].pack(pady=5)

        self.advanced_button.pack(pady=10)

        sample_rate_frame.pack(pady=(5, 0))
        self.sample_rate_menu.pack(side=tk.LEFT, padx=5)

        for text in ["Zero-Crossing Rate", "Spectral Centroid", "Spectral Bandwidth", "Spectral Flatness"]:
            self.advanced_labels[text] = tk.Label(self.advanced_frame, text=f"{text}: N/A")
            self.advanced_labels[text].pack(pady=5)

        self.advanced_frame.pack(pady=5)
        self.advanced_frame.pack_forget()

    def setup_buttons(self, button_frame):
        buttons = [
            ("Play", self.play_music),
            ("Pause", self.pause_music),
            ("Resume", self.resume_music),
            ("Stop", self.stop_music)
        ]

        for text, command in buttons:
            tk.Button(button_frame, text=text, command=command).pack(side=tk.LEFT, padx=10)

    def load_file(self):
        self.music_file = filedialog.askopenfilename(filetypes=[("MP3 Files", "*.mp3")])
        if self.music_file:
            mixer.music.load(self.music_file)
            self.song_length = mixer.Sound(self.music_file).get_length()
            self.file_label.config(text=f"Loaded: {self.music_file.split('/')[-1]}")
            self.time_label.config(text=f"0:00 / {format_time(self.song_length)}")
        else:
            self.file_label.config(text="No file loaded")

    def play_music(self):
        if self.music_file:
            mixer.music.play()
            self.update_progress_bar()

    def pause_music(self):
        if self.music_file:
            mixer.music.pause()

    def resume_music(self):
        if self.music_file:
            mixer.music.unpause()

    def stop_music(self):
        if self.music_file:
            mixer.music.stop()

    def set_volume(self, val):
        if self.music_file:
            mixer.music.set_volume(float(val))

    def update_progress_bar(self):
        if self.music_file and mixer.music.get_busy():
            current_position = mixer.music.get_pos() / 1000
            progress = (current_position / self.song_length) * self.progress_bar.winfo_width()
            self.progress_bar.coords(self.progress_rect, 0, 0, progress, 20)
            self.time_label.config(text=f"{format_time(current_position)} / {format_time(self.song_length)}")
            self.root.after(500, self.update_progress_bar)

    def analyze_audio_features(self):
        if self.music_file:
            start_time = time.time()
            self.perform_analysis()
            elapsed_time = time.time() - start_time
            print(f"Analysis completed in {elapsed_time:.2f} seconds")

    def perform_analysis(self):
        sr = self.sample_rate.get()

        y, sr = librosa.load(self.music_file, sr=sr)

        hop_length = 512
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, aggregate=np.median)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
        rounded_tempo = round(tempo[0] if isinstance(tempo, (np.ndarray, list)) else tempo)

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        key_index = chroma_mean.argmax()
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key = keys[key_index]
        mode = 'Minor' if key_index in [1, 3, 6, 8, 10] else 'Major'

        zcr = librosa.feature.zero_crossing_rate(y)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)

        self.labels["BPM"].config(text=f"BPM: {rounded_tempo}")
        self.labels["Key"].config(text=f"Key: {key} {mode}")
        self.advanced_labels["Zero-Crossing Rate"].config(text=f"Zero-Crossing Rate: {zcr.mean():.2f}")
        self.advanced_labels["Spectral Centroid"].config(text=f"Spectral Centroid: {spectral_centroid.mean():.2f}")
        self.advanced_labels["Spectral Bandwidth"].config(text=f"Spectral Bandwidth: {spectral_bandwidth.mean():.2f}")
        self.advanced_labels["Spectral Flatness"].config(text=f"Spectral Flatness: {spectral_flatness.mean():.2f}")
        print('Analysis complete')

    def toggle_advanced(self):
        self.advanced_mode = not self.advanced_mode
        if self.advanced_mode:
            self.advanced_frame.pack(pady=10)
            self.root.geometry("")
        else:
            self.advanced_frame.pack_forget()
            self.root.geometry("")


if __name__ == "__main__":
    root = tk.Tk()
    app = MusicAnalyzerApp(root)
    root.mainloop()
