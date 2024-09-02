import tkinter as tk
from music_analyzer.app import MusicAnalyzerApp

if __name__ == "__main__":
    root = tk.Tk()
    app = MusicAnalyzerApp(root)
    root.mainloop()
