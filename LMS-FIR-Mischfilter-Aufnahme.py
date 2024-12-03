import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, stft
import tkinter as tk
from tkinter import ttk

# Audioeinstellungen
fs = 44100  # Abtastrate
duration = 5  # Dauer der Aufnahme in Sekunden

# Funktion zur FIR-Filterung
def apply_fir_filter(data, b):
    return lfilter(b, 1, data)

# Funktion zur LMS-Filterung
def apply_lms_filter(desired, noisy, mu=0.01, order=32):
    """
    Einfacher LMS-Filter.
    - desired: gewünschtes Signal (z. B. original ohne Rauschen)
    - noisy: Signal mit Rauschen
    - mu: Lernrate
    - order: Filterordnung
    """
    n = len(noisy)
    w = np.zeros(order)  # Filterkoeffizienten
    filtered_signal = np.zeros(n)

    for i in range(order, n):
        x = noisy[i-order:i][::-1]  # Eingangsvektor
        y = np.dot(w, x)  # Gefilterter Wert
        e = desired[i] - y  # Fehler
        w += 2 * mu * e * x  # LMS-Update
        filtered_signal[i] = y

    return filtered_signal

# FIR-Filter Design
def design_fir_filter(order, lowcut, highcut, fs):
    nyquist = 0.5 * fs
    return firwin(order, [lowcut / nyquist, highcut / nyquist], pass_zero=False)

# Funktion zur Aufnahme
def record_audio(duration, fs):
    print("Aufnahme beginnt ...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Warten bis die Aufnahme abgeschlossen ist
    print("Aufnahme beendet.")
    return audio.flatten()

# Funktion zur Erzeugung von braunem Rauschen
def generate_brown_noise(n_samples, amplitude=0.05):
    """
    Generiert Braunes Rauschen durch Integration von weißem Rauschen.
    - n_samples: Anzahl der zu generierenden Abtastwerte
    - amplitude: Maximale Amplitude des braunen Rauschens
    """
    white_noise = np.random.normal(0, 1, n_samples)
    brown_noise = np.cumsum(white_noise)  # Integration
    brown_noise = amplitude * brown_noise / np.max(np.abs(brown_noise))  # Normierung
    return brown_noise

# Funktion zum Abspielen eines Signals
def play_audio(signal, fs, title):
    print(f"Jetzt abspielen: {title}")
    sd.play(signal, samplerate=fs)
    sd.wait()
    print(f"{title} - Wiedergabe abgeschlossen.\n")

# Plot-Funktion
def plot_signals(original, noisy, filtered, fs):
    t = np.linspace(0, duration, len(original))
    plt.figure(figsize=(12, 12))

    plt.subplot(3, 1, 1)
    plt.plot(t, original, label="Original Signal")
    plt.title("Originales Signal")
    plt.xlabel("Zeit [s]")
    plt.ylabel("Amplitude")
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(t, noisy, label="Mit Rauschen")
    plt.title("Signal mit Rauschen")
    plt.xlabel("Zeit [s]")
    plt.ylabel("Amplitude")
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(t, filtered, label="Gefiltert")
    plt.title("Gefiltertes Signal")
    plt.xlabel("Zeit [s]")
    plt.ylabel("Amplitude")
    plt.grid()

    plt.tight_layout()
    plt.show()

# Hauptprogramm mit GUI
class FilterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Filter GUI")

        # Initialwerte
        self.order = tk.IntVar(value=101)
        self.lowcut = tk.DoubleVar(value=100.0)
        self.highcut = tk.DoubleVar(value=5000.0)
        self.mu = tk.DoubleVar(value=0.01)  # LMS Lernrate
        self.filter_type = tk.StringVar(value="FIR")  # Standard ist FIR

        # Originalsignal und gefilterte Signale
        self.recorded_signal = record_audio(duration, fs)
        self.noise = generate_brown_noise(len(self.recorded_signal), amplitude=0.05)
        self.noisy_signal = self.recorded_signal + self.noise
        self.filtered_signal = None

        # Widgets für GUI
        self.create_widgets()

    def create_widgets(self):
        # Filtertyp auswählen
        tk.Label(self.root, text="Filtertyp auswählen:").pack()
        ttk.Combobox(
            self.root,
            textvariable=self.filter_type,
            values=["FIR", "LMS"],
            state="readonly"
        ).pack()

        # Schieberegler für FIR-Parameter
        tk.Label(self.root, text="FIR-Filter: Filterordnung (Order):").pack()
        tk.Scale(self.root, from_=1, to=500, variable=self.order, orient="horizontal").pack()

        tk.Label(self.root, text="FIR-Filter: Low Cut Frequenz (Hz):").pack()
        tk.Scale(self.root, from_=0, to=fs // 2, variable=self.lowcut, orient="horizontal").pack()

        tk.Label(self.root, text="FIR-Filter: High Cut Frequenz (Hz):").pack()
        tk.Scale(self.root, from_=0, to=fs // 2, variable=self.highcut, orient="horizontal").pack()

        # LMS-Parameter
        tk.Label(self.root, text="LMS-Filter: Lernrate (mu):").pack()
        tk.Scale(self.root, from_=0.001, to=0.1, resolution=0.001, variable=self.mu, orient="horizontal").pack()

        # Buttons
        tk.Button(self.root, text="Filter anwenden", command=self.apply_filter).pack()
        tk.Button(self.root, text="Gefiltertes Signal abspielen", command=self.play_filtered).pack()
        tk.Button(self.root, text="Visualisieren", command=self.visualize).pack()

    def apply_filter(self):
        filter_type = self.filter_type.get()
        if filter_type == "FIR":
            b = design_fir_filter(self.order.get(), self.lowcut.get(), self.highcut.get(), fs)
            self.filtered_signal = apply_fir_filter(self.noisy_signal, b)
            print(f"FIR-Filter angewendet: Order={self.order.get()}, Lowcut={self.lowcut.get()} Hz, Highcut={self.highcut.get()} Hz")
        elif filter_type == "LMS":
            self.filtered_signal = apply_lms_filter(self.recorded_signal, self.noisy_signal, mu=self.mu.get(), order=self.order.get())
            print(f"LMS-Filter angewendet: Lernrate={self.mu.get()}, Order={self.order.get()}")

    def play_filtered(self):
        if self.filtered_signal is not None:
            play_audio(self.filtered_signal, fs, "Gefiltertes Signal")
        else:
            print("Bitte erst den Filter anwenden!")

    def visualize(self):
        if self.filtered_signal is not None:
            plot_signals(self.recorded_signal, self.noisy_signal, self.filtered_signal, fs)
        else:
            print("Bitte erst den Filter anwenden!")

# Starte das GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = FilterGUI(root)
    root.mainloop()
