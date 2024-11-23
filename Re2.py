import sounddevice as sd
import numpy as np
from scipy.signal import butter, lfilter, firwin 
import tkinter as tk
from tkinter import Scale
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading

# Sample rate and buffer size
fs = 44100
buffer_size = 1024  # Size of the buffer for visualization

# Initialize filter parameters
lowcut = 300  # Initial low cutoff frequency
highcut = 3000  # Initial high cutoff frequency
order = 5  # Initial filter order

# Initialize an empty array for the audio buffer
audio_buffer = np.zeros(buffer_size)

# Function to create a bandpass filter
# Filter Butterworth (hier nicht verwendet)
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Function to apply the filter to the data
#def apply_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = lfilter(b, a, data)
    return y

# Filter FIRWIN (Höherer Rechenaufwand)
def bandpass (lowcut, highcut, fs, order):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b = firwin(order, [low, high], pass_zero=False)
    return b

# Function to apply the filter to the data
def apply_filter(data, lowcut, highcut, fs, order):
    b = bandpass(lowcut, highcut, fs, order)
    y = lfilter(b, 1, data)  #Faltung mit FIR- Filter
    return y

# Audio callback function 
def audio_callback(indata, outdata, frames, time, status):
    global audio_buffer
    if status: # Gibt Statusmeldungen aus, falls Fehler auftreten
        print(status)
    # Apply the filter and store the filtered data
    filtered_data = apply_filter(indata[:, 0], lowcut, highcut, fs, order)
    outdata[:, 0] = filtered_data  # send the filtered signal to the output
    audio_buffer = filtered_data[:buffer_size]  # Store the latest buffer for visualization

# GUI for controlling filter parameters
def update_lowcut(val):
    global lowcut
    lowcut = int(val)

def update_highcut(val):
    global highcut
    highcut = int(val)

def update_order(val):
    global order
    order = int(val)

# Visualization setup
fig, (ax1, ax2) = plt.subplots(2, 1)

# Time-domain plot
# Empty plots mit Grenzen
time_line, = ax1.plot([], [], lw=2)  #Komma sorgt dafür nur das erste Element zu kriegen
ax1.set_ylim(-1, 1)
ax1.set_xlim(0, buffer_size)
ax1.set_title("Zeitbereich (Gefiltertes Signal)")
ax1.set_xlabel("Sample")
ax1.set_ylabel("Amplitude")

# Frequency-domain plot
freq_line, = ax2.plot([], [], lw=2) # x und y leer initialisieren
ax2.set_ylim(0, 0.1)
ax2.set_xlim(0, fs / 2)
ax2.set_title("Frequenzbereich (Spektrum des gefilterten Signals)")
ax2.set_xlabel("Frequenz (Hz)")
ax2.set_ylabel("Amplitude")

# Initialize the lines
# Sagen, empty plots aktualisiern sich regelmässig
def init():
    time_line.set_data([], [])    # Selbe wie bei Time-domain plot 
    freq_line.set_data([], [])
    return time_line, freq_line

# Update the plots
# Data der zu aktualisierenen Plots definieren 
def update_plot(frame):
    # Update time-domain plot
    time_line.set_data(np.arange(buffer_size), audio_buffer) # Mit filtriertem Data aktualisieren mit arange in x Achse
    # audio_buffer ist das filtrierte Signal

    # Compute and update frequency-domain plot
    fft_data = np.abs(np.fft.rfft(audio_buffer)) / buffer_size  #rfft - Nur positiven Frequenzen, NORMIERT 
    freqs = np.fft.rfftfreq(buffer_size, 1/fs) 
    freq_line.set_data(freqs, fft_data) # Mit Amplituden-Spektrum aktualisieren und frequenzauflösung in x Achse
    
    return time_line, freq_line

# Animate the plot
ani = FuncAnimation(fig, update_plot, init_func=init, blit=True, interval=50)


# Function to run the tkinter GUI in a separate thread
def run_gui():
    root = tk.Tk()  # Hauptfenster erstellen 
    root.title("Echtzeit-Rauschunterdrückung - Filtereinstellungen")

    # Scale Widgets
    lowcut_scale = Scale(root, from_=20, to=10000, orient="horizontal", label="Low Cut Frequency", command=update_lowcut)
    lowcut_scale.set(lowcut)
    lowcut_scale.pack()

    highcut_scale = Scale(root, from_=20, to=10000, orient="horizontal", label="High Cut Frequency", command=update_highcut)
    highcut_scale.set(highcut)
    highcut_scale.pack()

    order_scale = Scale(root, from_=1, to=1000, orient="horizontal", label="Filter Order", command=update_order)
    order_scale.set(order)
    order_scale.pack()

    root.mainloop()

# Start audio stream and run both GUI and plot simultaneously
def main(): # Zur Klarheit in der Struktur
    # Start the tkinter GUI in a separate thread
    gui_thread = threading.Thread(target=run_gui)
    gui_thread.start()

    # Start audio stream and show matplotlib plot
    # Länge indata definiert automatisch, sonst blocksize eintragen 
    # Definierte Audiogrösse: frames = blocksize; Audiolänge: frames/ fs
    with sd.Stream(callback=audio_callback, samplerate=fs, channels=1, dtype='float32'):
        plt.show()

main()
