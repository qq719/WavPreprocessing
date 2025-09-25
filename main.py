import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

def wav_to_spec_png(wav_dir, wav_file, png_dir):
    wav_path = f"{wav_dir}{wav_file}"
    png_path = f"{png_dir}{os.path.splitext(wav_file)[0]}.png"

    sample_rate, samples = wavfile.read(wav_path)
    freqs, times, sxx = signal.spectrogram(samples, sample_rate)
    
    plt.figure(figsize=(10, 5), frameon=False)
    plt.axis('off')
    plt.pcolormesh(times, freqs, np.log10(sxx), shading="gouraud")
    plt.savefig(png_path)
    plt.close()
    
    print(f"[-] {wav_path} -> {png_path}")
    
def main():
    png_dir = "./png_data/"
    if not(os.path.isdir(png_dir)):
        print(f"[-] {png_dir} is missing")
        return

    # Convert WAV files to PNG spectrogram
    wav_dir = "./wav_data/"
    if not(os.path.isdir(wav_dir)):
        print(f"[-] {wav_dir} is missing")
        return
    wav_files = os.listdir(wav_dir)
    for f in wav_files:
        wav_to_spec_png(wav_dir, f, png_dir)
    
if (__name__ == "__main__"):
    main()
