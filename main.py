import os
import io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import wavfile
from scipy import signal

def parquet_to_spec_png(parquet_dir, parquet_file, png_dir):
    parquet_path = f"{parquet_dir}{parquet_file}"
    df = pd.read_parquet((parquet_path), engine = "pyarrow")
    for i in range(df["audio"].size):
        label = "drone/" if df["label"][i] == 1 else "other/"
        orig_name = df["audio"][i]["path"]
        path = f"{png_dir}{label}{os.path.splitext(orig_name)[0]}.png"

        wav_file = io.BytesIO(df["audio"][i]["bytes"])
        sample_rate, samples = wavfile.read(wav_file)
        freqs, times, sxx = signal.spectrogram(samples, sample_rate)
    
        plt.figure(figsize=(10, 5), frameon=False)
        plt.axis('off')
        plt.pcolormesh(times, freqs, np.log10(sxx), shading="gouraud")
        plt.savefig(path)
        plt.close()
        print(f"[-] {parquet_file} -> {os.path.splitext(orig_name)[0]}.png")
        
def main():
    parquet_dir = "./parquet_data/"
    png_dir = "./png_data/"
    if not(os.path.isdir(parquet_dir)):
        print(f"[-] {parquet_dir} is missing")
        return

    parquet_files = os.listdir(parquet_dir)
    for f in parquet_files:
        parquet_to_spec_png(parquet_dir, f, png_dir)


if (__name__ == "__main__"):
    main()
