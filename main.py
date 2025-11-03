import os
import io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import wavfile
from scipy import signal

def parquet_to_spec_png(parquet_dir, parquet_file, png_dir):
    parquet_path = f"{parquet_dir}{parquet_file}"
    parquet_id = parquet_file.split('-')[1]
    df = pd.read_parquet((parquet_path), engine = "pyarrow")
    for i in range(df["audio"].size):
        label = "drone/" if df["label"][i] == 1 else "other/"
        orig_name = df["audio"][i]["path"]
        path = f"{png_dir}{label}{os.path.splitext(orig_name)[0]}-{parquet_id}.png"

        os.makedirs(os.path.dirname(path), exist_ok=True)

        wav_file = io.BytesIO(df["audio"][i]["bytes"])
        sample_rate, samples = wavfile.read(wav_file)
        if (samples.ndim > 1):
            print("f[-] wav file needs to be mono")
            return

        freqs, times, zxx = signal.stft(samples, fs=sample_rate)
        magnitude = 20 * np.log10(np.abs(zxx) + 1e-10)

        plt.figure(figsize=(10, 5), frameon=False)
        plt.axis('off')
        plt.pcolormesh(times, freqs, magnitude, shading="gouraud")
        plt.savefig(path)
        plt.close()
        
        print(f"[-] {parquet_file} -> {os.path.splitext(orig_name)[0]}-{parquet_id}.png")

def main():
    parquet_dir = "../parquet_data/"
    png_dir = "../png_data/"
    if not(os.path.isdir(parquet_dir)):
        print(f"[-] {parquet_dir} is missing")
        return

    parquet_files = os.listdir(parquet_dir)
    for f in parquet_files:
        parquet_to_spec_png(parquet_dir, f, png_dir)


if (__name__ == "__main__"):
    main()
