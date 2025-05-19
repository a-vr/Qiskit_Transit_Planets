import os
import pandas as pd
import numpy as np

def normalize_lightcurves(input_dir):
    os.makedirs("normalized_" + input_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(input_dir, filename)
            try:
                # Load data
                df = pd.read_csv(filepath)
                if 'flux' not in df.columns:
                    print(f"Skipping {filename}: 'flux' column not found.")
                    continue

                # Drop NaNs from flux
                df = df.dropna(subset=['flux'])

                # Normalize flux
                median_flux = np.nanmedian(df['flux'])
                df['flux'] = df['flux'] / median_flux

                # Preprocessing: trim/pad flux
                L = 500
                flux = df['flux'].values
                if len(flux) > L:
                    flux = flux[:L]
                elif len(flux) < L:
                    flux = np.pad(flux, (0, L - len(flux)), 'constant', constant_values=0)

                # Preprocessing: trim/pad time (if it exists)
                if 'time' in df.columns:
                    time = df['time'].values
                    time = (time - np.min(time)) / (np.max(time) - np.min(time))  # normalize time
                    if len(time) > L:
                        time = time[:L]
                    elif len(time) < L:
                        time = np.pad(time, (0, L - len(time)), 'constant', constant_values=0)
                    out_df = pd.DataFrame({'time': time, 'flux': flux})
                else:
                    out_df = pd.DataFrame({'flux': flux})

                # Save
                output_path = os.path.join("normalized_" + input_dir, "norm_" + filename)
                out_df.to_csv(output_path, index=False)
                print(f"Saved: {output_path}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")
