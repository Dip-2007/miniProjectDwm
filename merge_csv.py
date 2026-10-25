"""
SafeTraffic - CSV Dataset Merge Utility
Combines all individual network traffic CSV files from the data/ folder
into a single consolidated CSV file (data/merged_traffic.csv).
"""

import glob
import os
import time


def merge_traffic_csvs(
    data_dir: str = "data",
    output_filename: str = "merged_traffic.csv",
) -> str:
    start_time = time.time()
    output_path = os.path.join(data_dir, output_filename)

    # Exclude any previously merged files
    input_files = sorted([
        f for f in glob.glob(os.path.join(data_dir, "*.csv"))
        if os.path.basename(f) not in (output_filename, "combined_traffic.csv")
    ])

    if not input_files:
        print("No CSV files found in data directory.")
        return ""

    print(f"Found {len(input_files)} CSV file(s) to merge into '{output_path}':")
    for f in input_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  - {os.path.basename(f)} ({size_mb:.1f} MB)")

    total_rows = 0

    with open(output_path, "wb") as outfile:
        for i, file_path in enumerate(input_files):
            with open(file_path, "rb") as infile:
                if i == 0:
                    # Write header + content for the first file
                    header = infile.readline()
                    outfile.write(header)
                else:
                    # Skip header for all subsequent files
                    infile.readline()

                # Stream file in 1MB chunks
                while True:
                    chunk = infile.read(1024 * 1024)
                    if not chunk:
                        break
                    outfile.write(chunk)

            print(f"Merged: {os.path.basename(file_path)}")

    # Verify line count
    with open(output_path, "rb") as f:
        total_lines = sum(1 for _ in f)

    total_rows = total_lines - 1
    elapsed = time.time() - start_time
    size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print("\nMerge Complete:")
    print(f"  - Output file: {output_path}")
    print(f"  - File size:   {size_mb:.2f} MB")
    print(f"  - Total rows:  {total_rows:,} records")
    print(f"  - Time taken:  {elapsed:.2f} seconds")

    return output_path


if __name__ == "__main__":
    merge_traffic_csvs()
