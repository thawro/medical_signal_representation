import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import wfdb
from joblib import Parallel, delayed
from tqdm import tqdm

if __name__ == "__main__":
    mimic_name = "mimic3wdb-matched"

    required_signals = ["ABP", "PLETH", "II"]

    all_signals_available = lambda header: all([signal in header for signal in required_signals])

    all_records = requests.get(f"https://physionet.org/files/{mimic_name}/1.0/RECORDS").text.split("/\n")
    all_records = [record for record in all_records if len(record) > 0]

    def is_good_measurement(record, measurement):
        header = requests.get(f"https://physionet.org/files/{mimic_name}/1.0/{record}/{measurement}.hea").text
        if all_signals_available(header):
            return f"{record}/{measurement}"
        else:
            return None

    good_measurements = []

    for record in tqdm(all_records, desc="Records"):
        record_measurements = requests.get(f"https://physionet.org/files/{mimic_name}/1.0/{record}/RECORDS").text.split(
            "\n"
        )
        record_measurements = [measurement for measurement in record_measurements if len(measurement) > 0]
        results = Parallel(n_jobs=-1)(
            delayed(is_good_measurement)(record, measurement)
            for measurement in tqdm(record_measurements, desc="Record measurements", leave=False)
        )
        results = [result for result in results if result is not None]
        good_measurements.extend(results)

    textfile = open("good_measurements.txt", "w")

    for element in good_measurements:

        textfile.write(element + "\n")

    textfile.close()
