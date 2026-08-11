# 🌌 Detection and Analysis of GW170814: A Binary Black Hole Merger

---

## 📌 Project Overview
This project presents an end-to-end data analysis and study of the gravitational wave signal **GW170814**. Observed on August 14, 2017, GW170814 was a landmark astrophysical event—marking the first-ever joint detection of a binary black hole (BBH) coalescence by the three-detector network comprising **LIGO Hanford**, **LIGO Livingston**, and **Virgo**.

---

## 🌌 Key Objectives
- 📡 Analyze the strain time-series data from LIGO and Virgo observatories for event GW170814.
- 🎛️ Perform signal processing including bandpass filtering, notch filtering, and whitening to isolate the gravitational wave chirp from detector noise.
- 📉 Generate time-frequency representations (**Q-transforms / Spectrograms**) to observe the chirp signal evolution across all three detectors.
- 🎯 Evaluate parameter estimation aspects including chirp mass, peak strain amplitude, and sky localization benefits of triple-detector coincidence.

---

## 🛠️ Tech Stack & Libraries
- **Language**: Python 3
- **Primary Libraries**: `GWpy`, `PyCBC`, `NumPy`, `Matplotlib`, `SciPy`
- **Data Source**: LIGO Open Science Center (GWOSC)

---

## 📂 Project Structure
```text
├── Detection_and_Analysis_of_GW170814.pdf    # Comprehensive Project Report
├── gw170814_analysis.py                      # Signal processing & spectrogram script
└── README.md                                 # Project documentation
