# ⚙️ Tennis Match Charting Data Preprocessing Pipeline

This directory contains the data transformation pipeline for converting raw event-level Match Charting Project point logs (`charting-m-points.csv`) into structured, kinematic spatial features ready for Machine Learning training (`processed_epv_data.csv`).

---

## 📌 Overview

Raw tennis match charting datasets log point-by-point play-by-play events (such as shot sequences, rally length, and serve indicators), but lack direct spatial player positioning coordinates. 

`preprocess_charting_data.py` bridges this gap by parsing match mechanics and mapping shot pressure, forced lateral movement, and rally fatigue into three continuous spatial features used by the Kinematic Expected Point Value (EPV) Engine.

---

## 🛠️ Input & Output Data Specs

### 1. Input: `charting-m-points.csv`
The raw dataset from the Match Charting Project. The ETL script specifically parses:
* `isSvrWinner`: Point outcome (1 = Server Won, 0 = Server Lost).
* `rallyLen`: Number of total shots in the rally.
* `isForced`: Flag for forced errors.
* `isUnforced`: Flag for unforced errors.
* `isAce`: Flag for serve aces.

### 2. Output: `processed_epv_data.csv`
The exported, cleaned dataset contains 4 feature columns:

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `recovery_deficit_m` | Float | Estimated distance between ideal defensive reach spot and actual player reach spot (meters). |
| `exposed_area_m2` | Float | Estimated open court surface area left unprotected due to positioning deficit ($m^2$). |
| `deficit_x_exposed` | Float | Interaction term (`recovery_deficit_m` × `exposed_area_m2`). |
| `target` | Integer | Binary target variable ($1 = \text{Point Won by Server}$, $0 = \text{Point Lost by Server}$). |

