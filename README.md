# Household Power Consumption Forecasting — LSTM | Deep Learning ⚡🏠

A deep learning time series forecasting model using an LSTM (Long Short-Term Memory) 
network to predict daily household global active power consumption. Trained with a 
30-day lookback window using TensorFlow/Keras.

---

## 🧩 Overview

This project uses an **LSTM neural network** to learn temporal patterns in household 
energy consumption data and forecast future daily power usage. LSTMs are well-suited 
for time series tasks due to their ability to retain long-term dependencies through 
memory cells and gating mechanisms.

---

## 📦 Dataset

- **Source:** UCI Household Power Consumption Dataset
- **File:** `household_power_consumption.txt`
- **Feature used:** `Global_active_power` (kilowatts)
- **Resampled:** Minute-level data aggregated to **daily averages**
- **Missing values:** Dropped (`NaN` and `?` entries removed)

### Preprocessing
- Data resampled to daily frequency using `.resample('D').mean()`
- Normalized using **MinMaxScaler** to range [0, 1]
- Sequences created with a **30-day lookback window**
- **Train/Test split:** 80% training, 20% testing

---

## 🧠 LSTM Model Architecture

```
Input (30 timesteps, 1 feature)
    ↓
LSTM(50 units, activation='relu')
    ↓
Dense(1)   ← regression output
```

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Batch size | 32 |
| Optimizer | Adam |
| Loss | Mean Squared Error |
| Validation | X_test / y_test |

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **RMSE** | 3.4082 kW |

> Note: Training loss returned `NaN` values due to missing data in certain daily 
> aggregations. NaN values in predictions were handled using `np.nan_to_num()`.

### Visualization
- **Predictions vs Actual** — daily global active power over the test period

---

## ⚠️ Known Issues & Handling

| Issue | Handling |
|-------|---------|
| NaN in y_test | 6 NaN values replaced with 0 via `np.nan_to_num()` |
| NaN in y_pred | 283 NaN values replaced with 0 via `np.nan_to_num()` |
| Training loss NaN | Caused by NaN propagation in daily resampled data |

---

## 🛠️ Setup & Usage

### Prerequisites
- Python 3.10+
- TensorFlow / Keras
- scikit-learn, pandas, numpy, matplotlib

### Install Dependencies
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
```

### Run
```bash
jupyter notebook LSTM.ipynb
```


---

## 📁 File Structure

```
Household-Power-Consumption-LSTM/
├── LSTM.ipynb                          # Main notebook
├── household_power_consumption.txt     # Dataset file
└── README.md
```

---

## 📝 License

MIT License — for academic use.
