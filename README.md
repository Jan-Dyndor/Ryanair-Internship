# ✈️ Ryanair Take-Off Weight Prediction – Regression Project

## 📌 Project Summary
This project was completed as part of a recruitment process for a Data Science Internship at Ryanair.  
The goal was to develop a regression model capable of accurately predicting the **Take-Off Weight (TOW)** of an aircraft based on historical flight data.

---

## 🧠 Objective
Create a machine learning pipeline that:
- Preprocesses and encodes features (including high-cardinality variables).
- Trains a **regression model** to predict `ActualTOW` using the **training dataset (Oct 1–15, 2016)**.
- Applies the trained model to a **validation set** and saves predictions in `.csv` format.
- Evaluation metric: **Root Mean Squared Error (RMSE)**.

---

## 📁 Dataset Overview

| Column Name         | Description                          |
|---------------------|--------------------------------------|
| DepartureDate       | Date of departure                    |
| DepartureYear       | Year of departure                    |
| DepartureMonth      | Month of departure                   |
| DepartureDay        | Day of departure                     |
| FlightNumber        | Flight number                        |
| DepartureAirport    | Departure airport code               |
| ArrivalAirport      | Arrival airport code                 |
| Route               | Route (Departure–Arrival)            |
| ActualFlightTime    | Flight time (in minutes)             |
| ActualTotalFuel     | Burnt fuel (in kg)                   |
| FlownPassengers     | Number of flown passengers           |
| BagsCount           | Number of checked-in bags            |
| FlightBagsWeight    | Total weight of bags (in kg)         |
| ActualTOW           | ✅ Target variable (Take-Off Weight) |

---

## 🔍 Exploratory Data Analysis (EDA)
EDA was conducted to:
- Check for missing values, invalid strings (e.g., `(null)`), and data types.
- Investigate the impact of:
  - Weekends and day of the week on TOW.
  - Passenger count and fuel consumption.
  - Airport and route frequency distribution.
- Data distribution
- Identify data cleaning steps required before modeling.

---

## 🛠️ Feature Engineering
- **Datetime features**: extracted `is_weekend` and `day_of_week`.
- **Numerical imputation**: median imputation + missing indicators.
- **Categorical encoding**:
  - `Route` → Target mean encoding using `MeanEncoder` and rare label grouping
  - `DepartureAirport` & `ArrivalAirport` → Frequency encoding and rare label grouping.
- **Scaling & Transformation**:
  - Applied `PowerTransformer` (Yeo-Johnson) and `StandardScaler` to numerical features.

---

## 🤖 Modeling Approach

### ⏳ Cross-Validation
Used **TimeSeriesSplit (n_splits=5)** for time-respecting validation.

In this case I belive that chronological integrity of the data is critical, as each record corresponds to a real historical flight.  
Using standard random `train_test_split` could result in **data leakage**, where the model "sees the future" during training (e.g., training on data from October 10 and validating on October 5).  
To avoid this, we used `TimeSeriesSplit`, which:
- Preserves temporal order between training and validation sets.
- Mimics real-life scenario: training on past data to predict future outcomes.
- Provides a more realistic estimate of model performance on unseen future flights.

### 🔄 Stacking Regressor
This ensemble approach leverages the strengths of both tree-based and linear models, allowing the final predictor (Ridge) to combine their outputs in an optimized way.  
It helps improve generalization and reduce overfitting compared to individual models.

Built an ensemble using multiple models:

```python
stacking_model = StackingRegressor(
 estimators = [
    ("rf", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
    ("gb", GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)),
    ("xgb", XGBRegressor(random_state=42)),
    ("lasso",Lasso())
]
    final_estimator=Ridge(alpha=1.0, random_state=42)
```
## 🚧 Challenges & Reflections
- Balancing between high-cardinality categorical variables and overfitting risk.
- Ensuring consistent preprocessing between training and validation datasets.
- Selecting optimal models for ensemble stacking without extensive overfitting.
  
- ⚠️ Due to time constraints, I was not able to perform full hyperparameter tuning.
I'm aware that this step could significantly improve the model's generalization ability and would definitely include it if more time was available.
