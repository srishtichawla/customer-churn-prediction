import pandas as pd

URL = "https://raw.githubusercontent.com/AlaaNabil98/CodeClause_Customer_Churn_Rate_Analysis/main/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(URL)
df["Churn"] = (df["Churn"] == "Yes").astype(int)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.to_csv("telco_clean.csv", index=False)
print("Saved telco_clean.csv:", df.shape, "| churn rate:", round(df["Churn"].mean(), 3))
