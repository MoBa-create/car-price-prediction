import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

url_cars = "https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Car%20Price.csv"
df = pd.read_csv(url_cars)

df['Car_Age'] = 2026 - df['Year']
df = df.drop('Year', axis=1)

df['Model_Short'] = df['Model'].apply(lambda x: str(x).split()[0])
df = df.drop('Model', axis=1)

df = pd.get_dummies(df, columns=['Fuel', 'Seller_Type', 'Transmission', 'Owner'], drop_first=True)

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_df = X_train.copy()
train_df['Selling_Price'] = y_train

model_means = train_df.groupby('Model_Short')['Selling_Price'].mean()
brand_means = train_df.groupby('Brand')['Selling_Price'].mean()
overall_mean = y_train.mean()

X_train['Model_Avg'] = X_train['Model_Short'].map(model_means).fillna(overall_mean)
X_train['Brand_Avg'] = X_train['Brand'].map(brand_means).fillna(overall_mean)

X_test['Model_Avg'] = X_test['Model_Short'].map(model_means).fillna(X_test['Brand'].map(brand_means)).fillna(overall_mean)
X_test['Brand_Avg'] = X_test['Brand'].map(brand_means).fillna(overall_mean)

X_train = X_train.drop(['Model_Short', 'Brand'], axis=1)
X_test = X_test.drop(['Model_Short', 'Brand'], axis=1)

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- النتائج النهائية المعتمدة للمشروع ---")
print(f"MAE (متوسط الخطأ): {mae:,.2f}")
print(f"R² Score (الدقة): {r2 * 100:.2f}%")