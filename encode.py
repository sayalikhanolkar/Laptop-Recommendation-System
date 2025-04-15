import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv('laptop_price_updated.csv', encoding='latin-1')

# Data Cleaning
df['Ram'] = df['Ram'].astype(str).str.replace('GB', '').astype(int)
df['Weight'] = df['Weight'].astype(str).str.replace('kg', '').astype(float)

# Extract CPU Model and Speed
def split_cpu(cpu):
    cpu = cpu.replace('GHz', '').strip()
    parts = cpu.rsplit(' ', 1)
    if len(parts) == 2:
        model, speed = parts
        return pd.Series([model.strip(), speed.strip()])
    else:
        return pd.Series([None, None])

df[['Cpu_Model', 'Cpu_Speed']] = df['Cpu'].apply(split_cpu)

# Expand rows with '+' in Memory column
df = df.assign(Memory=df['Memory'].str.split(' + ')).explode('Memory', ignore_index=True)

# Function to split Memory into Size and Type
def split_memory(memory):
    parts = memory.split(' ', 1)
    if len(parts) == 2:
        size, mem_type = parts
        return pd.Series([size.strip(), mem_type.strip()])
    else:
        return pd.Series([None, None])

df[['Memory_Size', 'Memory_Type']] = df['Memory'].apply(split_memory)
df['Memory_Type'] = df['Memory_Type'].str.replace(r'\+', '', regex=True)

# Convert TB to GB in Memory_Size
df['Memory_Size'] = df['Memory_Size'].apply(lambda x: float(x.replace('TB', '')) * 1024 if 'TB' in x else float(x.replace('GB', '')))

# Create target column
df['Company_Product'] = df['Company'] + ' ' + df['Product']

# Select features for modeling
feature_columns = [
    'TypeName',
    'Inches',
    'Ram',
    'Cpu_Model',
    'Cpu_Speed',
    'Memory_Size',
    'Memory_Type',
    'ScreenResolution',
    'Gpu',
    'OpSys',
    'Weight',
    'Price_euros'
]

# Initialize encoders dictionary
encoders = {}

# Create and fit encoders for categorical columns
categorical_columns = ['TypeName', 'Cpu_Model', 'Cpu_Speed', 'Memory_Type','Memory_Size', 'ScreenResolution', 'Gpu', 'OpSys']
for col in categorical_columns:
    encoders[col] = LabelEncoder()
    df[col] = df[col].fillna('Unknown')
    df[col] = encoders[col].fit_transform(df[col].astype(str))

# Create target encoder
target_encoder = LabelEncoder()
df['Company_Product'] = target_encoder.fit_transform(df['Company_Product'])

# Prepare features and target
X = df[feature_columns]
y = df['Company_Product']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Save model and encoders
with open('model_rf.pkl', 'wb') as f:
    pickle.dump(model_rf, f)

with open('input_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

with open('target_encoder.pkl', 'wb') as f:
    pickle.dump(target_encoder, f)

# Print verification
print("\nEncoders created for:")
for col, encoder in encoders.items():
    print(f"{col}: {len(encoder.classes_)} classes")

# Verify saved files
print("\nVerifying saved files:")
with open('input_encoders.pkl', 'rb') as f:
    loaded_encoders = pickle.load(f)
    print("\nLoaded encoders contain:", list(loaded_encoders.keys()))

# Missing value analysis
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
print("\nMissing values percentage per column:")
print(missing_percentage)