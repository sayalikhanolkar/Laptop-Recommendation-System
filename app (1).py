from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load files
try:
    with open("model_rf.pkl", "rb") as f:
        model = pickle.load(f)
    with open("input_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open("target_encoder.pkl", "rb") as f:
        target_encoder = pickle.load(f)
    print("Loaded encoders contain:", list(encoders.keys()))
except Exception as e:
    print(f"Error loading files: {e}")
    raise

@app.route("/")
def index():
    try:
        # Create options dictionary
        options = {
            "TypeName": list(encoders["TypeName"].classes_),
            "Cpu_Model": list(encoders["Cpu_Model"].classes_),
            "Cpu_Speed": list(encoders["Cpu_Speed"].classes_),
            "Memory_Size": list(encoders["Memory_Size"].classes_),
            "Memory_Type": list(encoders["Memory_Type"].classes_),
            "ScreenResolution":list(encoders["ScreenResolution"].classes_),
            "Gpu": list(encoders["Gpu"].classes_),
            "OpSys": list(encoders["OpSys"].classes_)
        }
        
        # Print available options for debugging
        print("\nAvailable options:")
        for key, values in options.items():
            print(f"{key}: {len(values)} options")
            
        return render_template("index.html", options=options)
    except Exception as e:
        print(f"Error in index route: {e}")
        return str(e), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Create input dataframe
        input_data = {
            "TypeName": data["TypeName"],
            "Inches": float(data["Inches"]),
            "Ram": int(data["Ram"]),
            "Cpu_Model": data["Cpu_Model"],
            "Cpu_Speed": data["Cpu_Speed"],
            "Memory_Size": data["Memory_Size"],
            "Memory_Type": data["Memory_Type"],
            "ScreenResolution":data["ScreenResolution"],
            "Gpu": data["Gpu"],
            "OpSys": data["OpSys"],
            "Weight": float(data["Weight"]),
            "Price_euros": float(data["Price_euros"])
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical features
        for col, encoder in encoders.items():
            if col in input_df.columns:
                input_df[col] = encoder.transform(input_df[col].astype(str))
        
        # Make prediction
        prediction = model.predict(input_df)
        result = target_encoder.inverse_transform(prediction)[0]
        
        return render_template("result.html", recommendation=result)
    except Exception as e:
        print(f"Error in predict route: {e}")
        return str(e), 500

if __name__ == "__main__":
    app.run(debug=True)