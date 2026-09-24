import os
import joblib
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

# 1. LOAD THE ADVANCED ACCIDENT MODEL
MODEL_FILE = "accident_model.joblib"

if os.path.exists(MODEL_FILE):
    model_pipeline = joblib.load(MODEL_FILE)
    print("Advanced AI Model loaded into GUI!")
else:
    raise FileNotFoundError(f"'{MODEL_FILE}' not found. Please run 'train_model.py' first!")


# 2. TKINTER GUI APPLICATION
class AdvancedAccidentRiskApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Accident Risk Predictor")
        self.root.geometry("480x620")
        self.root.configure(bg="#1e1e2e")
        self.root.resizable(False, False)

        title_label = tk.Label(
            root, text="AI Accident Risk Predictor", 
            font=("Helvetica", 16, "bold"), fg="#cdd6f4", bg="#1e1e2e", pady=15
        )
        title_label.pack()

        frame = tk.Frame(root, bg="#1e1e2e", padx=20, pady=5)
        frame.pack(fill="both", expand=True)

        # Form Dropdowns
        self.weather_var = tk.StringVar(value="Clear")
        self.create_dropdown(frame, "Weather Condition:", self.weather_var, ["Clear", "Rain", "Fog", "Snow"])

        self.road_var = tk.StringVar(value="Dry")
        self.create_dropdown(frame, "Road Condition:", self.road_var, ["Dry", "Wet", "Icy"])

        self.time_var = tk.StringVar(value="Day")
        self.create_dropdown(frame, "Time of Day:", self.time_var, ["Day", "Night"])

        # Speed Limit Entry
        tk.Label(frame, text="Speed Limit (mph):", font=("Helvetica", 10), fg="#a6adc8", bg="#1e1e2e").pack(anchor="w", pady=(5, 2))
        self.speed_entry = ttk.Entry(frame)
        self.speed_entry.insert(0, "45")
        self.speed_entry.pack(fill="x", pady=(0, 5))

        # Driver Age Entry
        tk.Label(frame, text="Driver Age (years):", font=("Helvetica", 10), fg="#a6adc8", bg="#1e1e2e").pack(anchor="w", pady=(5, 2))
        self.age_entry = ttk.Entry(frame)
        self.age_entry.insert(0, "25")
        self.age_entry.pack(fill="x", pady=(0, 5))

        # Visibility Entry
        tk.Label(frame, text="Visibility Distance (km):", font=("Helvetica", 10), fg="#a6adc8", bg="#1e1e2e").pack(anchor="w", pady=(5, 2))
        self.vis_entry = ttk.Entry(frame)
        self.vis_entry.insert(0, "8.0")
        self.vis_entry.pack(fill="x", pady=(0, 10))

        # Predict Button
        predict_btn = tk.Button(
            frame, text="ASSESS ACCIDENT RISK", font=("Helvetica", 11, "bold"),
            bg="#89b4fa", fg="#11111b", activebackground="#74c7ec",
            relief="flat", pady=8, command=self.predict_risk
        )
        predict_btn.pack(fill="x", pady=10)

        # Output Display Card
        self.result_card = tk.Frame(frame, bg="#313244", padx=15, pady=12)
        self.result_card.pack(fill="x", pady=10)

        self.prob_label = tk.Label(self.result_card, text="Risk Probability: --%", font=("Helvetica", 12, "bold"), fg="#cdd6f4", bg="#313244")
        self.prob_label.pack()

        self.status_label = tk.Label(self.result_card, text="Status: Awaiting Input", font=("Helvetica", 10), fg="#a6adc8", bg="#313244")
        self.status_label.pack(pady=(4, 0))

    def create_dropdown(self, parent, text, variable, options):
        tk.Label(parent, text=text, font=("Helvetica", 10), fg="#a6adc8", bg="#1e1e2e").pack(anchor="w", pady=(4, 2))
        dropdown = ttk.Combobox(parent, textvariable=variable, values=options, state="readonly")
        dropdown.pack(fill="x", pady=(0, 4))

    def predict_risk(self):
        try:
            speed = float(self.speed_entry.get())
            age = float(self.age_entry.get())
            vis = float(self.vis_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for Speed, Age, and Visibility.")
            return

        input_data = pd.DataFrame([{
            'weather': self.weather_var.get(),
            'road_condition': self.road_var.get(),
            'time_of_day': self.time_var.get(),
            'speed_limit': speed,
            'driver_age': age,
            'visibility_km': vis
        }])

        risk_prob = model_pipeline.predict_proba(input_data)[0][1] * 100
        
        self.prob_label.config(text=f"Calculated Risk Probability: {risk_prob:.1f}%")

        if risk_prob >= 50.0:
            self.result_card.config(bg="#f38ba8")
            self.prob_label.config(bg="#f38ba8", fg="#11111b")
            self.status_label.config(text="CRITICAL ACCIDENT RISK", bg="#f38ba8", fg="#11111b")
        else:
            self.result_card.config(bg="#a6e3a1")
            self.prob_label.config(bg="#a6e3a1", fg="#11111b")
            self.status_label.config(text="SAFE DRIVING CONDITIONS", bg="#a6e3a1", fg="#11111b")


if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedAccidentRiskApp(root)
    root.mainloop()
