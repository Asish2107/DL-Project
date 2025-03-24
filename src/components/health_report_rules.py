import pandas as pd

# ---------- Helper Functions ---------- #

def parse_bp(bp_str):
    try:
        systolic, diastolic = map(int, bp_str.split('/'))
        return systolic, diastolic
    except:
        return None, None

def generate_health_summary(row):
    summary = []
    recommendations = []

    # BMI
    if row['BMI'] < 18.5:
        summary.append("Underweight")
        recommendations.append("Increase caloric intake, consult a dietitian.")
    elif 18.5 <= row['BMI'] < 25:
        summary.append("Normal weight")
        recommendations.append("Maintain healthy lifestyle and regular exercise.")
    elif 25 <= row['BMI'] < 30:
        summary.append("Overweight")
        recommendations.append("Adopt a calorie-controlled diet, increase physical activity.")
    else:
        summary.append("Obese")
        recommendations.append("Seek professional diet and fitness guidance.")

    # Blood Pressure
    systolic, diastolic = parse_bp(row['Blood_Pressure_mmHg'])
    if systolic and diastolic:
        if systolic < 90 or diastolic < 60:
            summary.append("Low blood pressure")
            recommendations.append("Stay hydrated and consult doctor if symptoms persist.")
        elif systolic <= 120 and diastolic <= 80:
            summary.append("Normal blood pressure")
            recommendations.append("Maintain healthy diet and exercise.")
        elif systolic <= 140 or diastolic <= 90:
            summary.append("Pre-hypertension")
            recommendations.append("Reduce salt intake, manage stress levels.")
        else:
            summary.append("Hypertension")
            recommendations.append("Consult a doctor and follow a low-sodium diet.")

    # Heart Rate
    if row['Heart_Rate_bpm'] < 60:
        summary.append("Bradycardia (Low Heart Rate)")
        recommendations.append("If symptomatic, consult a cardiologist.")
    elif row['Heart_Rate_bpm'] > 100:
        summary.append("Tachycardia (High Heart Rate)")
        recommendations.append("Reduce stress, caffeine intake and seek medical advice.")
    else:
        summary.append("Normal heart rate")
        recommendations.append("Good cardiovascular health.")

    # Blood Sugar
    if row['Blood_Sugar_mg_dL'] < 70:
        summary.append("Hypoglycemia")
        recommendations.append("Eat frequent small meals, increase healthy carbs.")
    elif row['Blood_Sugar_mg_dL'] <= 99:
        summary.append("Normal blood sugar")
        recommendations.append("Maintain diet and exercise.")
    elif row['Blood_Sugar_mg_dL'] <= 125:
        summary.append("Prediabetic")
        recommendations.append("Limit sugar intake and increase physical activity.")
    else:
        summary.append("Diabetic")
        recommendations.append("Lifestyle changes and medical management advised.")

    # Cholesterol HDL
    if row['Cholesterol_HDL'] < 40:
        summary.append("Low HDL (Good Cholesterol)")
        recommendations.append("Increase intake of omega-3 fats and exercise regularly.")
    else:
        summary.append("Normal HDL levels")
        recommendations.append("Maintain healthy fat intake and lifestyle.")

    # Cholesterol LDL
    if row['Cholesterol_LDL'] < 100:
        summary.append("Optimal LDL (Bad Cholesterol)")
        recommendations.append("Continue current diet.")
    elif row['Cholesterol_LDL'] < 160:
        summary.append("Borderline High LDL")
        recommendations.append("Avoid fried and fatty foods.")
    else:
        summary.append("High LDL Cholesterol")
        recommendations.append("Strict dietary changes and clinical consultation recommended.")

    # SpO2
    if row['SpO2_Percentage'] < 94:
        summary.append("Low Oxygen Saturation")
        recommendations.append("Pulmonary evaluation recommended.")
    else:
        summary.append("Normal Oxygen Saturation")
        recommendations.append("Healthy respiratory function.")

    # Sleep
    if row['Sleep_Hours'] < 6:
        summary.append("Sleep Deprivation")
        recommendations.append("Ensure at least 6–8 hours of quality sleep.")
    elif row['Sleep_Hours'] > 9:
        summary.append("Excessive Sleep")
        recommendations.append("Check for fatigue-related causes or sleep disorders.")
    else:
        summary.append("Healthy Sleep Pattern")
        recommendations.append("Maintain regular sleep habits.")

    # Smoking
    if row['Smoking_Status'] == "Yes":
        summary.append("Smoker")
        recommendations.append("Strongly advised to quit smoking immediately.")
    elif row['Smoking_Status'] == "Occasional":
        summary.append("Occasional Smoking")
        recommendations.append("Avoid smoking to reduce long-term risk.")

    # Alcohol
    if row['Alcohol_Consumption'] == "Yes":
        summary.append("Alcohol Consumption")
        recommendations.append("Limit intake; consider periodic detox.")
    elif row['Alcohol_Consumption'] == "Occasional":
        summary.append("Occasional Alcohol Intake")
        recommendations.append("Monitor alcohol consumption patterns.")

    return "; ".join(set(summary)), " | ".join(set(recommendations))

# ---------- Load & Process Dataset ---------- #

def generate_report(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df[['Health_Summary', 'Recommendations']] = df.apply(generate_health_summary, axis=1, result_type='expand')
    df.to_csv(output_csv, index=False)
    print(f"[✓] Report generated and saved to: {output_csv}")

# ---------- Main Function ---------- #
if __name__ == "__main__":
    input_csv = "medical_dataset_5000_rows.csv"
    output_csv = "medical_dataset_with_health_report.csv"
    generate_report(input_csv, output_csv)