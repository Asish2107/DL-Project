import pandas as pd

# ---------- Helper Functions ---------- #

def parse_bp(bp_str):
    try:
        if isinstance(bp_str, str) and '/' in bp_str:
            systolic, diastolic = map(int, bp_str.split('/'))
            return systolic, diastolic
        else:
            return None, None
    except ValueError:
        return None, None

def generate_health_summary(row):
    summary = []
    recommendations = []

    # BMI (Based on WHO Classification)
    if row['BMI'] < 18.5:
        summary.append("Underweight")
        recommendations.append("Increase caloric intake with nutrient-dense foods; consult a dietitian.")
    elif 18.5 <= row['BMI'] < 25:
        summary.append("Normal weight")
        recommendations.append("Maintain a balanced diet and regular physical activity.")
    elif 25 <= row['BMI'] < 30:
        summary.append("Overweight")
        recommendations.append("Adopt a calorie-controlled diet, engage in at least 150 minutes of exercise per week.")
    else:
        summary.append("Obese")
        recommendations.append("Seek professional diet and fitness guidance; consider medical evaluation if needed.")

    # Blood Pressure (Based on AHA Guidelines)
    systolic, diastolic = parse_bp(row['Blood_Pressure_mmHg'])
    if systolic is not None and diastolic is not None:
        if systolic < 90 or diastolic < 60:
            summary.append("Hypotension (Low Blood Pressure)")
            recommendations.append("Increase hydration and salt intake if recommended by a doctor.")
        elif 90 <= systolic < 120 and 60 <= diastolic < 80:
            summary.append("Normal Blood Pressure")
            recommendations.append("Maintain a healthy diet, exercise regularly.")
        elif 120 <= systolic < 130 and diastolic < 80:
            summary.append("Elevated Blood Pressure")
            recommendations.append("Reduce sodium intake, exercise regularly, and monitor blood pressure.")
        elif 130 <= systolic < 140 or 80 <= diastolic < 90:
            summary.append("Hypertension Stage 1")
            recommendations.append("Adopt a low-sodium diet (DASH diet), manage stress, and consult a physician.")
        elif systolic >= 140 or diastolic >= 90:
            summary.append("Hypertension Stage 2")
            recommendations.append("Medical intervention may be required; follow a strict low-sodium diet and lifestyle changes.")

    # Heart Rate (Normal Range: 60-100 bpm)
    if row['Heart_Rate_bpm'] < 60:
        summary.append("Bradycardia (Low Heart Rate)")
        recommendations.append("If experiencing dizziness or fatigue, consult a cardiologist.")
    elif row['Heart_Rate_bpm'] > 100:
        summary.append("Tachycardia (High Heart Rate)")
        recommendations.append("Limit caffeine, manage stress, and seek medical evaluation if persistent.")
    else:
        summary.append("Normal Heart Rate")
        recommendations.append("Maintain cardiovascular fitness through regular exercise.")

    # Blood Sugar (Based on ADA Guidelines)
    if row['Blood_Sugar_mg_dL'] < 70:
        summary.append("Hypoglycemia (Low Blood Sugar)")
        recommendations.append("Consume fast-acting carbohydrates; seek medical advice if recurrent.")
    elif 70 <= row['Blood_Sugar_mg_dL'] <= 99:
        summary.append("Normal Blood Sugar")
        recommendations.append("Maintain a balanced diet and regular physical activity.")
    elif 100 <= row['Blood_Sugar_mg_dL'] <= 125:
        summary.append("Prediabetes")
        recommendations.append("Reduce sugar intake, increase fiber, and engage in at least 150 minutes of exercise per week.")
    else:
        summary.append("Diabetes")
        recommendations.append("Monitor blood sugar levels, follow a diabetic-friendly diet, and consult a physician.")

    # Cholesterol HDL (Good Cholesterol)
    if row['Cholesterol_HDL'] < 40:
        summary.append("Low HDL (Good Cholesterol)")
        recommendations.append("Increase intake of omega-3 fatty acids, engage in regular aerobic exercise.")
    else:
        summary.append("Normal HDL Levels")
        recommendations.append("Maintain healthy fat intake and an active lifestyle.")

    # Cholesterol LDL (Bad Cholesterol)
    if row['Cholesterol_LDL'] < 100:
        summary.append("Optimal LDL (Bad Cholesterol)")
        recommendations.append("Continue following a heart-healthy diet.")
    elif 100 <= row['Cholesterol_LDL'] < 160:
        summary.append("Borderline High LDL")
        recommendations.append("Reduce saturated fats, increase fiber intake, and avoid processed foods.")
    else:
        summary.append("High LDL Cholesterol")
        recommendations.append("Adopt strict dietary changes, increase physical activity, and consider medical intervention.")

    # SpO2 (Normal Range: 95-100%)
    if row['SpO2_Percentage'] < 94:
        summary.append("Low Oxygen Saturation")
        recommendations.append("Monitor oxygen levels; seek medical evaluation if persistent.")
    else:
        summary.append("Normal Oxygen Saturation")
        recommendations.append("Healthy respiratory function, maintain an active lifestyle.")

    # Sleep (Recommended: 7-9 hours per night)
    if row['Sleep_Hours'] < 6:
        summary.append("Sleep Deprivation")
        recommendations.append("Improve sleep hygiene; aim for at least 7-9 hours of sleep per night.")
    elif row['Sleep_Hours'] > 9:
        summary.append("Excessive Sleep")
        recommendations.append("Assess for underlying conditions such as sleep apnea or chronic fatigue.")
    else:
        summary.append("Healthy Sleep Pattern")
        recommendations.append("Maintain a consistent sleep schedule and avoid screen exposure before bedtime.")

    # Smoking
    if row['Smoking_Status'] == "Yes":
        summary.append("Smoker")
        recommendations.append("Strongly advised to quit smoking; consider smoking cessation programs.")
    elif row['Smoking_Status'] == "Occasional":
        summary.append("Occasional Smoking")
        recommendations.append("Avoid smoking to prevent long-term health risks.")

    # Alcohol Consumption
    if row['Alcohol_Consumption'] == "Yes":
        summary.append("Regular Alcohol Consumption")
        recommendations.append("Limit intake to recommended levels; consider periodic detox.")
    elif row['Alcohol_Consumption'] == "Occasional":
        summary.append("Occasional Alcohol Intake")
        recommendations.append("Monitor alcohol consumption; ensure moderation.")

    return "; ".join(set(summary)), " | ".join(set(recommendations))