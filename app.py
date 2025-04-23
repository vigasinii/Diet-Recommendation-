import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="Diet Recommendation System",
    page_icon="🍎",
    layout="wide"
)

# Function to load data
@st.cache_data
def load_data():
    try:
        # Try the standard path first
        data = pd.read_csv("diet_recommendations_dataset.csv")
    except:
        try:
            # Try the Windows path if the first attempt fails
            data = pd.read_csv(r"C:\Users\Vigasini\Downloads\diet_recommendations_dataset.csv")
        except:
            # If both fail, show error message
            st.error("Unable to load dataset. Please make sure 'diet_recommendations_dataset.csv' is in the current directory.")
            data = None
    return data

# App title and description
st.title("🍎 Diet Recommendation System")
st.markdown("Enter your health metrics for a personalized diet plan.")

# Main application logic
def main():
    # Load data
    data = load_data()
    
    # Check if model exists, otherwise train it
    if not os.path.exists('diet_model.pkl') or not os.path.exists('preprocessor.pkl'):
        with st.spinner("Initializing model for first-time use..."):
            train_model()
        st.success("Model initialized successfully!")
    
    # Load model and preprocessor
    model = pickle.load(open('diet_model.pkl', 'rb'))
    preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
    
    # Create form for user input
    with st.form("prediction_form"):
        st.subheader("Your Health Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=80, value=40)
            gender = st.selectbox("Gender", options=["Male", "Female"])
            weight = st.number_input("Weight (kg)", min_value=40.0, max_value=150.0, value=70.0, step=0.1)
            height = st.number_input("Height (cm)", min_value=140, max_value=220, value=170)
            disease_type = st.selectbox("Medical Condition", options=["None", "Obesity", "Diabetes", "Hypertension"])
            
        with col2:
            physical_activity = st.selectbox("Physical Activity Level", options=["Sedentary", "Moderate", "Active"])
            daily_calories = st.number_input("Daily Caloric Intake", min_value=1500, max_value=3500, value=2000, step=50)
            cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=150.0, max_value=250.0, value=180.0, step=0.1)
            blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=110, max_value=180, value=120, step=1)
            glucose = st.number_input("Glucose (mg/dL)", min_value=70.0, max_value=200.0, value=100.0, step=0.1)
            
        with col3:
            dietary_restriction = st.selectbox("Dietary Restrictions", options=["None", "Low_Sugar", "Low_Sodium"])
            allergies = st.selectbox("Allergies", options=["None", "Peanuts", "Gluten"])
            preferred_cuisine = st.selectbox("Preferred Cuisine", options=["Italian", "Chinese", "Mexican", "Indian"])
            weekly_exercise = st.number_input("Weekly Exercise Hours", min_value=0.0, max_value=10.0, value=3.0, step=0.5)
            severity = st.selectbox("Condition Severity (if applicable)", options=["Mild", "Moderate", "Severe"])
            
        # Hidden fields (used by the model but not shown to user)
        adherence = 0.7  # Default value
        nutrient_imbalance = 0.3  # Default value
        
        # Calculate BMI
        bmi = weight / ((height/100) ** 2)
        
        submit_button = st.form_submit_button("Get Diet Recommendation")
        
        if submit_button:
            # Create input dataframe
            input_data = pd.DataFrame({
                'Age': [age],
                'Gender': [gender],
                'Weight_kg': [weight],
                'Height_cm': [height],
                'BMI': [bmi],
                'Disease_Type': [disease_type],
                'Severity': [severity],
                'Physical_Activity_Level': [physical_activity],
                'Daily_Caloric_Intake': [daily_calories],
                'Cholesterol_mg/dL': [cholesterol],
                'Blood_Pressure_mmHg': [blood_pressure],
                'Glucose_mg/dL': [glucose],
                'Dietary_Restrictions': [dietary_restriction],
                'Allergies': [allergies],
                'Preferred_Cuisine': [preferred_cuisine],
                'Weekly_Exercise_Hours': [weekly_exercise],
                'Adherence_to_Diet_Plan': [adherence],
                'Dietary_Nutrient_Imbalance_Score': [nutrient_imbalance]
            })
            
            # Preprocess input data
            processed_data = preprocessor.transform(input_data)
            
            # Make prediction
            base_prediction = model.predict(processed_data)[0]
            
            # Override prediction based on health conditions
            prediction = base_prediction
            
            # BMI category
            if bmi < 18.5:
                bmi_category = "Underweight"
            elif bmi >= 18.5 and bmi < 25:
                bmi_category = "Normal weight"
            elif bmi >= 25 and bmi < 30:
                bmi_category = "Overweight"
            else:
                bmi_category = "Obese"
            
            # Override rules based on health metrics
            if bmi >= 30 and base_prediction == "Balanced" and disease_type != "Diabetes":
                prediction = "Low_Carb"
            if disease_type == "Hypertension" and blood_pressure > 140:
                prediction = "Low_Sodium"
            if disease_type == "Diabetes" and glucose > 150 and base_prediction == "Balanced":
                prediction = "Low_Carb"
            
            # Display results in a clean, focused way
            st.header("Your Personalized Diet Plan")
            
            # Show BMI info in a small info box
            st.info(f"BMI: {bmi:.1f} ({bmi_category})")
            
            # Create tabs for diet recommendations
            tab1, tab2 = st.tabs(["Main Recommendation", "Alternative Diets"])
            
            with tab1:
                # Main recommendation
                if prediction == "Balanced":
                    st.subheader("✅ Balanced Diet")
                    
                    st.markdown("### Daily Nutrition Breakdown")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Protein", "25%", "Lean meats, fish, eggs, legumes")
                    with col2:
                        st.metric("Carbs", "45-50%", "Whole grains, fruits, vegetables")
                    with col3:
                        st.metric("Fats", "25-30%", "Olive oil, avocados, nuts")
                        
                    st.markdown("### Recommended Foods")
                    
                    # Recommendations based on user's profile
                    recommendations = []
                    
                    # Add personalized recommendations
                    if bmi > 30:
                        recommendations.append("Focus on portion control while maintaining nutrient density")
                    elif bmi < 18.5:
                        recommendations.append("Increase calorie intake with nutrient-dense foods")
                    
                    if disease_type == "Diabetes":
                        recommendations.append("Choose low glycemic index foods and maintain consistent carbohydrate intake")
                    elif disease_type == "Hypertension":
                        recommendations.append("Limit sodium to under 2,000mg per day")
                    
                    if cholesterol > 200:
                        recommendations.append("Limit saturated fats and increase soluble fiber intake")
                    
                    # Add cuisine-specific recommendations
                    if preferred_cuisine == "Italian":
                        st.write("🍝 **Italian-inspired balanced meals:**")
                        st.write("- Whole grain pasta with tomato sauce, vegetables and a small amount of lean protein")
                        st.write("- Minestrone soup with beans and vegetables")
                        st.write("- Grilled fish with roasted vegetables and olive oil")
                    elif preferred_cuisine == "Chinese":
                        st.write("🥢 **Chinese-inspired balanced meals:**")
                        st.write("- Stir-fried vegetables with small amount of lean protein and brown rice")
                        st.write("- Steamed fish with ginger and steamed vegetables")
                        st.write("- Buddha's delight vegetable dish with tofu")
                    elif preferred_cuisine == "Mexican":
                        st.write("🌮 **Mexican-inspired balanced meals:**")
                        st.write("- Bean burritos with whole grain tortillas and plenty of vegetables")
                        st.write("- Fish tacos with cabbage slaw and avocado")
                        st.write("- Vegetable and chicken fajitas with minimal oil")
                    elif preferred_cuisine == "Indian":
                        st.write("🍛 **Indian-inspired balanced meals:**")
                        st.write("- Dal (lentil curry) with brown rice and vegetable sides")
                        st.write("- Tandoori chicken with raita and vegetable curry")
                        st.write("- Chickpea curry with whole grain roti")
                    
                    # Diet restrictions and allergies
                    if len(recommendations) > 0:
                        st.markdown("### Special Considerations")
                        for rec in recommendations:
                            st.write(f"- {rec}")
                    
                    # Sample meal plan
                    st.markdown("### Sample Daily Meal Plan")
                    meal_col1, meal_col2 = st.columns(2)
                    with meal_col1:
                        st.write("**Breakfast:** Whole grain toast with avocado and eggs")
                        st.write("**Lunch:** Grilled chicken salad with mixed vegetables and olive oil dressing")
                    with meal_col2:
                        st.write("**Dinner:** Baked fish with roasted vegetables and quinoa")
                        st.write("**Snacks:** Greek yogurt with berries, handful of nuts, or fresh fruit")
                
                elif prediction == "Low_Carb":
                    st.subheader("✅ Low-Carb Diet")
                    
                    st.markdown("### Daily Nutrition Breakdown")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Protein", "30-35%", "Eggs, fish, poultry, meat, tofu")
                    with col2:
                        st.metric("Carbs", "15-30%", "Non-starchy vegetables, limited fruits")
                    with col3:
                        st.metric("Fats", "40-60%", "Avocados, olive oil, nuts, seeds")
                    
                    st.markdown("### Recommended Foods")
                    
                    # Personalized recommendations
                    recommendations = []
                    
                    if disease_type == "Diabetes":
                        recommendations.append("Monitor blood glucose regularly and adjust carb intake accordingly")
                    
                    if cholesterol > 200:
                        recommendations.append("Choose heart-healthy unsaturated fats over saturated fats")
                    
                    # Add cuisine-specific recommendations
                    if preferred_cuisine == "Italian":
                        st.write("🍝 **Italian-inspired low-carb meals:**")
                        st.write("- Chicken cacciatore with side of roasted vegetables")
                        st.write("- Zucchini noodles with bolognese sauce")
                        st.write("- Italian antipasto salad with cured meats, cheese, and vegetables")
                    elif preferred_cuisine == "Chinese":
                        st.write("🥢 **Chinese-inspired low-carb meals:**")
                        st.write("- Egg drop soup with added vegetables")
                        st.write("- Lettuce wraps with stir-fried meat and vegetables")
                        st.write("- Steamed fish with ginger and spring onions")
                    elif preferred_cuisine == "Mexican":
                        st.write("🌮 **Mexican-inspired low-carb meals:**")
                        st.write("- Taco salad with ground beef, avocado, and vegetables (no shell)")
                        st.write("- Fajita filling (chicken/beef with peppers and onions) without the tortilla")
                        st.write("- Ceviche with avocado")
                    elif preferred_cuisine == "Indian":
                        st.write("🍛 **Indian-inspired low-carb meals:**")
                        st.write("- Tandoori chicken or fish with spiced cauliflower rice")
                        st.write("- Paneer tikka with vegetable curry (no rice)")
                        st.write("- Egg curry with side of sautéed spinach")
                    
                    # Diet restrictions and allergies
                    if len(recommendations) > 0:
                        st.markdown("### Special Considerations")
                        for rec in recommendations:
                            st.write(f"- {rec}")
                    
                    # Sample meal plan
                    st.markdown("### Sample Daily Meal Plan")
                    meal_col1, meal_col2 = st.columns(2)
                    with meal_col1:
                        st.write("**Breakfast:** Omelet with spinach, cheese, and avocado")
                        st.write("**Lunch:** Large salad with grilled chicken, olive oil dressing")
                    with meal_col2:
                        st.write("**Dinner:** Baked salmon with asparagus and cauliflower mash")
                        st.write("**Snacks:** Hard-boiled eggs, cheese, nuts, or celery with almond butter")
                
                else:  # Low_Sodium
                    st.subheader("✅ Low-Sodium Diet")
                    
                    st.markdown("### Daily Nutrition Breakdown")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sodium", "< 2,300mg/day", "Fresh foods, herbs for flavor")
                    with col2:
                        st.metric("Potassium", "High intake", "Fruits, vegetables, legumes")
                    with col3:
                        st.metric("Balanced Macros", "Standard", "Fresh, unprocessed foods")
                    
                    st.markdown("### Recommended Foods")
                    
                    # Personalized recommendations
                    recommendations = []
                    
                    if disease_type == "Hypertension":
                        recommendations.append("Aim for the lower sodium limit (1,500mg daily)")
                    
                    if blood_pressure > 140:
                        recommendations.append("Combine with DASH diet principles for better blood pressure management")
                    
                    # Add cuisine-specific recommendations
                    if preferred_cuisine == "Italian":
                        st.write("🍝 **Italian-inspired low-sodium meals:**")
                        st.write("- Homemade pasta with fresh tomato sauce (no added salt)")
                        st.write("- Grilled vegetables with fresh herbs and a small amount of olive oil")
                        st.write("- Fresh fish baked with lemon, herbs, and garlic")
                    elif preferred_cuisine == "Chinese":
                        st.write("🥢 **Chinese-inspired low-sodium meals:**")
                        st.write("- Steamed vegetables and chicken with ginger and garlic (no soy sauce)")
                        st.write("- Congee with lean protein and vegetables")
                        st.write("- Stir-fried vegetables with minimal low-sodium sauce")
                    elif preferred_cuisine == "Mexican":
                        st.write("🌮 **Mexican-inspired low-sodium meals:**")
                        st.write("- Fresh salsa with no added salt, used as topping for grilled chicken")
                        st.write("- Black bean and vegetable soft tacos with fresh herbs")
                        st.write("- Grilled fish with lime and cilantro")
                    elif preferred_cuisine == "Indian":
                        st.write("🍛 **Indian-inspired low-sodium meals:**")
                        st.write("- Vegetable curry with plenty of spices but no added salt")
                        st.write("- Tandoori chicken marinated in yogurt and spices (no salt)")
                        st.write("- Cumin-scented rice with vegetables (minimal salt)")
                    
                    # Diet restrictions and allergies
                    if len(recommendations) > 0:
                        st.markdown("### Special Considerations")
                        for rec in recommendations:
                            st.write(f"- {rec}")
                    
                    # Sample meal plan
                    st.markdown("### Sample Daily Meal Plan")
                    meal_col1, meal_col2 = st.columns(2)
                    with meal_col1:
                        st.write("**Breakfast:** Oatmeal with fresh fruit and unsalted nuts")
                        st.write("**Lunch:** Homemade soup with vegetables and chicken (no added salt)")
                    with meal_col2:
                        st.write("**Dinner:** Grilled fish with herbs, fresh vegetables, and brown rice")
                        st.write("**Snacks:** Fresh fruit, unsalted popcorn, or yogurt")
            
            with tab2:
                # Determine appropriate alternative diets
                alternative_diets = []
                
                # Logic for suggesting alternatives
                if prediction != "Low_Carb" and (bmi > 28 or glucose > 110):
                    alternative_diets.append("Low-Carb")
                
                if prediction != "Low_Sodium" and (blood_pressure > 130 or disease_type == "Hypertension"):
                    alternative_diets.append("Low-Sodium")
                
                if cholesterol > 200 or blood_pressure > 130:
                    alternative_diets.append("Mediterranean")
                
                if blood_pressure > 140:
                    alternative_diets.append("DASH")
                
                if bmi > 30 and disease_type != "Diabetes" and glucose < 120:
                    alternative_diets.append("Ketogenic")
                
                if len(alternative_diets) > 0:
                    st.write("Based on your health profile, these alternative diets may also be beneficial:")
                    
                    for diet in alternative_diets:
                        if diet == "Low-Carb" and prediction != "Low-Carb":
                            st.markdown("### Low-Carb Diet")
                            st.write("**Focus on:** Proteins, healthy fats, and non-starchy vegetables")
                            st.write("**Limit:** Grains, starchy vegetables, sugars, and processed carbs")
                            st.write("**Best for:** Weight management and blood sugar control")
                        
                        elif diet == "Low-Sodium" and prediction != "Low-Sodium":
                            st.markdown("### Low-Sodium Diet")
                            st.write("**Focus on:** Fresh foods, herbs and spices for flavor")
                            st.write("**Limit:** Processed foods, added salt, and high-sodium condiments")
                            st.write("**Best for:** Blood pressure management and heart health")
                        
                        elif diet == "Mediterranean":
                            st.markdown("### Mediterranean Diet")
                            st.write("**Focus on:** Olive oil, vegetables, fruits, whole grains, fish, and nuts")
                            st.write("**Limit:** Red meat, processed foods, and added sugars")
                            st.write("**Best for:** Heart health, longevity, and reduced inflammation")
                        
                        elif diet == "DASH":
                            st.markdown("### DASH Diet")
                            st.write("**Focus on:** Fruits, vegetables, whole grains, lean proteins, and low-fat dairy")
                            st.write("**Limit:** Sodium, sweets, and foods high in saturated fats")
                            st.write("**Best for:** Lowering blood pressure and improving heart health")
                        
                        elif diet == "Ketogenic":
                            st.markdown("### Ketogenic Diet")
                            st.write("**Focus on:** High fat (70-80%), moderate protein (15-20%), very low carb (5-10%)")
                            st.write("**Limit:** Almost all carbohydrates, including fruits, grains, and starchy vegetables")
                            st.write("**Best for:** Significant weight loss and certain metabolic conditions")
                            st.write("*Note: Should be implemented under healthcare supervision*")
                else:
                    st.write("Your primary diet recommendation is the most suitable for your health profile.")

# Function to train the model
def train_model():
    # Load data
    data = load_data()
    
    # Define features and target
    X = data.drop(['Patient_ID', 'Diet_Recommendation'], axis=1)
    y = data['Diet_Recommendation']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define categorical and numerical features
    categorical_features = ['Gender', 'Disease_Type', 'Severity', 'Physical_Activity_Level', 
                          'Dietary_Restrictions', 'Allergies', 'Preferred_Cuisine']
    numerical_features = ['Age', 'Weight_kg', 'Height_cm', 'BMI', 'Daily_Caloric_Intake', 
                        'Cholesterol_mg/dL', 'Blood_Pressure_mmHg', 'Glucose_mg/dL', 
                        'Weekly_Exercise_Hours', 'Adherence_to_Diet_Plan', 'Dietary_Nutrient_Imbalance_Score']
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Define the model pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Save preprocessor and model
    pickle.dump(model['preprocessor'], open('preprocessor.pkl', 'wb'))
    pickle.dump(model['classifier'], open('diet_model.pkl', 'wb'))
    
    return model

if __name__ == "__main__":
    main()