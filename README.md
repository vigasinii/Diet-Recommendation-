# Diet-Recommendation-
# Diet Recommendation System

A machine learning-powered Streamlit application that provides personalized diet recommendations based on individual health metrics and preferences.

![Diet Recommendation System](https://raw.githubusercontent.com/yourusername/diet-recommendation-system/main/screenshots/app_preview.png)

## Features

- **Personalized Diet Plans**: Get customized diet recommendations based on your health profile
- **Health Metrics Analysis**: Input your physical measurements, medical conditions, and lifestyle factors
- **Cuisine Preferences**: Diet plans incorporate your preferred cuisine types
- **Alternative Diet Suggestions**: View alternative diet options that may benefit your health profile
- **Custom Meal Examples**: See meal suggestions tailored to your recommended diet
- **Machine Learning Backend**: Recommendations driven by a Random Forest classifier trained on dietary data

## Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/diet-recommendation-system.git
cd diet-recommendation-system
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Place the dataset file in the project directory:
- Ensure that `diet_recommendations_dataset.csv` is in the project root

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

3. Enter your health information in the form:
   - Basic metrics (age, gender, weight, height)
   - Medical conditions
   - Activity level
   - Dietary preferences
   - Health goals

4. Click "Get Diet Recommendation" to receive your personalized diet plan

## Multi-Page Structure

You can implement a multi-page structure within a single file using Streamlit's session state for navigation:

```python
# Add this to the beginning of your script
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Create sidebar for navigation
st.sidebar.title("Navigation")
if st.sidebar.button("Home"):
    st.session_state.page = 'home'
if st.sidebar.button("Health Profile"):
    st.session_state.page = 'profile'
if st.sidebar.button("Diet Recommendations"):
    st.session_state.page = 'diet'
# Add more pages as needed

# Display the appropriate page based on the current state
if st.session_state.page == 'home':
    show_home_page()
elif st.session_state.page == 'profile':
    show_profile_page()
# Add more page conditions
```

## How It Works

1. **Data Collection**: The application collects user health metrics through an intuitive form interface
2. **Data Preprocessing**: Input data is standardized and encoded for the machine learning model
3. **Prediction**: A Random Forest classifier predicts the most suitable diet type
4. **Customization**: Rule-based systems refine the recommendation based on specific health conditions
5. **Presentation**: Results are displayed with detailed nutritional information and meal suggestions

## Diet Types

The system recommends one of the following diet types:

- **Balanced Diet**: Equal proportions of proteins, carbohydrates, and fats
- **Low-Carb Diet**: Higher protein and fat, reduced carbohydrate intake
- **Low-Sodium Diet**: Reduced salt intake for better blood pressure management

## Model Training

The application uses a Random Forest classifier trained on dietary data. If the trained model is not found, the system automatically trains a new model during first-time use.

```python
# Define the model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X_train, y_train)
```

## Dataset Structure

The application requires a dataset with the following columns:
- Patient_ID
- Age
- Gender
- Weight_kg
- Height_cm
- BMI
- Disease_Type
- Severity
- Physical_Activity_Level
- Daily_Caloric_Intake
- Cholesterol_mg/dL
- Blood_Pressure_mmHg
- Glucose_mg/dL
- Dietary_Restrictions
- Allergies
- Preferred_Cuisine
- Weekly_Exercise_Hours
- Adherence_to_Diet_Plan
- Dietary_Nutrient_Imbalance_Score
- Diet_Recommendation

## Extending the Application

You can extend this application by:
1. Adding more diet types to the recommendation system
2. Implementing a meal planning feature with specific recipes
3. Adding progress tracking to monitor health improvements
4. Creating a knowledge base for nutritional information
5. Building a community feature for sharing experiences

## Dependencies

- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- pickle

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset source: [provide source if applicable]
- Streamlit for the wonderful web app framework
- scikit-learn for machine learning tools

## Contact

For questions or suggestions, please contact [your email or contact information].
