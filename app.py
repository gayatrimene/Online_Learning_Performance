import pickle
from flask import Flask, request, render_template
import numpy as np

# Initialize Flask App
app = Flask(__name__)

# Load the pre-trained Random Forest model
model_path = 'Online_Learning_Performance_RF.pkl'  # Ensure this path is correct
model = pickle.load(open(model_path, 'rb'))

# Define home route
@app.route('/')
def home():
    return render_template('index.html')

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract and log form data
        gender = request.form.get('gender')
        nationality = request.form.get('nationality')
        place_of_birth = request.form.get('place_of_birth')
        stage_id = request.form.get('stage_id')
        grade_id = request.form.get('grade_id')
        section_id = request.form.get('section_id')
        topic = request.form.get('topic')
        semester = request.form.get('semester')
        relation = request.form.get('relation')
        raised_hands = float(request.form.get('raised_hands'))
        visited_resources = float(request.form.get('visited_resources'))
        announcements_view = float(request.form.get('announcements_view'))
        discussion_participation = float(request.form.get('discussion'))
        parent_answering_survey = request.form.get('parent_answering_survey')
        parent_school_satisfaction = request.form.get('parent_school_satisfaction')
        student_absence_days = request.form.get('student_absence_days')

        print("Form data received:", request.form)

        # Encode features
        gender_encoded = 1 if gender == 'M' else 0
        semester_encoded = 1 if semester == 'F' else 0
        relation_encoded = 1 if relation == 'Father' else 0
        parent_answering_survey_encoded = 1 if parent_answering_survey == 'Yes' else 0
        parent_school_satisfaction_encoded = 1 if parent_school_satisfaction == 'Good' else 0
        student_absence_days_encoded = 1 if student_absence_days == 'Under-7' else 0
        stage_id_encoded = 1 if stage_id == 'lower level' else 0
        grade_id_encoded = 1 if grade_id == 'G-04' else 0
        section_id_encoded = 1 if section_id == 'A' else 0
        topic_encoded = 1 if topic == 'IT' else 0
        nationality_encoded = 1 if nationality == 'KW' else 0
        place_of_birth_encoded = 1 if place_of_birth == 'KuwaIT' else 0


        # Prepare feature array
        features = [
            gender_encoded, raised_hands, visited_resources, announcements_view,
            discussion_participation, parent_answering_survey_encoded,
            parent_school_satisfaction_encoded, student_absence_days_encoded
        ]
        features.extend([stage_id_encoded, grade_id_encoded, section_id_encoded, topic_encoded, semester_encoded, relation_encoded, nationality_encoded, place_of_birth_encoded])

        features_array = np.array(features).reshape(1, -1)
        print("Features array:", features_array)

        # Predict
        prediction = model.predict(features_array)
        print("Model prediction:", prediction)

        output_mapping = {'L': 'Low', 'M': 'Medium', 'H': 'High'}
        output = output_mapping.get(prediction[0], "Unknown")
        print("Mapped output:", output)

        return render_template('index.html', prediction_text=f'Predicted Performance: {output}')
    except Exception as e:
        print("Error during prediction:", str(e))
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)