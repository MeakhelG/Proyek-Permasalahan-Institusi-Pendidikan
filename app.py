import datetime
import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler
import io

# File buffer for Excel download
buffer = io.BytesIO()

# Data preprocessing function
def data_preprocessing(data_input, single_data, n):
    df = pd.read_csv('student_data_filtered.csv')
    df = df.drop(columns=['Status'], axis=1)  # Drop Status column if present
    df = pd.concat([data_input, df])

    # Standardize the data
    df = StandardScaler().fit_transform(df)

    if single_data:
        return df[[n]]
    else:
        return df[0: n]

# Model prediction function
def model_predict(df):
    model = joblib.load('model_rf.joblib')  # Load your pre-trained model
    return model.predict(df)

# Function to color map the status (for visualization in Streamlit)
def color_mapping(value):
    color = 'green' if value == 'Graduate' else 'red'
    return f'color: {color}'

# Main Streamlit app function
def main():
    st.title('Jaya Jaya Institute Student Prediction')

    # Mappings for categorical data
    gender_mapping = {'Male': 1, 'Female': 0}
    marital_status_mapping = {
        'Single': 1, 'Married': 2, 'Widower': 3, 'Divorced': 4,
        'Facto Union': 5, 'Legally Separated': 6
    }

    application_mapping = {
        1: '1st Phase - General Contingent', 2: 'Ordinance No. 612/93', 5: '1st Phase - Special Contingent (Azores Island)',
        7: 'Holders of Other Higher Courses', 10: 'Ordinance No. 854-B/99', 15: 'International Student (Bachelor)',
        16: '1st Phase - Special Contingent (Madeira Island)', 17: '2nd Phase - General Contingent', 
        18: '3rd Phase - General Contingent', 26: 'Ordinance No. 533-A/99, Item B2 (Different Plan)', 
        27: 'Ordinance No. 533-A/99, Item B3 (Other Institution)', 39: 'Over 23 Years Old', 42: 'Transfer', 
        43: 'Change of Course', 44: 'Technological Specialization Diploma Holders', 51: 'Change of Institution/Course', 
        53: 'Short Cycle Diploma Holders', 57: 'Change of Institution/Course (International)'
    }

    course_mapping = {
        33: 'Biofuel Production Technologies', 171: 'Animation and Multimedia Design', 8014: 'Social Service (Evening Attendance)',
        9003: 'Agronomy', 9070: 'Communication Design', 9085: 'Veterinary Nursing', 9119: 'Informatics Engineering',
        9130: 'Equinculture', 9147: 'Management', 9238: 'Social Service', 9254: 'Tourism', 9500: 'Nursing', 
        9556: 'Oral Hygiene', 9670: 'Advertising and Marketing Management', 9773: 'Journalism and Communication', 
        9853: 'Basic Education', 9991: 'Management (Evening Attendance)'
    }

    # Seperate predictions for single data and multiple data
    tab_single, tab_multiple = st.tabs(['Single Data', 'Multiple Data'])

    # Prediction container for single data using input fields
    with tab_single:
        # Collect single data inputs
        with st.container():
            col_gender, col_age, col_marital = st.columns([2, 2, 3])
            with col_gender:
                gender = st.radio('Gender', options=['Male', 'Female'], help='The gender of the student')
            with col_age:
                age = st.number_input('Age at Enrollment', min_value=17, max_value=70, help='The age of the student at the time of enrollment')
            with col_marital:
                marital_status = st.selectbox('Marital Status', ('Single', 'Married', 'Widower', 'Divorced', 'Facto Union', 'Legally Separated'),
                    help='The marital status of the student')

        # Input other features for prediction
        st.write('')
        st.write('')
        with st.container():
            col_application, col_prev_grade, col_admission_grade = st.columns([3, 1.65, 1.1])
            with col_application:
                application_mode = st.selectbox('Application Mode', list(application_mapping.values()), help='The method of application used by the student')
            with col_prev_grade:
                prev_qualification_grade = st.number_input('Previous Qualification Grade', help='Grade of previous qualification (0-200)', min_value=0, max_value=200)
            with col_admission_grade:
                admission_grade = st.number_input('Admission Grade', help="Student's admission grade (0-200)", min_value=0, max_value=200)

        # Checkbox inputs for binary features
        with st.container():
            col_scholarship, col_tuition, col_displaced, col_debtor = st.columns([1.7, 2.1, 1.55, 1])
            with col_scholarship:
                scholarship_holder = 1 if st.checkbox('Scholarship', help='Whether the student is a scholarship holder') else 0
            with col_tuition:
                tuition_fees = 1 if st.checkbox('Tuition up to date', help="Whether the student's tuition fees are up to date") else 0
            with col_displaced:
                displaced = 1 if st.checkbox('Displaced', help='Whether the student is a displaced person') else 0
            with col_debtor:
                debtor = 1 if st.checkbox('Debtor', help='Whether the student is a debtor') else 0

        # Curricular units inputs
        st.write('')
        st.write('')
        with st.container():
            col_1_enroll, col_2_enroll, col_2_eval = st.columns([1, 1, 1.2])
            with col_1_enroll:
                curricular_units_1st_sem_enrolled = st.number_input('Units 1st Semester Enrolled', min_value=0, max_value=26, help='Number of curricular units enrolled by the student in the first semester')
            with col_2_enroll:
                curricular_units_2nd_sem_enrolled = st.number_input('Units 2nd Semester Enrolled', min_value=0, max_value=23, help='Number of curricular units enrolled by the student in the second semester')
            with col_2_eval:
                curricular_units_2nd_sem_evaluations = st.number_input('Units 2nd Semester Evaluations', min_value=0, max_value=33, help='Number of curricular units evaluations by the student in the second semester')

        # Feature data as dictionary
        data = [[
            marital_status, application_mapping.get(application_mode), prev_qualification_grade, admission_grade, displaced, debtor, 
            tuition_fees, gender_mapping.get(gender), scholarship_holder, age, curricular_units_1st_sem_enrolled, curricular_units_2nd_sem_enrolled, 
            curricular_units_2nd_sem_evaluations
        ]]

        # Convert the data to a dataframe for prediction
        df = pd.DataFrame(data, columns=[
            'Marital_status', 'Application_mode', 'Previous_qualification_grade', 'Admission_grade', 'Displaced', 'Debtor', 
            'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder', 'Age_at_enrollment', 
            'Curricular_units_1st_sem_enrolled', 'Curricular_units_2nd_sem_enrolled', 'Curricular_units_2nd_sem_evaluations'
        ])

        # Predict button for single data
        if st.button('✨ Predict'):
            data_input = data_preprocessing(df, True, 0)
            output = model_predict(data_input)
            prediction = 'Graduate' if output == 1 else 'Dropout'
            st.write(f'Prediction: {prediction}')

    # Prediction container for multiple data using file upload
    with tab_multiple:
        with st.expander('**User Guide**'):
            st.write("""
                1. Download the student data Excel template.
                2. Complete the data in the Excel file.
                3. Upload the student data file.
                4. Click the '✨ Predict Data' button.
                5. The results will be displayed in a table below.
                6. The results can be downloaded as an Excel file.
            """)
            with open('student_data_template.xlsx', 'rb') as file:
                st.download_button(label='Download Template', data=file, file_name='Student Data Template.xlsx')

        uploaded_file = st.file_uploader(label='Upload Student Data', type=['xlsx', 'xls'])

        if uploaded_file is not None:
            up = pd.read_excel(uploaded_file)
            up['ID'] = up['ID'].astype(str)

            # Preview the uploaded data
            preview = st.slider('**Preview Rows**', 1, len(up), 5)
            st.dataframe(up.head(preview))

            # Data preprocessing and prediction for multiple rows
            if st.button('✨ Predict Data'):
                # Preprocess the uploaded data
                df_up = data_preprocessing(up, False, len(up))
                output = model_predict(df_up)
                prediction = ['Graduate' if pred == 1 else 'Dropout' for pred in output]
                result = pd.DataFrame({
                    'ID': up['ID'], 'Name': up['Name'], 'Status': prediction
                })

                # Display the results
                st.write('**Results**')
                st.dataframe(result.style.applymap(color_mapping, subset=['Status']))

                # Prepare the Excel file for download
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    result.to_excel(writer, sheet_name='Prediction', index=False)

                # Download button for predictions
                st.download_button(
                    label='Download Prediction',
                    data=buffer.getvalue(),
                    file_name='Student_Data_Prediction.xlsx',
                    mime='application/vnd.ms-excel'
                )

# Run the app
if __name__ == '__main__':
    main()
