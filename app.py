import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import io

# Load the pre-trained model
model = joblib.load('model_rf.joblib')

# Mapping
application_mode_mapping = {
    1: '1st Phase - General Contingent',
    2: 'Ordinance No. 612/93',
    5: '1st Phase - Special Contingent (Azores Island)',
    7: 'Holders of Other Higher Courses',
    10: 'Ordinance No. 854-B/99',
    15: 'International Student (Bachelor)',
    16: '1st Phase - Special Contingent (Madeira Island)',
    17: '2nd Phase - General Contingent',
    18: '3rd Phase - General Contingent',
    26: 'Ordinance No. 533-A/99, Item B2 (Different Plan)',
    27: 'Ordinance No. 533-A/99, Item B3 (Other Institution)',
    39: 'Over 23 Years Old',
    42: 'Transfer',
    43: 'Change of Course',
    44: 'Technological Specialization Diploma Holders',
    51: 'Change of Institution/Course',
    53: 'Short Cycle Diploma Holders',
    57: 'Change of Institution/Course (International)'
}

course_mapping = {
    33: 'Biofuel Production Technologies',
    171: 'Animation and Multimedia Design',
    8014: 'Social Service (Evening Attendance)',
    9003: 'Agronomy',
    9070: 'Communication Design',
    9085: 'Veterinary Nursing',
    9119: 'Informatics Engineering',
    9130: 'Equinculture',
    9147: 'Management',
    9238: 'Social Service',
    9254: 'Tourism',
    9500: 'Nursing',
    9556: 'Oral Hygiene',
    9670: 'Advertising and Marketing Management',
    9773: 'Journalism and Communication',
    9853: 'Basic Education',
    9991: 'Management (Evening Attendance)'
}

gender_mapping = {'Male': 1, 'Female': 0}
application_mode_reverse = {v: k for k, v in application_mode_mapping.items()}
course_reverse = {v: k for k, v in course_mapping.items()}

def preprocess_data(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled

def make_prediction(data):
    prediction = model.predict(data)
    return ['Graduate' if p == 1 else 'Dropout' for p in prediction]

st.title("🎓 Aplikasi Prediksi Status Mahasiswa")

tab1, tab2 = st.tabs(["Prediksi Satu Mahasiswa", "Prediksi Banyak Mahasiswa"])

with tab1:
    st.header("Input Data Mahasiswa")

    prev_qualification_grade = st.number_input("Previous Qualification Grade", 0, 200)
    admission_grade = st.number_input("Admission Grade", 0, 200)
    tuition_fees_up_to_date = 1 if st.checkbox("Tuition Fees Up To Date") else 0
    age_at_enrollment = st.number_input("Age at Enrollment", 15, 100)
    curricular_units_1st_sem_evaluations = st.number_input("1st Sem Evaluations", 0)
    curricular_units_1st_sem_approved = st.number_input("1st Sem Approved", 0)
    curricular_units_1st_sem_grade = st.number_input("1st Sem Grade", 0.0, 20.0)
    curricular_units_2nd_sem_evaluations = st.number_input("2nd Sem Evaluations", 0)
    curricular_units_2nd_sem_approved = st.number_input("2nd Sem Approved", 0)
    curricular_units_2nd_sem_grade = st.number_input("2nd Sem Grade", 0.0, 20.0)
    scholarship_holder = 1 if st.checkbox("Scholarship Holder") else 0
    debtor = 1 if st.checkbox("Debtor") else 0
    displaced = 1 if st.checkbox("Displaced") else 0
    gender = st.selectbox("Gender", list(gender_mapping.keys()))
    application_mode = st.selectbox("Application Mode", list(application_mode_mapping.values()))
    course = st.selectbox("Course", list(course_mapping.values()))

    input_data = pd.DataFrame([[
        prev_qualification_grade,
        admission_grade,
        tuition_fees_up_to_date,
        age_at_enrollment,
        curricular_units_1st_sem_evaluations,
        curricular_units_1st_sem_approved,
        curricular_units_1st_sem_grade,
        curricular_units_2nd_sem_evaluations,
        curricular_units_2nd_sem_approved,
        curricular_units_2nd_sem_grade,
        scholarship_holder,
        debtor,
        displaced,
        gender_mapping[gender],
        application_mode_reverse[application_mode],
        course_reverse[course]
    ]], columns=[
        'Previous_qualification_grade',
        'Admission_grade',
        'Tuition_fees_up_to_date',
        'Age_at_enrollment',
        'Curricular_units_1st_sem_evaluations',
        'Curricular_units_1st_sem_approved',
        'Curricular_units_1st_sem_grade',
        'Curricular_units_2nd_sem_evaluations',
        'Curricular_units_2nd_sem_approved',
        'Curricular_units_2nd_sem_grade',
        'Scholarship_holder',
        'Debtor',
        'Displaced',
        'Gender',
        'Application_mode',
        'Course'
    ])

    if st.button("Prediksi"):
        processed_data = preprocess_data(input_data)
        result = make_prediction(processed_data)
        st.success(f"Status Mahasiswa: **{result[0]}**")

with tab2:
    st.header("Upload Data Mahasiswa")

    uploaded_file = st.file_uploader("Unggah file Excel", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)

        df['Gender'] = df['Gender'].map(gender_mapping)
        df['Application_mode'] = df['Application_mode'].map(application_mode_reverse)
        df['Course'] = df['Course'].map(course_reverse)

        for binary in ['Tuition_fees_up_to_date', 'Scholarship_holder', 'Debtor', 'Displaced']:
            df[binary] = df[binary].apply(lambda x: 1 if x == 1 else 0)

        features = [
            'Previous_qualification_grade',
            'Admission_grade',
            'Tuition_fees_up_to_date',
            'Age_at_enrollment',
            'Curricular_units_1st_sem_evaluations',
            'Curricular_units_1st_sem_approved',
            'Curricular_units_1st_sem_grade',
            'Curricular_units_2nd_sem_evaluations',
            'Curricular_units_2nd_sem_approved',
            'Curricular_units_2nd_sem_grade',
            'Scholarship_holder',
            'Debtor',
            'Displaced',
            'Gender',
            'Application_mode',
            'Course'
        ]

        input_data = df[features]

        if st.button("Prediksi Batch"):
            processed = preprocess_data(input_data)
            pred = make_prediction(processed)
            df['Predicted_Status'] = pred
            st.dataframe(df)

            buffer = io.BytesIO()
            df.to_excel(buffer, index=False, sheet_name="Predictions")
            buffer.seek(0)

            st.download_button(
                label="📥 Unduh Hasil Prediksi",
                data=buffer,
                file_name="prediksi_mahasiswa.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
