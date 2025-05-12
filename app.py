import streamlit as st
import pandas as pd
import joblib
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# -------------------- Fungsi Encoding --------------------
def encoding(dataset):
    # Fitur kategorikal & numerikal
    numerical_features = ['Age_at_enrollment', 'Displaced', 'Previous_qualification_grade', 'Admission_grade', 
        'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade', 'Curricular_units_1st_sem_evaluations',
        'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_evaluations',
        'Tuition_fees_up_to_date', 'Debtor']
    categorical_features = ['Gender', 'Application_mode', 'Course']

    df = pd.read_csv('./dataset/dataset_predict.csv')
    df = df.drop(columns=['Status_Binary'], axis=1)
    df = pd.concat([dataset, df])

    # Pipeline untuk preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ],
        sparse_threshold=0  # <-- paksa output menjadi dense
    )

    # Transformasi fit dan transform
    X_processed = preprocessor.fit_transform(df)

    # Ambil nama kolom hasil encoding
    ohe_columns = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
 
    # Gabungkan kolom numerik dan kolom hasil encoding
    feature_names = numerical_features + ohe_columns.tolist()

    # Jadikan DataFrame
    X_encoded = pd.DataFrame(X_processed, columns=feature_names)

    return X_encoded

# -------------------- Fungsi Prediksi --------------------
def predict(model, X_encoded):
    # Selected features
    selected_features = ['Previous_qualification_grade',
                         'Admission_grade',
                         'Tuition_fees_up_to_date',
                         'Age_at_enrollment',
                         'Curricular_units_1st_sem_evaluations',
                         'Curricular_units_1st_sem_approved',
                         'Curricular_units_1st_sem_grade',
                         'Curricular_units_2nd_sem_evaluations',
                         'Curricular_units_2nd_sem_approved',
                         'Curricular_units_2nd_sem_grade']

    # Pilih hanya fitur yang terpilih untuk prediksi
    X_selected = X_encoded[selected_features]

    # Prediksi dengan model
    y_pred_rfc = model.predict(X_selected)

    return y_pred_rfc[0]

def main():
    # SIDEBAR
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/7/77/Streamlit-logo-primary-colormark-darktext.png")
        url = "https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance"
        link_text = "Klik link untuk mengunduh dataset"
        st.write('Submission Akhir: Menyelesaikan Permasalahan Institusi Pendidikan')
        st.write(f"[{link_text}]({url})")
        st.write('Copyright (C) 2025 by Meakhel Gunawan')
    
    # JUDUL
    st.title('Prediksi Mahasiswa dari Jaya Jaya Institut')

    # DATA DEMOGRAFIS
    st.markdown("---")
    st.markdown("### Data Demografis")
    col1, col2, col3 = st.columns([3, 1.5, 1.5])
    with col1:
        Age_at_enrollment = st.number_input(label='Umur saat Enrollment', value=20,
            help='Umur mahasiswa ketika mengambil kelas')
    with col2:
        Gender = st.radio('Jenis Kelamin', options=['Laki-laki', 'Perempuan'],
                    help='Jenis kelamin mahasiswa')
    with col3:
        Displaced = 1 if st.checkbox(
            'Displaced', help='Apakah mahasiswa terlantar atau tidak') else 0            

    # DATA LATAR BELAKANG PENDIDIKAN
    st.markdown("---")
    st.markdown("### Data Latar Belakang Pendidikan")
    col1, col2= st.columns([4, 4])
    with col1:
        Application_mode = st.selectbox('Application Mode', (
            '1st Phase - General Contingent',
            '1st Phase - Special Contingent (Azores Island)',
            '1st Phase - Special Contingent (Madeira Island)',
            '2nd Phase - General Contingent', '3rd Phase - General Contingent',
            'Ordinance No. 612/93', 'Ordinance No. 854-B/99',
            'Ordinance No. 533-A/99, Item B2 (Different Plan)',
            'Ordinance No. 533-A/99, Item B3 (Other Institution)',
            'International Student (Bachelor)', 'Over 23 Years Old',
            'Transfer', 'Change of Course', 'Holders of Other Higher Courses',
            'Short Cycle Diploma Holders',
            'Technological Specialization Diploma Holders',
            'Change of Institution/Course',
            'Change of Institution/Course (International)'),
            help='Metode aplikasi yang dipakai mahasiswa')
    with col2:
        Course = st.selectbox('Course', (
            'Biofuel Production Technologies',
            'Animation and Multimedia Design',
            'Social Service (Evening Attendance)',
            'Agronomy',
            'Communication Design',
            'Veterinary Nursing',
            'Informatics Engineering',
            'Equinculture',
            'Management',
            'Social Service',
            'Tourism',
            'Nursing',
            'Oral Hygiene',
            'Advertising and Marketing Management',
            'Journalism and Communication',
            'Basic Education',
            'Management (Evening Attendance)'),
            help='Kelas yang diambil mahasiswa')
    col1, col2= st.columns([4, 4])
    with col1:
        Previous_qualification_grade = st.number_input(label='Previous Qualification Grade', value=80, min_value=0, max_value=200, 
            help='Nilai kualifikasi mahasiswa sebelumnya')
    with col2:
        Admission_grade = st.number_input(label='Admission Grade', value=80, min_value=0, max_value=200, 
            help='Nilai admission mahasiswa')

    # DATA KIINERJA AKADEMIK
    st.markdown("---")
    st.markdown("### Data Kinerja Akademik")
    col1, col2, col3= st.columns([4, 4, 4])
    with col1:
        Curricular_units_1st_sem_approved = st.number_input(label='Curricular Units 1st Sem Approved', value=0, min_value=0, 
            help='Unit kurikulum semester 1 yang di approve')
    with col2:
        Curricular_units_1st_sem_grade = st.number_input(label='Curricular Units 1st Sem Grade', value=0, min_value=0, 
            help='Nilai unit kurikulum semester 1')
    with col3:
        Curricular_units_1st_sem_evaluations = st.number_input(label='Curricular Units 1st Sem Eval', value=0, min_value=0, 
            help='Evauasi unit kurikulum semester 1')
    col1, col2, col3= st.columns([4, 4, 4])
    with col1:
        Curricular_units_2nd_sem_approved = st.number_input(label='Curricular Units 2nd Sem Approved', value=0, min_value=0, 
            help='Unit kurikulum semester 2 yang di approve')
    with col2:
        Curricular_units_2nd_sem_grade = st.number_input(label='Curricular Units 2nd Sem Grade', value=0, min_value=0, 
            help='Nilai unit kurikulum semester 2')
    with col3:
        Curricular_units_2nd_sem_evaluations = st.number_input(label='Curricular Units 2nd Sem Eval', value=0, min_value=0, 
            help='Evauasi unit kurikulum semester 2')

    # DATA KONDISI EKONOMI
    st.markdown("---")
    st.markdown("### Data Kondisi Ekonomi")
    col1, col2= st.columns([4, 4])
    with col1:
        Tuition_fees_up_to_date = 1 if st.checkbox(
            'Tuition fees up to date', help='Apakah mahasiswa tidak terlambat dalam pembayaran uang kuliah atau tidak') else 0            
    with col2:
        Debtor = 1 if st.checkbox(
            'Debtor', help='Apakah mahasiswa seorang penghutang atau tidak') else 0            

    st.markdown("---")
    st.markdown("Akurasi Model: 84%")

    # Mengubahnya menjadi dataframe
    data = [[Age_at_enrollment, Gender, Displaced, Application_mode, Course,
             Previous_qualification_grade, Admission_grade, 
             Curricular_units_1st_sem_approved, Curricular_units_1st_sem_grade, Curricular_units_1st_sem_evaluations,
             Curricular_units_2nd_sem_approved, Curricular_units_2nd_sem_grade, Curricular_units_2nd_sem_evaluations,
             Tuition_fees_up_to_date, Debtor]]

    predict_df = pd.DataFrame(data, columns=[
        'Age_at_enrollment', 'Gender', 'Displaced', 'Application_mode', 'Course',
        'Previous_qualification_grade', 'Admission_grade', 
        'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade', 'Curricular_units_1st_sem_evaluations',
        'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_evaluations',
        'Tuition_fees_up_to_date', 'Debtor'])
    
    # Memprediksi hasil
    @st.dialog('Hasil')
    def hasil(output):
        status = "Not Dropout" if output == 1 else "Dropout"
        # Menambahkan gaya menggunakan Markdown (terbatas pada warna teks)
        if status == "Not Dropout":
            st.markdown(f"### Student Status Prediction: **{status}**")
        else:
            st.markdown(f"### Student Status Prediction: **{status}**")

    # Tombol prediksi
    if st.button('✨ Prediksi'):
        X_encoded = encoding(predict_df)
        model = joblib.load('./model/joblib_model.pkl')
        # with open('pickle_model.pkl', 'rb') as file:
        #    model = pickle.load(file)
        output = predict(model, X_encoded)
        hasil(output)

    st.caption('Copyright (C) 2025 by Meakhel Gunawan')

if __name__ == '__main__':
    main()