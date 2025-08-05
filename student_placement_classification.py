import numpy as np
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

api = KaggleApi()
api.authenticate()

data_name = "sahilislam007/college-student-placement-factors-dataset"
api.dataset_download_files(data_name, path='./dataset', unzip=True)

student_placement = pd.read_csv('./dataset/college_student_placement_dataset.csv')

student_placement['Internship_Experience'] = student_placement['Internship_Experience'].apply(lambda x: 1 if x == 'Yes' else 0)
student_placement['Placement'] = student_placement['Placement'].apply(lambda x: 1 if x == 'Yes' else 0)

def remove_outlier_iqr(df, columns, x):
    data_clean = df.copy()
    for col in columns:
        q1 = data_clean[col].quantile(0.25)
        q3 = data_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - x * iqr
        upper_bound = q3 + x * iqr
        data_clean = data_clean[(data_clean[col] >= lower_bound) & (data_clean[col] <= upper_bound)]
    return data_clean

numeric_cols = student_placement.select_dtypes(exclude='object').columns.tolist()
numeric_cols.remove('Placement')

student_placement_no_outliers = remove_outlier_iqr(student_placement, numeric_cols, 1.5)

#select_cols = student_placement_no_outliers.select_dtypes(exclude='object').columns.tolist()
#student_placement_no_outliers[select_cols].corr().style.background_gradient(cmap='Blues', vmin=0, vmax=1).format(precision=2)

X = student_placement_no_outliers.drop(['College_ID', 'Placement'], axis=1).values
y = student_placement_no_outliers['Placement'].values

smote = SMOTE(random_state=42)

x_smote, y_smote = smote.fit_resample(X, y)

xtrain, xtest, ytrain, ytest = train_test_split(x_smote, y_smote, test_size=0.2, stratify=y_smote, random_state=42)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier())
])

param_grid = [
    {
        'model': [KNeighborsClassifier()],
        'model__n_neighbors': [3, 5, 7, 9],
        'model__weights': ['uniform', 'distance'],
        'model__p': [1, 2]
    },
    {   # SVM
        'model': [SVC()],
        'model__C': [0.1, 1, 10],
        'model__kernel': ['linear', 'rbf'],
        'model__gamma': ['scale', 'auto']
    },
    {   # Random Forest
        'model': [RandomForestClassifier(random_state=42)],
        'model__n_estimators': [50, 100],
        'model__max_depth': [None, 10, 20]
    },
    {   # Logistic Regression
        'model': [LogisticRegression(max_iter=10000, solver='liblinear')],
        'model__C': [0.01, 0.1, 1, 10],
        'model__penalty': ['l1', 'l2']
    }
]

grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy'
)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_score = cross_val_score(grid_search, xtrain, ytrain, cv=kf, scoring='accuracy')
print(f'performance pada masing-masing bagian : {cv_score}')

grid_search.fit(xtrain, ytrain)

ypredict_model = grid_search.predict(xtest)

accuracy_model = accuracy_score(ytest, ypredict_model)

print(classification_report(ytest, ypredict_model))

while True:
    college_id = input("College ID : ")
    iq = int(input("Student IQ (40-155) : "))
    prev_sem_result = float(input("Previous Semester Result (0-10): "))
    cgpa = float(input('CGPA (0-10): '))
    academic_performance = int(input("Academic Performance score (0-10): "))
    intern_experience = input("Internship Experience (Yes/No) : ")
    excul_score = int(input("Extra Curricular Score (0-10): "))
    communaction_skills = int(input("Communication Skills Score (0-10): "))
    project_completed = int(input("Project Completed : "))
    
    new_data = (iq, prev_sem_result, cgpa, academic_performance, intern_experience, excul_score, communaction_skills, project_completed)
    new_data_encoded = [1 if item == 'Yes' else item for item in new_data]
    
    input_data = np.array(new_data_encoded)
    
    data_reshape = input_data.reshape(1, -1)
    
    ypredict_new = grid_search.predict(data_reshape)
    
    if ypredict_new == 0:
        print("❌ PREDIKSI: NO PLACEMENT")
    else:
        print("✅ PREDIKSI: PLACEMENT")
        
    while True:
        continue_input = input("Apakah Anda ingin memasukkan data lagi? (Y/T): ")
        if continue_input == "Y" or continue_input == "y":
            break
        elif continue_input == "T" or continue_input == "t":
            print("program dihentikan")
            exit()
        else:
            print('terjadi kesalahan, program terhenti')
           