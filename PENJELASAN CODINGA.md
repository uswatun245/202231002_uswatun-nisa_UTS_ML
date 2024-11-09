# Import libraries
import pandas as pd # Mengimpor library pandas untuk mengelola dan memproses data.
import numpy as np # Mengimpor library numpy untuk operasi numerik dan array.
import matplotlib.pyplot as plt # Mengimpor matplotlib untuk visualisasi data.
from sklearn.model_selection import train_test_split # Mengimpor fungsi
from sklearn.linear_model import LinearRegression # Mengimpor kelas LinearRegression untuk membuat model regresi linier.
from sklearn.metrics import mean_squared_error, r2_score # untuk mengevaluasi performa model.

heart_disease_df = pd.read_csv('K01_heart_disease.csv') # Memuat dataset penyakit jantung dari file CSV dan menyimpannya ke dalam DataFrame
students_performance_df = pd.read_csv('R01_students_performance.csv') # Memuat dataset performa siswa dari file CSV dan menyimpannya ke dalam DataFrame

print("Heart Disease Dataset Sample:") # Mencetak teks "Heart Disease Dataset Sample:"
display(heart_disease_df.head()) # Menampilkan lima baris pertama dari DataFrame heart_disease_df.

X_heart = heart_disease_df[['Age']].values  # Mengambil kolom Age sebagai fitur independen X_heart untuk prediksi.
y_heart = heart_disease_df['Cholesterol'].values  # Mengambil kolom Cholesterol sebagai target y_heart untuk prediksi.

# Split data
X_train_heart, X_test_heart, y_train_heart, y_test_heart = train_test_split(X_heart, y_heart, test_size=0.3, random_state=42) # Membagi data menjadi data latih (70%) dan data uji (30%), dengan pengacakan yang konsisten 

# Linear Regression Model
reg_heart = LinearRegression() # Membuat objek model regresi linier untuk dataset penyakit jantung.
reg_heart.fit(X_train_heart, y_train_heart) # Melatih model pada data latih
y_pred_heart = reg_heart.predict(X_test_heart) # Menggunakan model terlatih untuk memprediksi nilai kolesterol pada data uji

# Evaluate model
mse_heart = mean_squared_error(y_test_heart, y_pred_heart) # Menghitung Mean Squared Error (MSE) antara data aktual dan prediksi sebagai indikator kesalahan model.
r2_heart = r2_score(y_test_heart, y_pred_heart) # Menghitung koefisien determinasi (R-squared) sebagai indikator kecocokan model dengan data.

# Plot Heart Disease Regression
plt.figure(figsize=(10, 5))  #  Membuat gambar berukuran 10x5 inci.
plt.scatter(X_test_heart, y_test_heart, color='blue', label='Actual Data') # Membuat plot sebar untuk data uji dengan warna biru.
plt.plot(X_test_heart, y_pred_heart, color='red', linewidth=2, label='Regression Line') # Membuat garis regresi berdasarkan prediksi model.
plt.title('Regression Model for Age vs Cholesterol')
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.legend()
plt.show()

# Menampilkan hasil MSE dan R-squared untuk dataset penyakit jantung.
print("Heart Disease Dataset - Regression Results:")
print("Mean Squared Error:", mse_heart)
print("R-squared:", r2_heart)

X_students = students_performance_df[['Study Hours (X)']].values  # Mengambil kolom untuk prediksi.
y_students = students_performance_df['Exam Scores (Y)'].values  # Mengambil kolom untuk prediksi.

# Split data
# Membagi data menjadi data latih (70%) dan data uji (30%).
X_train_students, X_test_students, y_train_students, y_test_students = train_test_split(X_students, y_students, test_size=0.3, random_state=42)

# Linear Regression Model
# Membuat dan melatih model regresi linier untuk dataset performa siswa, kemudian memprediksi nilai ujian pada data uji.
reg_students = LinearRegression()
reg_students.fit(X_train_students, y_train_students)
y_pred_students = reg_students.predict(X_test_students)

# Evaluate model
# Menghitung Mean Squared Error (MSE) dan R-squared (koefisien determinasi) untuk dataset performa siswa.
mse_students = mean_squared_error(y_test_students, y_pred_students)
r2_students = r2_score(y_test_students, y_pred_students)

# Membuat plot untuk data asli dan garis regresi yang dihasilkan dari model regresi.
plt.figure(figsize=(10, 5))
plt.scatter(X_test_students, y_test_students, color='blue', label='Actual Data')
plt.plot(X_test_students, y_pred_students, color='red', linewidth=2, label='Regression Line')
plt.title('Regression Model for Study Hours vs Exam Scores')
plt.xlabel('Study Hours')
plt.ylabel('Exam Scores')
plt.legend()
plt.show()

# Menampilkan hasil MSE dan R-squared untuk dataset performa siswa.
print("Student Performance Dataset - Regression Results:")
print("Mean Squared Error:", mse_students)
print("R-squared:", r2_students)
