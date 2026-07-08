import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def run_preprocessing(input_path, output_dir):
    print("Memulai proses preprocessing...")
    
    # 1. Memuat dataset
    df = pd.read_csv(input_path)

    # 2. Menghapus duplikat
    df_clean = df.drop_duplicates()

    # 3. Memisahkan fitur dan target
    X = df_clean.drop('output', axis=1)
    y = df_clean['output']

    # 4. Membagi data latih dan uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 5. Standarisasi data numerik
    numerical_cols = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # 6. Menggabungkan kembali fitur dan target untuk disimpan
    train_data = pd.concat([pd.DataFrame(X_train, columns=X.columns), y_train.reset_index(drop=True)], axis=1)
    test_data = pd.concat([pd.DataFrame(X_test, columns=X.columns), y_test.reset_index(drop=True)], axis=1)

    # 7. Menyimpan data yang sudah diproses
    os.makedirs(output_dir, exist_ok=True)
    train_data.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test_data.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"Preprocessing selesai! Data disimpan di folder: {output_dir}")

if __name__ == "__main__":
    # Menentukan jalur input data mentah dan output data bersih
    # Sesuaikan jalur ini jika struktur foldermu berbeda
    input_file = 'heart_raw/heart.csv'
    output_folder = 'heart_preprocessing'
    
    run_preprocessing(input_file, output_folder)