import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA

data_dir = "data/"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

dir_path = 'E:/scincefi_2/UMAFall_Dataset'
window_size = 450
slide_size = 50
begin = 50
end = 500

activity_dict = {"Bending":3, "Hopping":4, "Jogging":2, "LyingDown":7,
                 "Sitting":8, "Walking":1, "GoDownstairs":6, "GoUpstairs":5,
                 "backwardFall":11, "forwardFall":10, "lateralFall":9,
                 "ClappingHands": 12, "HandsUp": 13, "MakingACall": 14, 
                 "OpeningDoor": 15, "GettingUpOnAChair": 16, "OnABed": 17}

labels = []
X = []
for path, dirs, files in os.walk(dir_path):
    print("-------")
    dirs.sort()
    files.sort()

    for file in files:
        file_str_split = file.split("_")
        print(file_str_split)
        label_s = int(file_str_split[2])
        label_a = activity_dict.get(file_str_split[4], None)
        
        if label_a is None:
            print(f"Activity {file_str_split[4]} not found in activity_dict.")
            continue

        file_str = os.path.join(path, file)
        try:
            df = pd.read_csv(file_str, header=None, sep=",", comment="%", on_bad_lines='skip')
            print(df.shape)
        except pd.errors.ParserError as e:
            print(f"Error parsing {file_str}: {e}")
            continue

        try:
            df_selected = df.loc[(df[5] == 0) & (df[6] == 0)]
        except KeyError as e:
            print(f"KeyError: {e}. Please check if the column names are correct.")
            continue

        stop = df_selected.shape[0] // 4
        idx = [k * 4 for k in range(stop)]
        df_converted = df_selected.iloc[idx, 2:5]

        # Chuyển đổi giá trị không thể thành float sang NaN
        df_converted = df_converted.apply(pd.to_numeric, errors='coerce')

        # Xóa các hàng chứa giá trị NaN
        df_converted.dropna(inplace=True)

        scaler = StandardScaler()
        kpca = KernelPCA(n_components=1, kernel='rbf', random_state=2018, n_jobs=8)

        for j in range(5):
            X_window = df_converted[(begin + j * slide_size): (end + j * slide_size)].values
            if X_window.size == 0:
                print("No data available in the selected window. Skipping...")
                continue
            
            # Kiểm tra kích thước của X_window trước khi chuẩn hóa
            if X_window.shape[0] < window_size:
                print(f"Window size mismatch: expected {window_size}, got {X_window.shape[0]}. Skipping...")
                continue

            scaled = scaler.fit_transform(X_window)

            # Kiểm tra kích thước sau khi áp dụng KernelPCA
            if scaled.shape[0] != window_size:
                print(f"Scaled size mismatch: expected {window_size}, got {scaled.shape[0]}. Skipping...")
                continue

            X_transformed = kpca.fit_transform(scaled).reshape((window_size,))
            X.append(X_transformed)
            labels.append([label_a, label_s])

if len(X) > 0:
    X = np.vstack(X)
    print(X.shape)

    df_label = pd.DataFrame(labels)
    print(df_label.shape)

    pickle.dump(X, open("data/X_umafall_kpca.p", "wb"))
    pickle.dump(df_label, open("data/y_umafall_kpca.p", "wb"))
else:
    print("No data available for processing.")
