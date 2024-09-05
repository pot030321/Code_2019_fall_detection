import os
import numpy as np
import pandas as pd
import pickle

# Khởi tạo đường dẫn để đọc dữ liệu
data_dir = "data/"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Khởi tạo khung hình và đọc tất cả file CSV nằm trong đường dẫn 
dir_path = 'E:/scincefi_2/UMAFall_Dataset'

window_size = 450 * 3  # Kích thước cửa sổ dữ liệu (mỗi cửa sổ có 450 mẫu, với 3 trục x, y, z)
slide_size = 50        # Kích thước trượt để chia các cửa sổ dữ liệu
begin = 50             # Chỉ số bắt đầu cho cửa sổ dữ liệu
end = 500              # Chỉ số kết thúc cho cửa sổ dữ liệu

# Khởi tạo nhãn cho dữ liệu, ánh xạ tên hoạt động sang số
activity_dict = {"Bending": 3, "Hopping": 4, "Jogging": 2, "LyingDown": 7,
                 "Sitting": 8, "Walking": 1, "GoDownstairs": 6, "GoUpstairs": 5,
                 "backwardFall": 11, "forwardFall": 10, "lateralFall": 9,
                 "ClappingHands": 12, "HandsUp": 13, "MakingACall": 14, 
                 "OpeningDoor": 15, "GettingUpOnAChair": 16, "OnABed": 17}  # Thêm hoạt động mới

labels = []  # Danh sách để lưu nhãn dữ liệu
X = []       # Danh sách để lưu dữ liệu

# Duyệt qua tất cả các file trong thư mục
for path, dirs, files in os.walk(dir_path):
    print("-------")
    dirs.sort()
    files.sort()

    for file in files:
        file_str_split = file.split("_")
        print(file_str_split)
        
        # Lấy nhãn từ tên file
        label_s = int(file_str_split[2])  # ID của người thực hiện hoạt động
        label_a = activity_dict.get(file_str_split[4], None)  # Loại hoạt động (được ánh xạ từ từ điển)
        
        # Kiểm tra xem hoạt động có tồn tại trong từ điển không
        if label_a is None:
            print(f"Activity {file_str_split[4]} not found in activity_dict.")
            continue
        
        # Đọc dữ liệu từ file CSV
        file_str = os.path.join(path, file)
        df = pd.read_csv(file_str, header=0, sep=",", skiprows=40, comment="%")
        print(df.shape)
        
        # Kiểm tra xem DataFrame có chứa ít nhất 6 cột để thực hiện lọc
        if df.shape[1] < 7:
            print(f"DataFrame does not have enough columns for filtering: {df.shape[1]} columns found.")
            continue

        # Lọc dữ liệu chỉ lấy các mẫu được thu từ điện thoại và cảm biến Accelerometer
        try:
            df_selected = df.loc[(df.iloc[:, 5] == 0) & (df.iloc[:, 6] == 0)]
        except KeyError as e:
            print(f"KeyError: {e}. Please check if the column names are correct.")
            continue

        # Xác định giới hạn để dừng vòng lặp (tối ưu hóa bằng cách chia nhỏ dữ liệu thành các phần nhỏ hơn)
        stop = df_selected.shape[0] // 4  # Chia lấy nguyên để đảm bảo kết quả là số nguyên
        idx = [k * 4 for k in range(stop)]  # Khởi tạo danh sách chỉ số cho các dòng cần lấy mẫu
        
        # Lấy mẫu các dòng dữ liệu cần thiết
        df_converted = df_selected.iloc[idx, 2:5]  # Lấy các cột chứa thông tin trục x, y, z

        # Chia dữ liệu thành các cửa sổ thời gian và thêm vào danh sách X
        for j in range(5):
            X_selected = df_converted[(begin + j * slide_size):(end + j * slide_size)].values
            if X_selected.size == 0:
                print("No data available in the selected window. Skipping...")
                continue
            X.append(X_selected.reshape((window_size,)))
            labels.append([label_a, label_s])

# Chuyển danh sách X thành mảng numpy để lưu trữ
if len(X) > 0:
    X = np.vstack(X)
    print(X.shape)  # In ra kích thước mảng X sau khi gộp

    # Tạo DataFrame cho nhãn dữ liệu
    df_label = pd.DataFrame(labels)
    print(df_label.shape)  # In ra kích thước nhãn dữ liệu

    # Lưu dữ liệu đã xử lý và nhãn vào file
    pickle.dump(X, open("data/X_umafall_raw.p", "wb"))
    pickle.dump(df_label, open("data/y_umafall_raw.p", "wb"))
else:
    print("No data available for processing.")

'''
# Load raw data
X = pickle.load(open("data/X_umafall_raw.p", "rb"))
'''
