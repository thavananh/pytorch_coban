import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Bước 1: Đọc file CSV
# Giả sử file CSV có cấu trúc: user_id, movie, rating
file_path = 'ratings.csv'  # Thay đổi đường dẫn tới file CSV của bạn
data = pd.read_csv(file_path)

# Bước 2: Xử lý dữ liệu
# Tạo ma trận xếp hạng: hàng là user_id, cột là movie, giá trị là rating
ratings_matrix = data.pivot(index='user_id', columns='movie', values='rating').fillna(0)

# Bước 3: Tính toán cosine similarity
# Tính ma trận tương đồng giữa các người dùng
similarity_matrix = cosine_similarity(ratings_matrix)

# Bước 4: Hiển thị kết quả
# Chuyển ma trận tương đồng thành DataFrame để dễ xem
similarity_df = pd.DataFrame(similarity_matrix, index=ratings_matrix.index, columns=ratings_matrix.index)

# In ma trận độ tương đồng
print("Ma trận độ tương đồng cosine giữa các người dùng:")
print(similarity_df)

# Ví dụ: In độ tương đồng giữa User 1 và User 2
user1_id = 1  # Thay đổi nếu user_id khác
user2_id = 2  # Thay đổi nếu user_id khác
if user1_id in similarity_df.index and user2_id in similarity_df.index:
    similarity = similarity_df.loc[user1_id, user2_id]
    print(f"\nĐộ tương đồng cosine giữa User {user1_id} và User {user2_id} là: {similarity:.3f}")
else:
    print(f"Không tìm thấy User {user1_id} hoặc User {user2_id} trong dữ liệu.")