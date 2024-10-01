import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# สมมติว่าคุณมี dataset ที่ชื่อว่า dataset.csv
# อ่านข้อมูลจากไฟล์ CSV
data = pd.read_csv('vaccination_all_tweets.csv')

# สมมติว่า dataset มีฟีเจอร์เหล่านี้: 'user_name', 'user_location', 'user_description', 'text', 'hashtags', 'retweets', 'favorites'
# และคอลัมน์เป้าหมาย (target column) คือ 'label' (ที่คุณต้องการคาดเดา เช่น 1 = Retweet, 0 = Not Retweet)

# ก่อนอื่นเราต้องทำการแปลงข้อมูลที่เป็นข้อความให้เป็นตัวเลข (หากจำเป็น)
le = LabelEncoder()

# คุณอาจต้องการแปลงฟีเจอร์ที่เป็นประเภท (categorical features) เช่น user_name, user_location
data['user_name_enc'] = le.fit_transform(data['user_name'])
data['user_location_enc'] = le.fit_transform(data['user_location'])
data['user_description_enc'] = le.fit_transform(data['user_description'])
data['text_enc'] = le.fit_transform(data['text'])
data['hashtags_enc'] = le.fit_transform(data['hashtags'])

# เลือกฟีเจอร์ (features) สำหรับการฝึกโมเดล
X = data[['user_name_enc', 'user_location_enc', 'user_description_enc', 'text_enc', 'hashtags_enc', 'retweets', 'favorites']]

# กำหนดคอลัมน์เป้าหมาย (label)
y = data['is_retweet']  # คอลัมน์นี้ต้องประกอบด้วยค่าที่คุณต้องการให้โมเดลทำนาย เช่น 1 หรือ 0

# แบ่งข้อมูลสำหรับการฝึกโมเดลและการทดสอบ (training and testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดล Decision Tree
model = DecisionTreeClassifier()

# ฝึกโมเดล
model.fit(X_train, y_train)

# ทำนายผลลัพธ์สำหรับข้อมูลทดสอบ
y_pred = model.predict(X_test)

# ประเมินความแม่นยำของโมเดล
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# บันทึกโมเดลที่ฝึกแล้วลงในไฟล์ .pkl
joblib.dump(model, 'decision_tree_model.pkl')

print("Model saved as decision_tree_model.pkl")
