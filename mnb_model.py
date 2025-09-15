import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from joblib import dump
from time import time

#โหลดชุดข้อมูลที่บันทึกไว้
#เปลี่ยนตำแหน่งที่อยู่ไฟล์เป็นของคุณ
DIR = "datasets/"
train = pd.read_csv(DIR+"cleaned_train.csv")
val = pd.read_csv(DIR+"cleaned_val.csv")
test = pd.read_csv(DIR+"cleaned_test.csv")

#แบ่งข้อมูลเป็น text กับ label ทุกชุดข้อมูลและเปลี่ยนเป็น list เพื่อให้ง่ายต่อการเรียกใช้
x_train, y_train = train["text"].tolist(), train["label"].tolist()
x_val, y_val = val["text"].tolist(), val["label"].tolist()
x_test, y_test = test["text"].tolist(), test["label"].tolist()

#สร้างตัวแปรขึ้นมาเก็บค่าสำหรับวัดประสิทธิภาพของโมเดล
best_score = 0
best_alpha_value = None
best_model = None
#กำหนดค่า C ที่ต้องการจะทดสอบ
test_alpha_value = [0.1, 1.0, 10.0, 100.0]
#เริ่มจับเวลา
start = time()
#สร้างลูปขึ้นมาเพื่อวนทดสอบค่า alpha แต่ละค่าว่าค่าใดส่งผลทำให้โมเดลมีประสิทธิภาพมากที่สุด(ในที่จะวัดด้วยค่า Accuracy) และทำการเก็บค่า alpha ที่ดีที่สุดและโมเดลที่ดีที่สุด
for alpha_value in test_alpha_value:
  #สร้าง pipeline เชื่อมระหว่าง TF-IDF กับ MNB
  mnb_model_temp = Pipeline([
      ("tfidf", TfidfVectorizer(ngram_range=(1,3), max_features=100_000, min_df=2, max_df=0.8)),
      ("mnb", MultinomialNB(alpha=alpha_value))
  ])

  #ฝึกโมเดลด้วยชุดข้อมูล train
  mnb_model_temp.fit(x_train, y_train)
  #ประเมินผลโมเดลด้วยชุดข้อมูล val
  val_score = mnb_model_temp.score(x_val, y_val)

  #ทำการบันทึกค่า score, alpha และ model ไปยังตัวแปรที่เตรียมไว้
  if best_score < val_score:
    best_score = val_score
    best_alpha_value = alpha_value
    best_model = mnb_model_temp

  #แสดงผลค่า alpha ที่ใช้ในการฝึกและ Score ที่ได้จากการฝึก
  print(f"Alpha value: {alpha_value:.2f}\nScore: {val_score:.4f}")

#หยุดเวลา
end = time()
#แสดงผลระยะเวลาที่ใช้ในการฝึกโมเดล
print(f"Training time: {end-start:.2f} seconds")
#แสดงผลค่า alpha และ Score ที่ดีที่สุดที่ได้จากการฝึก
print(f"Best alpha value: {best_alpha_value:.2f}\nBest Score: {best_score:.4f}")
#ทดสอบโมเดลด้วยชุดข้อมูล test
mnb_pred = best_model.predict(x_test)
#แสดงผลการทดสอบด้วยชุดข้อมูล test
print("Test Set Evaluate\n",classification_report(y_test, mnb_pred))

#บันทึกโมเดล
#เปลี่ยนตำแหน่งที่เก็บเป็นของคุณเอง
SAVE_DIR = "models/"
dump(best_model, SAVE_DIR+"MNB_Model.joblib")