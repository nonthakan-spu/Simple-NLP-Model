# Simple NLP Model 
โปรเจคนี้จะเป็นการทดลองสร้างโมเดล NLP ง่ายๆ จากการแปลงข้อความเป็น Vector และใช้การแบ่งกลุ่มข้อความในการหาความสัมพันธ์ระหว่างข้อความกับ label โดยจะใช้โมเดลง่ายๆ เช่น LogisticRegression, SVM หรือ Naive Bayes ในการแบ่งกลุ่ม
## ขั้นตอนที่ 1 โหลดชุดข้อมูล IMDB Movie Reviews 
ชุดข้อมูลนี้เป็นชุดข้อมูลการรีวิวหนังจาก IMDB ทั้งหมด 50,000 รายการ โดยจะแบ่งเป็นชุด train 25,000 รายการและชุด test 25,000 รายการ
```python 
from datasets import load_dataset
import pandas as pd

#ทำการโหลดชุดข้อมูลมาเก็บไว้ในตัวแปร ds
ds = load_dataset("imdb")

#ทำการแบ่งข้อมูลชุด train และ test ออกจากกันและแปลงข้อมูลจาก DataDict เป็น DataFrame
ds_train, ds_test = pd.DataFrame(ds["train"]), pd.DataFrame(ds["test"])
```
## ขั้นตอนที่ 2 ทำความสะอาดข้อมูล แบ่งข้อมูลและบันทึกชุดข้อมูล
ลบข้อมูลที่ไม่จำเป็นออกไป เช่น tag HTML, ช่องว่าง หรือ ตัวอักษรพิเศษต่างๆ เพื่อทำให้ข้อมูลเตรียมพร้อมจะนำไปใช้ในการฝึกโมเดล เพื่อให้โมเดลเข้าใจและเรียนรู้ได้ตรงประเด็นมากที่สุด แบ่งข้อมูลจากชุด test 10% เพื่อเป็นชุด val (Validation) ใช้สำหรับประเมินประสิทธิภาพของโมเดลในระหว่างการฝึก
```python
import re
import pandas as pd
from sklearn.model_selection import train_test_split

#สร้างฟังก์ชันสำหรับการทำความสะอาดข้อมูล
def clean(s: str) -> str:
  #สำหรับลบ tag HTML และไม่แทนที่ในตำแหน่งที่ลบ
  s = re.sub(r"<.*?>", "", s)
  #สำหรับลบตัวอักษรทุกชนิดที่ไม่ใช่ตัวอักษร a-z, A-Z, 0-9 และ ช่องว่าง และแทนที่ด้วยช่องว่าง 1 ช่อง
  s = re.sub(r"[^a-zA-Z0-9\s]", " ", s)
  #สำหรับลบช่องว่างหลายๆ ช่องที่ติดกันและแทนที่ด้วยช่องว่างๆ 1 ช่อง รวมถึงลบช่องว่างในส่วนหัวและท้ายข้อความด้วย
  s = re.sub(r"\s+", " ", s).strip()
  #ส่งข้อความที่ทำความสะอาดแล้วคืนในรูปแบบที่เป็นตัวพิมพ์เล็กทั้งหมด
  return s.lower()

#ลูปข้อมูลภายใน ds_train และ ds_test เพื่อทำความสะอาดข้อมูลทั้งหมด
for df in (ds_train, ds_test):
  #ทำการแปลงข้อความเป็น string และส่งข้อมูลไปทำความที่ฟังก์ชัน clean
  df["text"] = df["text"].astype(str).apply(clean)
  #ทำการแปลง label เป็น integer
  df["label"] = df["label"].astype(int)

#แบ่งข้อมูลจากชุด test ไปยังชุด val 10% โดยอิงจาก label เพื่อให้สัดส่วนของข้อมูลมีความใกล้เคียงกัน
ds_test, ds_val = train_test_split(ds_test, test_size=0.1, random_state=42, stratify=ds_test["label"])

#บันทึกชุดข้อมูลเพื่อเก็บไว้ใช้ในครั้งหน้า
#เปลี่ยนตำแหน่งที่เก็บเป็นของคุณเอง
DIR = "/your/directory/path/"
#เปลี่ยนชื่อไฟล์หรือจะไม่เปลี่ยนก็ได้
ds_train.to_csv(DIR+"cleaned_train.csv")
ds_val.to_csv(DIR+"cleaned_val.csv")
ds_test.to_csv(DIR+"cleaned_test.csv")
```
## ขั้นตอนที่ 3 สร้าง Pipeline สำหรับโมเดลและ TF-IDF
การสร้าง pipeline ที่เชื่อมต่อระหว่างโมเดลกับตัวแปลงข้อความเป็น vector จะทำให้เราเขียนโค้ดสั้นลงและทำให้งานง่ายขึ้น ลดโอกาสความผิดพลาดที่จะเกิดขึ้นระหว่างการแปลงข้อความและส่งไปยังโมเดลเพื่อฝึก
```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from joblib import dump

#โหลดชุดข้อมูลที่บันทึกไว้
#เปลี่ยนตำแหน่งที่อยู่ไฟล์เป็นของคุณ
DIR = "/your/directory/path"
train = pd.read_csv(DIR+"cleaned_train.csv")
val = pd.read_csv(DIR+"cleaned_val.csv")
test = pd.read_csv(DIR+"cleaned_test.csv")

#แบ่งข้อมูลเป็น text กับ label ทุกชุดข้อมูลและเปลี่ยนเป็น list เพื่อให้ง่ายต่อการเรียกใช้
x_train, y_train = train["text"].tolist(), train["label"].tolist()
x_val, y_val = val["text"].tolist(), val["label"].tolist()
x_test, y_test = test["text"].tolist(), test["label"].tolist()

#สร้างตัวแปรขึ้นมาเก็บค่าสำหรับวัดประสิทธิภาพของโมเดล
best_score = 0
best_c_value = None
best_model = None
#กำหนดค่า C ที่ต้องการจะทดสอบ
test_c_value = [0.1, 1.0, 10.0, 100.0]
#สร้างลูปขึ้นมาเพื่อวนทดสอบค่า c แต่ละค่าว่าค่าใดส่งผลทำให้โมเดลมีประสิทธิภาพมากที่สุด(ในที่จะวัดด้วยค่า Accuracy) และทำการเก็บค่า c ที่ดีที่สุดและโมเดลที่ดีที่สุด
for c_value in test_c_value:
  #สร้าง pipeline เชื่อมระหว่าง TF-IDF กับ LogisticRegression
  lr_model_temp = Pipeline([
      ("tfidf", TfidfVectorizer(ngram_range=(1,3), max_features=100_000, min_df=2, max_df=0.8)),
      ("lr", LogisticRegression(max_iter=500, C=c_value, solver="liblinear", n_jobs=-1))
  ])

  #ฝึกโมเดลด้วยชุดข้อมูล train
  lr_model_temp.fit(x_train, y_train)
  #ประเมินผลโมเดลด้วยชุดข้อมูล val
  val_score = lr_model_temp.score(x_val, y_val)

  #ทำการบันทึกค่า score, C และ model ไปยังตัวแปรที่เตรียมไว้
  if best_score < val_score:
    best_score = val_score
    best_c_value = c_value
    best_model = lr_model_temp

  #แสดงผลค่า C ที่ใช้ในการฝึกและ Score ที่ได้จากการฝึก
  print(f"C value: {c_value:.2f}\nScore: {val_score:.4f}")

#แสดงผลค่า C และ Score ที่ดีที่สุดที่ได้จากการฝึก
print(f"Best C value: {best_c_value:.2f}\nBest Score: {best_score:.4f}")
#ทดสอบโมเดลด้วยชุดข้อมูล test
lr_pred = best_model.predict(x_test)
#แสดงผลการทดสอบด้วยชุดข้อมูล test
print("Test Set Evaluate\n",classification_report(y_test, lr_pred))

#บันทึกโมเดล
#เปลี่ยนตำแหน่งที่เก็บเป็นของคุณเอง
SAVE_DIR = "/your/directory/path/"
dump(SAVE_DIR+"LogisticRegression_Model.joblib")
```
