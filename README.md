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
ในส่วนของโมเดล SVM(Support Vector Matchine) ก็จะทำในลักษณะเดียวกันกับโมเดล LR(Logistic Regression) แต่จะมีบาง parameters ที่ไม่เหมือนกันสามารถดูตัวอย่างจากโค้ดด้านล่าง
```python
svm_model_temp = Pipeline([
  ("tfidf", TfidfVectorizer(ngram_range=(1,3), max_features=100_000, min_df=2, max_df=0.8)),
  ("svm", LinearSVC(max_iter=500, C=c_value, penalty='l2', dual=True))
])
```
ส่วนโมเดล MNB(Multinomial Naive Bayes) จะมี parameters น้อยกว่า 2 โมเดลก่อนหน้า และจะเปลี่ยนตัวแปร C เป็น alpha แทนซึ่งมีความแตกต่างในด้านการทำงานและส่งผลต่อโมเดล (จะนำไปอธิบายเพิ่มเติมในส่วนถัดไป)
```python
mnb_model_temp = Pipeline([
  ("tfidf", TfidfVectorizer(ngram_range=(1,3), max_features=100_000, min_df=2, max_df=0.8)),
  ("mnb", MultinomialNB(alpha=alpha_value))
])
```
### Parameters สำคัญที่ถูกใช้บ่อยในการสร้างโมเดล
`max_iter` คือ จำนวนรอบสูงสุดที่โมเดลสามารถวนเพื่อเรียนรู้ได้ แต่ไม่จำเป็นต้องวนให้ครบรอบจำนวนสูงสุดเสมอไปถ้าโมเดลสามารถเรียนรู้ได้จนถึงจุดที่ดีที่สุดแล้ว (ตัวอย่างง่ายๆ ลองนึกว่าอาจารย์สั่งให้คุณไปท่องตารางธาตุมา ซึ่งคุณเองก็ตั้งใจไว้ว่าจะท่องสูงสุดแค่ 500 รอบ(max_iter=500) เพื่อที่จะได้จำได้ทุกตัว แต่ว่าคุณท่องไปเพียง 348 รอบ จนคุณมั่นในแล้วว่าคุณจำได้ คุณก็ไม่จำเป็นต้องท่องต่อแล้ว เพราะว่าคุณมั่นใจแล้ว)

`C` คือ ความรุนแรงของการลงโทษโมเดล ยิ่งค่า C สูงก็จะยิ่งลงโทษรุนแรง และเสี่ยงเกิด Overfitting มากกว่า ถ้าค่า C น้อยก็จะลงโทษโมเดลเบา ทำให้โอกาสที่จะเกิด Overfitting น้อยกว่า (ตัวอย่างง่ายๆ ในระหว่างที่คุณกับเพื่อนของคุณกำลังท่องตารางธาตุอยู่นั้น คุณและเพื่อนก็ได้ตั้งกฎของตัวเองไว้ คุณบอกว่าทุกครั้งท่องผิดก็จะทำการดีดหนังยางที่ใส่ไว้ที่ข้อมือทุกครั้งเพื่อลงโทษ(ค่า C มาก) ส่วนเพื่อนของคุณบอกว่าถ้าท่องผิดติดต่อกัน 3 ครั้งถึงจะดีดหนังยางที่ข้อมือ 1 ครั้งเพื่อลงโทษ(ค่า C น้อย) จะเห็นได้ว่าความเข้มงวดของกฎของคุณนั้นมีมากกว่าเพื่อนของคุณซึ่งอาจจะทำให้คุณเกิดความเครียดได้มากกว่า(โอกาส Oerfitting มากกว่า) เพื่อนของคุณที่เครียดน้อยกว่าเพราะบทลงโทษที่เบากว่า(โอกาส Overfitting น้อยกว่า))

`solver` คือ อัลกอริทึมทางคณิตศาสตร์ที่ใช้หาค่าพารามิเตอร์ของโมเดล เพื่อลดค่า loss ให้น้อยที่สุด (ตัวอย่างง่ายๆ สมมติว่าคุณกำลังจะทำอาหารซึ่งคุณไม่รู้ว่าอาหารที่คุณจะทำมันจะต้องใส่เครื่องปรุงหรือส่วนผสมอะไรบ้าง `solver` เปรียบเสมือนกับสูตรอาหารที่คุณอยากใช้ บางสูตรอาจจะทำแล้วออกมาตรงใจคุณ บางสูตรอาจจะไม่ตรงใจคุณบ้าง เพราะฉะนั้นคุณจึงจำเป็นต้องหาสูตรที่ดีที่สุดเพื่อนำมาทำอาหารที่คุณชอบ(เลือก `solver` ที่เหมาะกับโมเดลเพื่อผลลัพธ์ที่ดีขึ้น))
