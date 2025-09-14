from datasets import load_dataset
import pandas as pd
import re
from sklearn.model_selection import train_test_split

#ทำการโหลดชุดข้อมูลมาเก็บไว้ในตัวแปร ds
ds = load_dataset("imdb")

#ทำการแบ่งข้อมูลชุด train และ test ออกจากกันและแปลงข้อมูลจาก DataDict เป็น DataFrame
ds_train, ds_test = pd.DataFrame(ds["train"]), pd.DataFrame(ds["test"])

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
DIR = "datasets/"
#เปลี่ยนชื่อไฟล์หรือจะไม่เปลี่ยนก็ได้
ds_train.to_csv(DIR+"cleaned_train.csv")
ds_val.to_csv(DIR+"cleaned_val.csv")
ds_test.to_csv(DIR+"cleaned_test.csv")