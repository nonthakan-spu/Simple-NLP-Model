# Simple NLP Model 
โปรเจคนี้จะเป็นการทดลองสร้างโมเดล NLP ง่ายๆ จากการแปลงข้อความเป็น Vector และใช้การแบ่งกลุ่มข้อความในการหาความสัมพันธ์ระหว่างข้อความกับ label โดยจะใช้โมเดลง่ายๆ เช่น LogisticRegression, SVM หรือ Naive Bayes ในการแบ่งกลุ่ม
## ขั้นตอนที่ 1 โหลดชุดข้อมูล IMDB Movie Reviews 
ชุดข้อมูลนี้เป็นชุดข้อมูลการรีวิวหนังจาก IMDB ทั้งหมด 50,000 รายการ โดยจะแบ่งเป็นชุด train 25,000 รายการและชุด test 25,000 รายการ
```python 
from datasets import load_dataset
import pandas as pd

#ทำการโหลดชุดข้อมูลมาเก็บไว้ในตัวแปร ds
ds = load_dataset("imdb")

#ทำการแบ่งข้อมูลชุด train และ test ออกจากกัน
ds_train, ds_test = ds["train"], ds["test"]
```
