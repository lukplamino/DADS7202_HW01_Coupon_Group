# DADS7202_Group Assignment 1 (Group 2 Coupon)
Which one is better for structured data, traditional ML or MLP?

## ✨Highlight
- Highlight1 เช่น ข้อคิดเห็น / การค้นพบ / insight
- Highlight2
- Highlight3
- Highlight4
- Highlight5

## 1.Introduction🎯 

Binary classification:

This project aims to train a classification model that can predict if a driver will accept a coupon recommended to his/her in different driving scenarios🚗.

(1: Accept coupons, 0: Deny coupons)

## 2. Data📑
#### 📍Data source: 
[In-vehicle coupon recommendation Data Set](https://archive.ics.uci.edu/ml/datasets/in-vehicle+coupon+recommendation)

This data was collected via a survey on Amazon Mechanical Turk. The survey describes different driving scenarios including the destination, current time, weather, passenger, etc., and then ask the person whether he will accept the coupon if he is the driver. 

For more information about the dataset, please refer to the paper:
Wang, Tong, Cynthia Rudin, Finale Doshi-Velez, Yimin Liu, Erica Klampfl, and Perry MacNeille. 'A bayesian framework for learning rule sets for interpretable classification.' The Journal of Machine Learning Research 18, no. 1 (2017): 2357-2393.

#### 🔎Exploratory Data Analysis(EDA): 
##### Data preparation:
##### Data pre-processing:
##### Data post-processing:
#### ✂️Data splitting (train/val/test):
#### 🔨How to solve imbalance data:
_หากชุดข้อมูลที่ใช้มีปัญหาบางประการ เช่น imbalance ในบางคลาส ให้ระบุแนวทางที่จะใช้แก้ปัญหาด้วย (โดยเฉพาะงาน classification)_

## 3. Network architecture📦

## 4. Training🔮

## 5. Results📈
_ - การแสดงผลลัพธ์เทียบ Train vs Validation (เช่น Loss/Accuracy)
ถ้าเป็นไปได้ควรแสดงไว้ในกราฟเดียวกันเพื่อให้สามารถเทียบ scale ค่าผลลัพธ์และดู underfit / overfit ได้ง่าย
 - ระบุให้ชัดเจนเสมอว่าสิ่งที่กล่าวถึงนั้นเป็นผลลัพธ์บน train set หรือ val set หรือ test set
 - ระบุที่มาของการคำนวณและอธิบายวิธีการคำนวณไว้เสมอ เช่น “ค่าเฉลี่ยของ xxx” ก็ต้องอธิบายว่ามีการใช้ค่าอะไรจากไหนกี่ค่าบ้างนำมาเฉลี่ยกัน
 - ระวังว่าการเปรียบเทียบใด ๆ ต้องเป็นไปบนพื้นฐานของเงื่อนไขที่ยุติธรรมต่อคู่เทียบ เช่น หากจะเปรียบเทียบ training time ว่าโมเดลไหนมากน้อยกว่ากัน ก็ควรจะเปรียบเทียบเป็น training 
time per one epoch (โดยเฉลี่ยค่าจากหลาย ๆ epoch), หากจะเปรียบเทียบ inference time per 
one sample ก็ควรจะต้องเป็นค่าที่เฉลี่ยมาจาก test samples ชุดเดียวกันที่รันบน CPU หรือ GPU 
เดียวกัน, หากจะเปรียบเทียบว่า loss มากหรือน้อยกว่ากัน ก็ควรจะเปรียบเทียบที่สมการการคำนวณ 
loss สมการเดียวกัน เป็นต้น_

## 6. Discussion💭

## 7. Conclusion📊
_การอภิปรายผลและการสรุปผล ต้องอ้างอิงกับผลการทดลองของเราที่ได้ออกมาเป็นหลัก

มิใช่การนำข้อสรุปที่เป็น general conclusion จากหนังสือ แบบเรียน หรือจากแหล่งอื่น ๆ 

ในอินเทอร์เน็ต มาเขียนซ้ำโดยไม่มีผลการทดลองใด ๆ ของเรามาช่วยสนับสนุนข้อสรุปดังกล่าว_

## 8. References🌐



## 👥 Members, Percent Contribution and Responsibility
|No  |ID                |Name                              |% Contribution |Responsibility                             |
|:---:|:---:             |---                               |:---:            |:---|
|1.  |**`6410422002`**  |Navapol San.                      |   **`25%`**     |**`Explore data`** **`Prepare dataset`** **`Experiment with traditional ML`** **`Experiment with MLP `**  
|2.  |**`6410422003`**  |Pakkawut Kam.                     |   **`25%`**     |**`Explore data`** **`Prepare dataset`** **`Experiment with traditional ML`** **`Experiment with MLP `** |
|3.  |**`6410422024`**  |Supisara Poo.                     |   **`25%`**     |**`Explore data`** **`Prepare dataset`** **`Experiment with MLP `** **`Evaluate and conclude result`**  |
|4.  |**`6410422027`**  |Kantima Tec.                      |   **`25%`**     |**`Explore data`** **`Prepare dataset`** **`Experiment with traditional ML`** **`Evaluate and conclude result`** |


## 🖇️End Credit 
This project is a part of **`DADS7202 Deep Learning`**

Term: 1 Year of education: 2022

🎓Master of Science Program in **`Data Analytics and Data Science`** (DADS)

🏫National Institute of Development Administration (**`NIDA`**)

<img src="https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252"/> 
