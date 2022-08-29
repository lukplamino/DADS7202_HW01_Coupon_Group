# DADS7202_Group Assignment 1 (Group 2 Coupon)
> **`Which one is better for structured data, traditional ML or MLP?`**

## ✨Highlight
- Highlight1 เช่น ข้อคิดเห็น / การค้นพบ / insight
- Highlight2
- Highlight3
- Highlight4
- Highlight5

## 1.Introduction🎯 

**`Binary classification`**:

This project aims to compare performance of traditional ML models and  a self-designed MLP network model by training  models that can predict if a driver will accept a coupon recommended to his/her in different driving scenarios🚗. (1: Accept coupons, 0: Deny coupons)

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
_รายละเอียดต่างๆของโมเดลที่เลือกใช้ เช่น จำนวนและตำแหน่งการวาง layer, จำนวน nodes, activation function, regularization) _
_(โดยใส่ข้อมูลให้ละเอียดพอที่คนที่มาอ่านจะสามารถไปสร้าง network ตามเราได้)_
## 4. Training🔮
_รายละเอียดของการ train และ validate ข้อมูล รวมถึงทรัพยากรที่ใช้ในการ train โมเดลหนึ่ง ๆ เช่น training strategy (เช่น single loss, compound loss, two-step training, end-to-end training), loss, optimizer (learning rate, momentum, etc), batch size, epoch, รุ่นและจำนวน CPU หรือ GPU หรือ TPU ที่ใช้, เวลาโดยประมาณที่ใช้ train โมเดลหนึ่งตัว ฯลฯ_
## 5. Results📈
 - การแสดงผลลัพธ์เทียบ Train vs Validation (เช่น Loss/Accuracy)
ถ้าเป็นไปได้ควรแสดงไว้ในกราฟเดียวกันเพื่อให้สามารถเทียบ scale ค่าผลลัพธ์และดู underfit / overfit ได้ง่าย
 - ระบุให้ชัดเจนเสมอว่าสิ่งที่กล่าวถึงนั้นเป็นผลลัพธ์บน train set หรือ val set หรือ test set
 - ระบุที่มาของการคำนวณและอธิบายวิธีการคำนวณไว้เสมอ เช่น “ค่าเฉลี่ยของ xxx” ก็ต้องอธิบายว่ามีการใช้ค่าอะไรจากไหนกี่ค่าบ้างนำมาเฉลี่ยกัน
 - ระวังว่าการเปรียบเทียบใด ๆ ต้องเป็นไปบนพื้นฐานของเงื่อนไขที่ยุติธรรมต่อคู่เทียบ เช่น หากจะเปรียบเทียบ training time ว่าโมเดลไหนมากน้อยกว่ากัน ก็ควรจะเปรียบเทียบเป็น training 
time per one epoch (โดยเฉลี่ยค่าจากหลาย ๆ epoch), หากจะเปรียบเทียบ inference time per 
one sample ก็ควรจะต้องเป็นค่าที่เฉลี่ยมาจาก test samples ชุดเดียวกันที่รันบน CPU หรือ GPU 
เดียวกัน, หากจะเปรียบเทียบว่า loss มากหรือน้อยกว่ากัน ก็ควรจะเปรียบเทียบที่สมการการคำนวณ 
loss สมการเดียวกัน เป็นต้น
 - แสดงตัวเลขผลลัพธ์ในรูปของค่าเฉลี่ย mean±SD โดยให้ทำการเทรนโมเดลด้วย initial random weights ที่แตกต่างกันอย่าง น้อย 3-5 รอบเพื่อให้ได้อย่างน้อย 3-5 โมเดลมาหาประสิทธิภาพเฉลี่ยกัน, แสดงผลลัพธ์การ train โมเดลเป็นกราฟเทียบ train vs.validation, สรุปผลว่าเกิด underfit หรือ overfit หรือไม่, อธิบาย evaluation metric ที่ใช้ในการประเมินประสิทธิภาพของโมเดลบน train/val/test sets ตามความเหมาะสมของปัญหา, หากสามารถเปรียบเทียบผลลัพธ์ของโมเดลเรากับโมเดลอื่น ๆ (ของคนอื่น) บน any standard benchmark dataset ได้ด้วยจะยิ่งทำให้งานดูน่าเชื่อถือยิ่งขึ้น เช่น เทียบความแม่นยำ เทียบเวลาที่ใช้train เทียบเวลาที่ใช้inference บนซีพียูและจีพียู เทียบขนาดโมเดล ฯลฯ

##  (Optional) Ablation study
_**โมเดลเราคงไม่ใหญ่ขนาดนั้นมั้ง หัวข้อนี้ก็อาจจะตัดทิ้งนะ_
_ในกรณีที่โมเดลมีขนาดใหญ่และมีโมเดลย่อยซ้อนอยู่ข้างในอีก
หลายส่วนจนทำให้ยากต่อการสรุปว่าโมเดลส่วนย่อยใดมีนัยสำคัญมากน้อยแค่ไหนต่อผลที่ได้บ้าง 
ในกรณีเหล่านี้นิยมทำ ablation study โดยทดลองลบโมเดลย่อยบางส่วนออก แล้ว train 
โมเดลดังกล่าวใหม่เพื่อดูว่าการดึงออกนี้มีผลทำให้ประสิทธิภาพโมเดลดีขึ้นหรือแย่ลงอย่างไร_ 

## 6. Discussion💭
_อภิปรายผลลัพธ์ที่ได้ว่ามีอะไรเป็นไปตามสมมติฐาน หรือมีอะไรผิดคาด ไม่เป็นไปตามสมมติฐานบ้าง, วิเคราะห์เพิ่มเติมว่าสิ่งที่ผิดคาดหรือผิดปกตินั้นน่าจะเกิดจากอะไร, ในกรณีที่ dataset มีปัญหา เช่น imalanced dataset ควรวิเคราะห์ด้วยว่าวิธีแก้ที่เราใช้สามารถแก้ปัญหาของ dataset ได้จริงหรือไม่_
## 7. Conclusion📊
_การอภิปรายผลและการสรุปผล ต้องอ้างอิงกับผลการทดลองของเราที่ได้ออกมาเป็นหลัก
มิใช่การนำข้อสรุปที่เป็น general conclusion จากหนังสือ แบบเรียน หรือจากแหล่งอื่น ๆ 
ในอินเทอร์เน็ต มาเขียนซ้ำโดยไม่มีผลการทดลองใด ๆ ของเรามาช่วยสนับสนุนข้อสรุปดังกล่าว_

_วิเคราะห์ด้วยว่าวิธีแก้ที่เราใช้สามารถแก้ปัญหาของ dataset ได้จริงหรือไม่ หรือจุดประสงค์หลัก (objective) ของการบ้านแต่ละครั้ง_

## 8. References🌐
_อ้างอิงไลบรารีที่ใช้ (พร้อมเวอร์ชัน), อ้างอิงเทคนิคที่ยืมมาใช้จากเปเปอร์, อ้างอิงโค้ดหรือรูปภาพที่หยิบยืมมาใช้จาก github หรือจากที่อื่น ๆ_

## Citing: 
_ ในกรณีที่มีคนอยาก cite (อ้างอิง) งานหรือ dataset ของเรา เราอยากให้เขา cite เราว่าอย่างไร ส่วนใหญ่นิยมเขียนในรูปแบบของ bibtex format ตามตัวอย่างในภาพ_


## 👥 Members, Percent Contribution and Responsibility
|No  |ID                |Name                              |% Contribution |Responsibility                             |
|:---:|:---:             |---                               |:---:            |:---|
|1.  |**`6410422002`**  |[Navapol San.](https://www.kaggle.com/navapol)                      |   **`25%`**     |**`Explore data`** **`Prepare dataset`** **`Experiment with traditional ML`** **`Experiment with MLP `**  
|2.  |**`6410422003`**  |[Pakkawut Kam.](https://www.kaggle.com/ppakawut)                     |   **`25%`**     |**`Explore data`** **`Prepare dataset`** **`Experiment with traditional ML`** **`Experiment with MLP `** |
|3.  |**`6410422024`**  |[Supisara Poo.](https://www.kaggle.com/supisarapo)                     |   **`25%`**     |**`Explore data`** **`Prepare dataset`** **`Experiment with MLP `** **`Evaluate and conclude result`**  |
|4.  |**`6410422027`**  |[Kantima Tec.](https://www.kaggle.com/kantimatec)                     |   **`25%`**     |**`Explore data`** **`Prepare dataset`** **`Experiment with traditional ML`** **`Evaluate and conclude result`** |


## 🖇️End Credit 
This project is a part of **`DADS7202 Deep Learning`**

Term: 1 Year of education: 2022

🎓Master of Science Program in **`Data Analytics and Data Science`** (DADS)

🏫National Institute of Development Administration (**`NIDA`**)

<img src="https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252"/> 

<img src="https://github.com/lukplamino/DADS7202_HW01_Coupon_Group/blob/main/images/Screenshot%202022-08-29%20174351.png"/>
