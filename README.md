# DADS7202_Group Assignment 1 (Group 2 MNLP)
> **`Which one is better for structured data, traditional ML or MLP?`**

<img src="https://github.com/lukplamino/DADS7202_HW01_MNLP_Group/blob/main/images/Screenshot%202022-08-29%20174351.png" alt="drawing" style="width:400px;"/>

## ✨Highlight
- Highlight1 เช่น ข้อคิดเห็น / การค้นพบ / insight
- Highlight2
- Highlight3
- Highlight4
- Highlight5

## Table of Contents
[Code.ipynp](https://github.com/lukplamino/DADS7202_HW01_MNLP_Group/blob/main/%5BMNLP_Team%5D_7202_HW1_Final_Version.ipynb)
 - [1. Introduction🎯](https://github.com/lukplamino/DADS7202_HW01_MNLP_Group#1-introduction)
 - [2. Data📑](https://github.com/lukplamino/DADS7202_HW01_MNLP_Group#2-data)
 - [3. Network architecture📦 and Training🔮](https://github.com/lukplamino/DADS7202_HW01_MNLP_Group#3-network-architecture-and-training)
 - [4. Results📈](https://github.com/lukplamino/DADS7202_HW01_MNLP_Group#4-results)
 - [5. Discussion💭](https://github.com/lukplamino/DADS7202_HW01_MNLP_Group#5-discussion)
 - [6. Conclusion📊](https://github.com/lukplamino/DADS7202_HW01_MNLP_Group#6-conclusion)
 - [7. References🌐](https://github.com/lukplamino/DADS7202_HW01_MNLP_Group#7-references)
 - [Citing](https://github.com/lukplamino/DADS7202_HW01_MNLP_Group#citing)
 - [👥 Members, Percent Contribution and Responsibility](https://github.com/lukplamino/DADS7202_HW01_MNLP_Group#-members-percent-contribution-and-responsibility)
 - [🖇️End Credit ](https://github.com/lukplamino/DADS7202_HW01_MNLP_Group#%EF%B8%8Fend-credit)


## 1. Introduction🎯 

**`Binary classification`**:

This project aims to compare performance of **`traditional ML models`** and  a **`self-designed MLP network model`** by training  models that can predict if a driver will accept a coupon recommended to his/her in different driving scenarios🚗. (1: Accept coupons, 0: Deny coupons)

## 2. Data📑
#### 📍Data source: 
[In-vehicle coupon recommendation Data Set](https://archive.ics.uci.edu/ml/datasets/in-vehicle+coupon+recommendation)

This data was collected via a survey on Amazon Mechanical Turk. The survey describes different driving scenarios including the destination, current time, weather, passenger, etc., and then ask the person whether he will accept the coupon if he is the driver. 

For more information about the dataset, please refer to the paper:
Wang, Tong, Cynthia Rudin, Finale Doshi-Velez, Yimin Liu, Erica Klampfl, and Perry MacNeille. 'A bayesian framework for learning rule sets for interpretable classification.' The Journal of Machine Learning Research 18, no. 1 (2017): 2357-2393.

#### 🔎Exploratory Data Analysis(EDA): 

#### Data preparation and pre-processing:
To get data ready for model:
 - We managed the difference types of data by converting nominal data into object and ordinal data into integer with order from smallest (0) to greatest.
 - We dealt with missing value by 1) drop the column (`car`) since only 1% data available and 2) drop NULL in the residual features because about 1% is missing and distribution does not change after drop out  
 - Lastly, (`Direction_same`) removed as it shares the same information with (`direction_opp`) column

#### 🔨How to solve imbalance data:
We found some features experience imbalance problem since it is dominated by only one class (`toCoupon_GEQ5min`: All '1') or one of the class contributes to over 80% (`toCoupon_GEQ25min`)
Consequently, we drop those columns out. And responsible result ('Y') seems be fine without imbalance (60/40)

All in all, data set is ...
**Devide `21 Attributes` into 3 groups** 

**Group I. Persona attributes**

 1. **`Age`**: (<21, 21-25, 26-30, 31-35, 36-40, 41-45, 46-50, >50)
 2. **`Gender`**: (Female, Male)
 3. **`MaritalStatus`**: (Unmarried partner, Single, Married partner, Divorced, Widowed)
 4. **`Has_Children`**: (1: Has, 0: Doesn't have)
 5. **`Education`**:  (Some college - no degree, Bachelors degree, Associates degree, High School Graduate, Graduate degree (Masters or Doctorate), Some High School)\
 6. **`Occupation`**: (Unemployed, Architecture & Engineering, Student,Education&Training&Library, Healthcare Support, Healthcare Practitioners & Technical, Sales & Related, Management, Arts Design,Arts Design Entertainment Sports & Media, Computer & Mathematical, Entertainment Sports & Media, Computer & Mathematical,Life Physical Social Science, Personal Care & Service, Community & Social Services, Office & Administrative Support, Construction & Extraction, Legal, Retired,Installation Maintenance & Repair, Transportation & Material Moving,Business & Financial, Protective Service,Food Preparation & Serving Related, Production Occupations,Building & Grounds Cleaning & Maintenance, Farming Fishing & Forestry)
 7. **`Income`**: ( Less than $12500, $12500 - $24999, $25000 - $37499, $37500 - $49999, $50000 - $62499, $62500 - $74999,  $75000 - $87499, $87500 - $99999, $100000 or More)
 8. **`Bar`**: How many times do you go to a bar every month? (never, less1, 1-3, 4-8, gt8, nan)
 9. **`CoffeeHouse`**: How many times do you go to a coffee house every month? (never, less1, 1-3, 4-8, gt8, nan)
 10. **`CarryAway`**: How many times do you get take-away food every month? (never, less1, 1-3, 4-8, gt8, nan)
 11. **`RestaurantLessThan20`**: How many times do you go to a restaurant with an average expense per person of less than $20 every month? (never, less1, 1-3, 4-8, gt8, nan)
 12. **`Restaurant20To50`**: How many times do you go to a restaurant with average expense per person of $20 - $50 every month? (never, less1, 1-3, 4-8, gt8, nan)


**Group II. Coupon attributes**

 13. **`Coupon`**: The coupon for...(Restaurant(<$20), Restaurant($20-$50, Coffee House, Carry out & Take away, Bar)
 14. **`Expiration`**: The coupon expires in 1 day or in 2 hours (1d, 2h)
 
**Group III. Other attributes**

 15. **`Destination`**: (No Urgent Place, Home, Work)
 16. **`Passanger`**: Who are the passengers in the car? (Alone, Friend(s), Kid(s), Partner)
 17. **`Weather`**: (Sunny, Rainy, Snowy)
 18. **`Temperature`**: (30, 50, 80)
 19. **`Time`**: (7AM, 10AM, 2PM, 6PM, 10PM)
 20. **`toCoupon_GEQ15min`**: Driving distance to the restaurant/bar for using the coupon is greater than 15 minutes (1,0)
 21. **`Direction_same`**: Whether the restaurant/bar is in the same direction as your current destination (1,0)

#### ✂️Data splitting (train/val/test):
- `random_state` = 88, 
- `test_size` = 0.25
- **`Train Shape`**: (9059, 73)
- **`Test Shape`**: (3020, 73)


[🔝](https://github.com/lukplamino/DADS7202_HW01_MNLP_Group#highlight)

## 3. Network architecture📦 and Training🔮
We experiment on each hyperparameter with the following default hyperparameter (change each hyperparameter and keep the default for others) and evaluate the result using **`model accuracy`**  on test set

- **`Random state`**: [88, 99, 100, 110]
- **`Number of Hidden layer`**: min value = 3, max value = 5
- **`Number of Units in Hidden layer`**: [32, 64, 128, 256, 512, 1024]
- **`Activation function in Hidden layer`**: [relu, tanh, sigmoid]
- **`Dropout`**: [0.2, 0.25, 0.3]
- **`Learning rate`**: [0.001, 0.0001, 0.00001, 0.00025]
- **`Activation function in Output layer`**: [softmax, sigmoid]
- **`Loss function`**: BinaryCrossentropy
- **`Optimizer`**: [Adam, Nadam, Adamax]
- **`Batch size`**: [64, 128, 256, 512]
- **`Epoch`**: [100, 200, 300, 400, 500, 600, 800, 1,000, 1,300, 1,500, 2,000]


<img src="https://github.com/lukplamino/DADS7202_HW01_MNLP_Group/blob/main/images/Train_models.png" alt="drawing" style="width:1500px;"/>

### Re-train model from the best model (Row 8)
- **`Random state`**: 88
- **`Number of Hidden layer`**: 3
- **`Number of Units in Hidden layer`**: [32, 64, 128]
- **`Activation function in Hidden layer`**: tanh
- **`Dropout`**: 0.25
- **`Learning rate`**: 0.0001
- **`Activation function in Output layer`**: sigmoid
- **`Loss function`**: BinaryCrossentropy
- **`Optimizer`**: Adam
- **`Batch size`**: 128
- **`Epoch`**: 200

<img src="https://github.com/lukplamino/DADS7202_HW01_MNLP_Group/blob/main/images/Model_Summary.png" alt="drawing" style="width:500px;"/>

[🔝](https://github.com/lukplamino/DADS7202_HW01_MNLP_Group#highlight)

## 4. Results📈
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
 
### Traditional Machine Learning (ML)
For training traditional machine learning, we picked **`Random Forest`**, **`Logistic Regression`**, **`Decision Tree`**, **`K Nearest Neighbor`**, and **`linear Support Vector Machine`** for trianing data to compare the results of them. We chose those models due to their simplicity and fast of training.

<img src="https://github.com/lukplamino/DADS7202_HW01_MNLP_Group/blob/main/images/Traditional_model.png" alt="drawing" style="width:500px;"/>

From the table, we can see that the **`Random Forest with Weighted Averages`** has the highest accuracy at **`75.6%`**.

### Multilayer Perceptron (MLP)



[🔝](https://github.com/lukplamino/DADS7202_HW01_MNLP_Group#highlight)


## 5. Discussion💭
_อภิปรายผลลัพธ์ที่ได้ว่ามีอะไรเป็นไปตามสมมติฐาน หรือมีอะไรผิดคาด ไม่เป็นไปตามสมมติฐานบ้าง, วิเคราะห์เพิ่มเติมว่าสิ่งที่ผิดคาดหรือผิดปกตินั้นน่าจะเกิดจากอะไร, ในกรณีที่ dataset มีปัญหา เช่น imalanced dataset ควรวิเคราะห์ด้วยว่าวิธีแก้ที่เราใช้สามารถแก้ปัญหาของ dataset ได้จริงหรือไม่_

[🔝](https://github.com/lukplamino/DADS7202_HW01_MNLP_Group#highlight)

## 6. Conclusion📊
_การอภิปรายผลและการสรุปผล ต้องอ้างอิงกับผลการทดลองของเราที่ได้ออกมาเป็นหลัก
มิใช่การนำข้อสรุปที่เป็น general conclusion จากหนังสือ แบบเรียน หรือจากแหล่งอื่น ๆ 
ในอินเทอร์เน็ต มาเขียนซ้ำโดยไม่มีผลการทดลองใด ๆ ของเรามาช่วยสนับสนุนข้อสรุปดังกล่าว_

_วิเคราะห์ด้วยว่าวิธีแก้ที่เราใช้สามารถแก้ปัญหาของ dataset ได้จริงหรือไม่ หรือจุดประสงค์หลัก (objective) ของการบ้านแต่ละครั้ง_

[🔝](https://github.com/lukplamino/DADS7202_HW01_MNLP_Group#highlight)

## 7. References🌐

### Library
- **`Pipeline`**
- **`SimpleImputer`**
- **`StandardScaler`**
- **`OneHotEncoder`**
- [**`ColumnTransformer`**](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html)
- **`SelectKBest`**
- **`precision_recall_fscore_support`**
- **`LogisticRegression`**
- **`SVC`**
- **`KNeighborsClassifier`**
- **`RandomForestClassifier`**

### Version
<img src="https://github.com/lukplamino/DADS7202_HW01_MNLP_Group/blob/main/images/Version.png" alt="drawing" style="width:400px;"/>

### References
- _ZHENGHAO XIAO. (2021, July 3)_
[**Classification on Categorical Data Part 1**](https://www.kaggle.com/code/iyet1killer/classification-on-categorical-data-part-1-sklearn#Model-Training): Sklearn. Kaggle. 
- _Natdanai, T., Wuthipoom, K., Nuj , L., Krisana, P., Songpol, B., Phawit, B_. (2022, February 6)
[**Powered by The Deep Sleeping Crew**](https://github.com/robinoud/BADS7604_HW3_Deep-Learning?fbclid=IwAR2dfuoK7UWRjvps-sSetnGYrIjDQ6ZzNirOOvJcstEQ30aVtTYlMeuwv0c). Github.


[🔝](https://github.com/lukplamino/DADS7202_HW01_MNLP_Group#highlight)

## Citing: 
[Bib.file](https://github.com/lukplamino/DADS7202_HW01_MNLP_Group/blob/main/Citing_MNLP.bib)

<img src="https://github.com/lukplamino/DADS7202_HW01_MNLP_Group/blob/main/images/citing.png" alt="drawing" style="width:600px;"/>

[🔝](https://github.com/lukplamino/DADS7202_HW01_MNLP_Group#highlight)

## 👥 Members, Percent Contribution and Responsibility 
|No  |ID                |Name                              |% Contribution |Responsibility                             |
|:---:|:---:             |---                               |:---:            |:---|
|1.  |**`6410422002`**  |[Navapol San.](https://www.kaggle.com/navapol)                      |   **`25%`**     |**`Explore data`** **`Prepare dataset`** **`Experiment with traditional ML`** **`Experiment with MLP `**  
|2.  |**`6410422003`**  |[Pakawut Kam.](https://www.kaggle.com/ppakawut)                     |   **`25%`**     |**`Explore data`** **`Prepare dataset`** **`Experiment with traditional ML`** **`Experiment with MLP `** |
|3.  |**`6410422024`**  |[Supisara Poo.](https://www.kaggle.com/supisarapo)                     |   **`25%`**     |**`Explore data`** **`Prepare dataset`** **`Experiment with MLP `** **`Evaluate and conclude result`**  |
|4.  |**`6410422027`**  |[Kantima Tec.](https://www.kaggle.com/kantimatec)                     |   **`25%`**     |**`Explore data`** **`Prepare dataset`** **`Experiment with traditional ML`** **`Evaluate and conclude result`** |

[🔝](https://github.com/lukplamino/DADS7202_HW01_MNLP_Group#highlight)

## 🖇️End Credit 
This project is a part of **`DADS7202 Deep Learning`**

Term: 1 Year of education: 2022

🎓Master of Science Program in **`Data Analytics and Data Science`** (DADS)

🏫National Institute of Development Administration (**`NIDA`**)

<img src="https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252"/> 



[🔝](https://github.com/lukplamino/DADS7202_HW01_MNLP_Group#highlight)
