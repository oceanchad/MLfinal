# NTU, Fall 2020, Machine Learning Techniques, Final Project

## Introduction
A hotel
booking company tries to predict the daily revenue of the company from the data of reservation.
The goal of the prediction is to accurately infer the future daily revenue of the company,
where the daily revenue is quantized to 10 scales.<p>[**More details**](https://www.csie.ntu.edu.tw/~htlin/course/ml20fall/project/project.pdf)

## Data Set
The data sets are processed from the [Kaggle hotel booking demand data.](https://www.kaggle.com/jessemostipak/hotel-booking-demand)


## EDA 
1. missing value
   1. children / country / agent / company (with nan) -> 4 / 468 / 13217 / 85917
   2. Meal (‘Undefined’) -> Undefined / SC are same
2. Invalid record (action: should drop)
   1. zero_guests (adults / children / babies) -> (0 / 0 / 0) but still have record
   2. is_canceled 32760/91531 (try to drop / 有取消的32760)
3. is_canceled (58771 未取消)

## Model traininig
1. is_canceled model buiding
   1. Random Forest
   2. XGBC
   3. Decision Tree
2. adr moder building
   1. XGBC

