# Assignment_3

### What is it about?
This course project is a reproduce of <br>
[Xu, Chen, Y., Xiao, T., Wang, J., & Wang, X. (2021). Predicting the trend of stock index based on feature engineering and CatBoost model. International Journal of Financial Engineering, 8(2)](https://doi.org/10.1142/S2424786321500274)
### Abstract
As an important tool to measure the current situation of the whole stock market, the stock index has always been the focus of researchers, especially for its prediction. This paper uses trend types, which are received by clustering price series under multiple time scale, combined with the day-of-the-week effect to construct a categorical feature combination. Based on the historical data of six kinds of Chinese stock indexes, the CatBoost model is used for training and predicting. Experimental results show that the out-of-sample prediction accuracy is 0.55, and the long–short trading strategy can obtain average annualized return of 34.43%, which is a great improvement compared with other classical classification algorithms. Under the rolling back-testing, the model can always obtain stable returns in each period of time from 2012 to 2020. Among them, the SSESC’s long–short strategy has the best performance with an annualized return of 40.85% and a sharp ratio of 1.53. Therefore, the trend information on multiple time-scale features based on feature engineering can be learned by the CatBoost model well, which has a guiding effect on predicting stock index trends.

### Get started
This project is based on Python 
1. conda create -n assign3 python=3.7
2. conda activate assign3
   pip install -r requirements.txt

### About Catboost
CatBoost is a machine learning method based on gradient boosting over decision trees. Main advantages of Catboosts are:
1. Superior quality when compared with other GBDT libraries on many datasets.
2. Best in class prediction speed.
3. Support for both numerical and categorical features.
4. Fast GPU and multi-GPU support for training out of the box.
5. Visualization tools included.
6. Fast and reproducible distributed training with Apache Spark and CLI.

You can find further information about Catboost and its documentation [here] (https://github.com/catboost/catboost)

### My contribution and improvement
Different from classifying of the shapes of stock index trend as the paper introduced. I applied the Catboost algorithm to predict the market intraday momentum. For more information about market intraday momentum, you can refer to: 

'
 [Gao, Han, Y., Zhengzi Li, S., & Zhou, G. (2018). Market intraday momentum. Journal of Financial Economics, 129(2), 394–414.](https://doi.org/10.1016/j.jfineco.2018.05.009)
'

1. First I split one day to several periods and calculate returns of each period
2. Find the relationship of returns of these periods and apply Catboost to predict the future returns.

### Data sets
Share Price Index Futures used in this assignment:
1. IH: SPIF of SSE 50
2. IC: SPIF of CSI 500
3. IF: SPIF of CSI 300

### Conclusions
By applying the Catboost, the backtest result can be improved, the following pictures show the backtest result In-Sample and Out-of-Sample:
#### In-Sample  (Symbol: IC)
##### Applying the Catboost algorithm:
![image](https://github.com/algo23-yifeizhou/Assignment_3/assets/125112527/525245a1-0c48-4327-97a6-0b23347c4397)

##### Traditional methods:
![image](https://github.com/algo23-yifeizhou/Assignment_3/assets/125112527/46e37965-31b8-4539-b006-7ee97a8ba3b4)

#### Out-of-Sample  (Symbol: IC)
##### Applying the Catboost algorithm:
![image](https://github.com/algo23-yifeizhou/Assignment_3/assets/125112527/319eeb76-9a0f-4910-b5b8-695e7c48f067)

##### Traditional methods:
![image](https://github.com/algo23-yifeizhou/Assignment_3/assets/125112527/8733b973-cf61-4049-95d4-7bb18027c91e)

#### Training result
##### Accuracy：
![image](https://github.com/algo23-yifeizhou/Assignment_3/assets/125112527/e886376f-5b30-4879-9c5f-c422d0888d16)

##### Logloss
![image](https://github.com/algo23-yifeizhou/Assignment_3/assets/125112527/e3c47771-c1c0-42c5-8da2-406c2de39c3c)

#### Cross Validation
##### Accuracy:
![image](https://github.com/algo23-yifeizhou/Assignment_3/assets/125112527/b0111e52-0286-437c-8f59-98b25e4178bd)

##### Logloss:
![image](https://github.com/algo23-yifeizhou/Assignment_3/assets/125112527/cb56ac84-a591-4063-9ac6-67f94434d507)

### Discussion
Still fresh to applying machine learning to predicting asset prices, much work need to be done in the future:
1. The parameters need to be testified and be chosen to get Generalized for Out-of-Sample data sets.
2. There is much to be done in Feature Engineering. In this project, I only used index future returns as features, and there is more information (eg. volume, volatility, the underlying index etc.) injecting to the market every day
3. Further usage of Catboost' s Superior quality and on other assets
