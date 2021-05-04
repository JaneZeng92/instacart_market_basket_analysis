# Instacart Market Basket Analysis
[Cheng Zeng (Jane)](https://www.linkedin.com/in/chengzeng92/), M.S. Data Science <br/>
George Washington University
## Introduction

With the development of the modern society and new technology, the population is developing a tendency to shop online rather than shop in physical stores. Online shopping means convenience, time saving, cost saving, and many other benefits.During online shopping, customers have more options and can view hundreds of products in a few minutes instead of  having to walk to different areas. Also, customers can also save time by avoiding travel time in between their home and the stores while also avoiding the waiting in a queue for checkout.
As a response, a company named [Instacart](https://www.instacart.com/) created a website and mobile application to offer a service that allows users to order groceries and personal items from participating retailers.  After the selection from users, Instacart application will review the users’ orders and do the in-store shopping and deliver the order to them. 

### Purpose & Goals:
The goal of this project is to use transactional data to develop models that predict which product a user will buy again, try for the first time, or add to their cart during a future session. The dataset we use is from [Kaggle](https://www.kaggle.com/c/instacart-market-basket-analysis), and it includes 3 million Instacart orders.

## Methodology
### Tools & Software & Mehtodology:
Python 3.6 package <br/>
PCA/K-mean <br/>
XGBoost


## Data Analysis
### Data Preprocessing and EDA

[Link to notebook](https://github.com/JaneZeng92/instacart_market_basket_analysis/blob/main/EDA.py)

Simply reviewed the data information and found out the files contain some “N/A” columns. However, we did not drop or replace any “N/A” columns so we can receive more accurate test result. 
Before the exploratory analysis, we read all the files as dataframe objects and found out there are several “N/A” columns. However, these “N/A” columns do not affect our data exploratory for now. In order to figure out the relationship in “order_products_prior”, “aisles”, “departments” and “orders” file, we merge these details together.

The order was highly concentrated in between 8 A.M to 6 P.M as shown by the [Figure](https://github.com/JaneZeng92/instacart_market_basket_analysis/blob/main/EDA/Frequency%20of%20Order%20Hours%20of%20Day.png). There is a clear ordering habit changes with the order by day of week. In addition, the frequency of order by weekday showed us that the most orders are on days 0 and 1 by [Figure](https://github.com/JaneZeng92/instacart_market_basket_analysis/blob/main/EDA/Frequency%20of%20Order%20by%20Week%20Day.png). There is no information regarding which value represent which day, but we assume the 0 is Saturday and 1 is Sunday. Therefore, we could say that customers have high chance of making an order during the weekend. Wednesday is the day with the lowest change of an order.Combining the day of the week and hour of the day, we can see the distribution clearer. The [figure](https://github.com/JaneZeng92/instacart_market_basket_analysis/blob/main/EDA/Frequency%20of%20Day%20of%20week%20Vs%20Hour%20of%20day.png) shows that Saturday evenings and Sunday mornings are the prime time for orders. 
As we mention in the data preprocessing section, we merged 4 datasets together. Thus, we used “value_count” to group up the same type of product (aisles) and department. From the [figure](https://github.com/JaneZeng92/instacart_market_basket_analysis/blob/main/EDA/Aisle%20Count.png) and [figure](https://github.com/JaneZeng92/instacart_market_basket_analysis/blob/main/EDA/Departments%20Distribution.png), we can observe that “fresh fruit” and “fresh vegetables” are the best-selling goods, and “produce” is the most ordered department. 

### PCA/K-mean
[Link to notebook](https://github.com/JaneZeng92/instacart_market_basket_analysis/blob/main/PCA.py)

With the [Elbow method], we can reduce the components to 6, which can explain 72% of the variability of the data. Once we reduced the dimensionality, we can build the K-mean model with 6 clusters. With the K-mean predict function, we found out the closest cluster for the merge dataset. Because we found out a possible cluster for customers, we can see the pattern that we have in the dataset. In the 6 components, we can confirm that the top 5 products that customer purchases are fresh fruits, fresh vegetables, packaged vegetables fruits, yogurt, and packaged cheeses. Therefore, we can predict these 5 products are the most popular one and people will reorder them.

### XGBoost
[Link to notebook](https://github.com/JaneZeng92/instacart_market_basket_analysis/blob/main/XGBoost.py)

XGboost has two different forms in python. One is [XGBClassifier](https://xgboost.readthedocs.io/en/latest/python/python_api.html), a sklearn wrapper for XGBoost; another one is [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), also a sklearn wrapper.

To calculate the total number of orders for each user and the reorder ratio of them, we grouped up the user_id and count the reorder times. Same as the product order time of each products. These two calculations can describe the frequency of a customer purchase a product and the frequency of a product being purchase.After the reorder ratio for each user and each product has been calculated, we can determine the probability of reorder by using the purchase of each user for each product divide by the total orders after the first order. Therefore, we can observe that the number of purchases for each customer bought each product; and from there we can measure the probability of reorder for each user of each product.Combining the order dataset with the updated dataset, we can easily to see the relationship in user, product, and reorder ratio. 
Grid Search with Gross-Validation as known as GridSearchCV. It is “is a brute force on finding the best hyperparameters for a specific dataset and model.”  The parameter that we use are two lists, so we can try to find the best score and best parameter for the model. The result we got is max_depth is 5 and colsample_bytree is 0.3 are the best parameter. Therefore, we use this model to predict. For the better result, the threshold we set to 0.2. Finally, we save the prediction result into the merge dataframe, and keep the column that we need (product_id, order_id and prediction). As the result, we can easily to see the relationship between order_id and products_id.

## Conclusion

This project used PCA, K-mean and XGBoost to predict the users’ order behavior. By comparing the different algorithms, GridSearchCV have better result, accuracy, and detail than PCA and K-mean. With XGBoost, we can better avoid overfitting or misclassification problem. However, XGBoost requires a good computing resource and longer training time if this is implemented on a personal computer. Finally, this prediction can be applied to similar shopping applications. Companies can easily make marketing decision to improve their customer’s satisfaction and brining more customers with this prediction.



