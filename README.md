# Instacart Market Basket Analysis
[Cheng Zeng (Jane)](https://www.linkedin.com/in/chengzeng92/), M.S. Data Science <br/>
George Washington University
## Introduction

With the development of the modern society and new technology, the population is developing a tendency to shop online rather than shop in physical stores. Online shopping means convenience, time saving, cost saving, and many other benefits.During online shopping, customers have more options and can view hundreds of products in a few minutes instead of  having to walk to different areas. Also, customers can also save time by avoiding travel time in between their home and the stores while also avoiding the waiting in a queue for checkout.
According to the report “Consumer buying behavior towards online shopping”, Johnson et al explains that “trade and commerce have been so diversified that multichannel has taken place and online shopping has increased significantly throughout the world in the twenty-first century”.  The e-commerce already constituted of about $2.29 trillion dollars market globally by 2018 due to the double-digit worldwide growth in sales and order. In the past year, e-commerce was boosted by COVID-19. Taylor Soper shows the new data from Adobe, the pandemic brought an additional $183 billion dollars from March 2020 to February 2021. This report also points out the pandemic boosted the online spending by about 20 percent during this period. As a response, a company named Instacart created a website and mobile application to offer a service that allows users to order groceries and personal items from participating retailers.  After the selection from users, Instacart application will review the users’ orders and do the in-store shopping and deliver the order to them. 

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



# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/MiyukiJade/instacart_market_basket_analysis/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
