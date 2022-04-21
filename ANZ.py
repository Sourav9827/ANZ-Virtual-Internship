# -*- coding: utf-8 -*-
"""
Created on Sun May  9 21:22:58 2021

@author: sRv
"""


#Task 1

import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
import statsmodels.api as s
from statsmodels.formula.api import ols
from scipy import stats
from datetime import date
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor,export_graphviz

os.chdir("D:\Temp")
anz = pd.read_excel('ANZ.xlsx')

#Exploratory Data Analysis.

anz.isnull().sum()
anz.drop("bpay_biller_code", axis=1, inplace = True)
anz.drop("merchant_code", axis=1, inplace = True)
anz["month"] = anz["date"].dt.month_name()
anz.describe()
anz.info()
anz.columns
anz["Age_Category"] = ""
for i in range(0,len(anz),1):
    if anz["age"][i] <= 22:
        anz["Age_Category"][i] = "(<22)"
    elif anz["age"][i] <= 28:
        anz["Age_Category"][i] = "(22-28)"
    elif anz["age"][i] <= 38:
        anz["Age_Category"][i] = "(28-38)"
    else:
        anz["Age_Category"][i] = "(>38)" 
anz["status"].value_counts()
anz["card_present_flag"].value_counts()
anz["currency"].value_counts()
anz["txn_description"].value_counts()
anz["gender"].value_counts()
anz["merchant_suburb"].value_counts()
anz["merchant_state"].value_counts()
anz["country"].value_counts()
anz["movement"].value_counts()
anz.corr()
anz.cov()

print("\n Average Transaction Amount is \t $", anz["amount"].mean())

plt.figure(figsize = (5, 5))
df1 = pd.DataFrame(anz.pivot_table(index = "merchant_state", values = ["transaction_id"], aggfunc = "count"))
df1["transaction_id"].plot.pie()
plt.ylabel("")
plt.title("Number Transactions From All States")
sns.set_style("dark")
sns.despine()
plt.show()

plt.figure(figsize = (20, 10))
df2 = pd.DataFrame(anz.pivot_table(index = "month",columns = "merchant_state", values = ["amount"], aggfunc = "mean"))
df2=df2.reindex(['August','September','October'])
df2.plot.line()
plt.title('Trend Avearge Transaction', loc='left') 
plt.ylim(20,100)
sns.set_style("dark")
plt.legend(bbox_to_anchor=(0.75, 1.15), ncol=1)
sns.despine()
plt.show()

plt.figure(figsize = (20, 10))
sns.lineplot(x='age', y='amount', data = anz, ci=68)
plt.title('Transaction trend for different ages',fontsize = 40)
plt.xlabel('Age',fontsize = 20) 
plt.ylabel('Amount',fontsize = 20)  
plt.xlim(18)
plt.ylim(0)
sns.set_style("dark")
sns.despine()
plt.show()

#Task 2

df_sal_trans = anz.loc[(anz['txn_description'] == "PAY/SALARY")]
df_sal_trans.drop(['card_present_flag','merchant_id','merchant_suburb', 'merchant_state','merchant_long_lat'], axis = 1, inplace  = True)
df_sal_sum = pd.DataFrame(df_sal_trans.pivot_table(index = 'customer_id',values = ["amount"], aggfunc = "sum"))
df_sal_counts = pd.DataFrame(df_sal_trans.pivot_table(index = 'customer_id', values = ["amount"], aggfunc = "count"))
df_sal_sum.reset_index(inplace = True)
df_sal_counts.reset_index(inplace = True)
df_sal = df_sal_sum.merge(df_sal_counts, on = 'customer_id')
df_sal.columns
df_sal.rename(columns = {'amount_x':'salary','amount_y':'sal_frequency' }, inplace = True)
df_sal['Annual_Salary'] = df_sal['salary'] * 4

plt.figure(figsize = (20, 10))
sns.distplot(df_sal_trans['balance'])
plt.title("Distribution of customers for annual salary",fontsize = 30)
plt.xlabel('Annual Salary',fontsize = 20) 
plt.ylabel('Amount',fontsize = 20)  
plt.show()

df_sal_trans=df_sal_trans.merge(df_sal, on = "customer_id")
df_sal_trans['sal_frequency'] = df_sal_trans['sal_frequency'] * 4
df_sal_trans.drop(['salary'], axis = 1, inplace = True)
df_sal_date = pd.DataFrame(df_sal_trans.pivot_table(index = 'customer_id',columns = 'month', values = ["amount"], aggfunc = "count", margins = True))
sns.pairplot(df_sal_trans, kind = "scatter")
df_sal_trans.corr()
df_sal_trans.columns

df_sal_1 = pd.DataFrame(df_sal_trans.pivot_table(index = 'customer_id', values = ['status', 'account', 'currency', 'long_lat', 'first_name', 'gender', 'age', 'country', 'movement', 'Age_Category', 'sal_frequency', 'Annual_Salary'], aggfunc = 'max'))
df_sal_2 = pd.DataFrame(df_sal_trans.pivot_table(index = 'customer_id', values ='balance', aggfunc = 'mean'))
df_sal_1.reset_index(inplace = True)
df_sal_2.reset_index(inplace = True)
df_sal = df_sal_1.merge(df_sal_2, on = 'customer_id')
df_sal.columns

df_trans = pd.DataFrame(anz.pivot_table(index = "customer_id", values = "transaction_id", aggfunc = "count"))
df_trans.rename(columns = {'transaction_id':'trans_frequency'}, inplace = True)
df_trans.reset_index(inplace = True)
df_sal = df_sal.merge(df_trans, on = "customer_id")

df_des_freq = pd.DataFrame(anz.pivot_table(index = "customer_id", columns = "txn_description", values = "transaction_id",aggfunc = "count"))
df_des_freq.reset_index(inplace = True)
df_sal = df_sal.merge(df_des_freq, on = "customer_id")
df_sal.rename(columns = {'INTER BANK':'INTER_BANK_freq','PAY/SALARY':'PAY_SALARY_freq', 'PAYMENT':'PAYMENT_freq', 'POS':'POS_freq', 'PHONE BANK':'PHONE_BANK_freq', 'SALES-POS':'SALES_POS_freq'}, inplace = True) 
df_sal.fillna(0, inplace = True)

df_des_sum = pd.DataFrame(anz.pivot_table(index = "customer_id", columns = "txn_description", values = "amount",aggfunc = "sum", margins = True, margins_name = "total_amount"))
df_des_sum.reset_index(inplace = True)
df_sal = df_sal.merge(df_des_sum, on = "customer_id")
df_sal.rename(columns = {'INTER BANK':'INTER_BANK_amount','PAY/SALARY':'PAY_SALARY_amount', 'PAYMENT':'PAYMENT_amount', 'POS':'POS_amount', 'PHONE BANK':'PHONE_BANK_amount', 'SALES-POS':'SALES_POS_amount'}, inplace = True) 

df_mov = pd.DataFrame(anz.pivot_table(index = "customer_id", columns = "movement", values = "transaction_id",aggfunc = "count"))
df_mov.reset_index(inplace = True)
df_sal = df_sal.merge(df_mov, on = "customer_id")
df_sal.fillna(0, inplace = True)

df_card = pd.DataFrame(anz.pivot_table(index = "customer_id", columns = 'card_present_flag', values = "transaction_id",aggfunc = "count"))
df_card.rename(columns = {1.0:'card_used', 0.0:'card_not_used'}, inplace = True)
df_card.reset_index(inplace = True)
df_sal = df_sal.merge(df_card, on = "customer_id")
df_sal.columns


df_sal = pd.get_dummies(df_sal, columns = ["gender"])
cor=df_sal.corr()
sns.pairplot(df_sal, kind = "scatter")
plt.show()

#Predictive Analysis

df_sal.fillna(0, inplace = True)
x = df_sal.drop(columns = ['customer_id', 'Age_Category', 'Annual_Salary', 'account','country', 'currency', 'first_name', 'long_lat', 'movement', 'status'])
y = df_sal['Annual_Salary']
#y = np.log(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
print("\n\n Shapes Of Training And Test Data:\n",x_train.shape, x_test.shape, y_train.shape, y_test.shape)
base_pred = np.mean(y_test)
print("Base Prediction:\t",base_pred)
base_pred = np.repeat(base_pred, len(y_test))
base_root_mean_square_error = np.sqrt(mean_squared_error(y_test, base_pred))
print("RMSE:\t",base_root_mean_square_error)


#OLS Model

lm = ols('Annual_Salary ~ age + sal_frequency + balance + \
         trans_frequency + total_amount + credit + debit + \
        card_used + card_not_used + INTER_BANK_freq + PAYMENT_freq + \
        PHONE_BANK_freq + POS_freq + SALES_POS_freq + \
        INTER_BANK_amount + PAYMENT_amount + PHONE_BANK_amount + POS_amount +\
        SALES_POS_amount + gender_F + gender_M', data = df_sal).fit()
lm.summary()

lm_modified = ols('Annual_Salary ~ balance + total_amount + INTER_BANK_amount + PAYMENT_amount + PHONE_BANK_amount + POS_amount + SALES_POS_amount ', data = df_sal).fit()
lm_modified.summary()


#Linear Regression

lnr = LinearRegression(fit_intercept=True)
lnr_model = lnr.fit(x_train, y_train)
bi = lnr_model.coef_
b0 = lnr_model.intercept_
salary_predictions_lnr_model = lnr.predict(x_test)

lnr_model_mse = mean_squared_error(y_test, salary_predictions_lnr_model)
lnr_model_rmse = np.sqrt(lnr_model_mse)
print("\n\n RMSE Of Linear Regression Model:\t",lnr_model_rmse)

r2_lnr_test = lnr_model.score(x_test, y_test)
r2_lnr_train = lnr_model.score(x_train, y_train)
print("R Squared Value For Test And Train Data Are: \t",r2_lnr_test, r2_lnr_train)

lnr_residuals=y_test-salary_predictions_lnr_model
sns.regplot(salary_predictions_lnr_model, lnr_residuals, scatter=True, fit_reg=True)


#Decision Tree

tree = DecisionTreeRegressor()
tree_model = tree.fit(x_train, y_train)
salary_predictions_tree_model = tree.predict(x_test)

tree_model_mse = mean_squared_error(y_test, salary_predictions_tree_model)
tree_model_rmse = np.sqrt(tree_model_mse)
print("\n\n RMSE Of Tree Regression Model:\t",tree_model_rmse)

"Thank You."















