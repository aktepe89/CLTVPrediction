# pip install lifetimes
# pip install sqlalchemy
from sqlalchemy import create_engine
import mysql.connector
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler


#Pandas settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


df_ = pd.read_excel('Datasets/online_retail_II.xlsx', sheet_name="Year 2010-2011")
df_.head()
df = df_.copy()


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#Read dataframe
df_ = pd.read_excel('Datasets/online_retail_II.xlsx', sheet_name="Year 2010-2011")
df_.head()
df = df_.copy()

#Selecting UK customers only to reduce processing time
df = df[df["Country"] == "United Kingdom"]

#Data pre-processing
df.describe().T

df.shape

#DropNAs, remove transactions with ID that contains "C" which implies the product is returned
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]

#Remove rows where quantity and price is 0
df = df[df['Price'] > 0]
df = df[df['Quantity'] > 0]

#Replace outliers with threshold values
replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

#Calculate TotalPrice
df["TotalPrice"] = df["Quantity"] * df["Price"]

#Determine a date to apply CLTV
today_date = dt.date(2011, 12, 11)

#Group the df by customer ID and add recency, tenure, frequency, monetary columns
cltv_df = df.groupby("Customer ID").agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (pd.Timestamp(today_date) - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)

#Rename the columns
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

#Get avg. monetary value each customer provides for each purchase
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
cltv_df = cltv_df[cltv_df["monetary"] > 0]

#Change recency and tenure into weeks
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7

cltv_df = cltv_df[cltv_df["frequency"] > 1]

#Create BG-NBD Model
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

#1 month expected purchase
cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])

#1 week expected purchase
cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])



cltv_df.head(20).sort_values(by='expected_purc_6_month', ascending=False)


#Gamma gamma
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

cltv_df['expected_avg_profit'] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                         cltv_df['monetary'])

#CLTV
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6, #6 months
                                   freq="W",
                                   discount_rate=0.01)

cltv.head()
cltv = cltv.reset_index()

cltv.sort_values(by="clv", ascending=False).head(10)

cltv_6months = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_6months.sort_values(by="clv", ascending=False)

#6 months
cltv_6months.head(20)


#1 Months CLTV
cltv_1Month = ggf.customer_lifetime_value(bgf,cltv_df['frequency'],
                                          cltv_df['recency'],
                                          cltv_df['T'],
                                          cltv_df['monetary'],
                                          time=1, #1 month,
                                          freq="W",
                                          discount_rate=0.01)
cltv_1Month = cltv_1Month.reset_index()
cltv_1Month.sort_values(by="clv", ascending=False)
cltv_1Month.shape


#12 Months CLTV
cltv_12Month = ggf.customer_lifetime_value(bgf,cltv_df['frequency'],
                                           cltv_df['recency'],
                                           cltv_df['T'],
                                           cltv_df['monetary'],
                                           time=12, #12 month,
                                           freq="W",
                                           discount_rate=0.01)

cltv_12Month = cltv_12Month.reset_index()
cltv_12Month.sort_values(by="clv", ascending=False)
cltv_12Month.shape

cltv_1Month.columns = ['Customer ID', 'clv_1month']
cltv_12Month.columns = ['Customer ID', 'clv_12month']

#combining results for 1 and 12 months
cltv_combined = cltv_1Month.merge(cltv_12Month, on="Customer ID", how="left")


cltv_combined.head(10).sort_values(by="clv_1month", ascending=False)
cltv_combined.head(10).sort_values(by="clv_12month", ascending=False)

#Same customers, but CLV values are much bigger for 12 month analysis - almost 11 times bigger.

#Let's create 5 segments based on customers' CLTV scores. 
#Use MinMaxScaler to scale the cltv_6months scores
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_6months[["clv"]])
cltv_6months["scaled_clv"] = scaler.transform(cltv_6months[["clv"]])

#Use QCut to divide it into 5 parts and label them.
cltv_6months["segment"] = pd.qcut(cltv_6months['scaled_clv'], 4, labels=['D', 'C', 'B', 'A'])




cltv_6months.sort_values(by="scaled_clv", ascending=False).head(10)


cltv_6months.groupby("segment").agg({"count", "mean", "sum"})

