##############################################################
# CLTV Prediction with BG-NBD & GAMMA GAMMA
##############################################################

# CLTV = (Customer_Value / Churn_Rate) x Profit_margin.
# Customer_Value = Average_Order_Value * Purchase_Frequency

##############################################################
# 1. Data Preparation
##############################################################

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler


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



df_ = pd.read_excel("datasets/online_retail_II.xlsx",
                    sheet_name="Year 2009-2010")
df = df_.copy()

df.head()
"""
  Invoice StockCode                          Description  Quantity         InvoiceDate  Price  Customer ID         Country
0  489434     85048  15CM CHRISTMAS GLASS BALL 20 LIGHTS        12 2009-12-01 07:45:00 6.9500   13085.0000  United Kingdom
1  489434    79323P                   PINK CHERRY LIGHTS        12 2009-12-01 07:45:00 6.7500   13085.0000  United Kingdom
2  489434    79323W                  WHITE CHERRY LIGHTS        12 2009-12-01 07:45:00 6.7500   13085.0000  United Kingdom
3  489434     22041         RECORD FRAME 7" SINGLE SIZE         48 2009-12-01 07:45:00 2.1000   13085.0000  United Kingdom
4  489434     21232       STRAWBERRY CERAMIC TRINKET BOX        24 2009-12-01 07:45:00 1.2500   13085.0000  United Kingdom
"""

#########################
# Data Preprocessing
#########################


df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")


df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11)


#########################
# Lifetimes
#########################

# recency: Time since last purchase. Weekly.
# T: How many days ago the first purchase was made. Weekly.
# frequency: total number of repeated purchases (frequency>1)
# monetary_value:  Average earning per purchase


cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

cltv_df
"""
            InvoiceDate             Invoice TotalPrice
             <lambda_0> <lambda_1> <lambda>   <lambda>
Customer ID                                           
12346.0000          196        726       11   372.8600
12347.0000           37        405        2  1323.3200
12348.0000            0        439        1   222.1600
12349.0000          181        590        3  2295.0200
12351.0000            0        376        1   300.9300
                 ...        ...      ...        ...
18283.0000          275        659        6   641.7700
18284.0000            0        432        1   448.6200
18285.0000            0        661        1   413.9400
18286.0000          247        724        2  1283.3700
18287.0000          188        572        4  2332.6500
[4312 rows x 4 columns]
"""

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']


cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

cltv_df["recency"] = cltv_df["recency"] / 7

cltv_df["T"] = cltv_df["T"] / 7


cltv_df
"""
             recency        T  frequency  monetary
Customer ID                                       
12346.0000   28.0000 103.7143         11   33.8964
12347.0000    5.2857  57.8571          2  661.6600
12349.0000   25.8571  84.2857          3  765.0067
12352.0000    2.2857  56.1429          2  171.9000
12356.0000    6.2857  60.7143          3 1187.4167
              ...      ...        ...       ...
18276.0000   48.0000 104.2857          5  264.1320
18277.0000   13.8571  70.7143          4  256.6650
18283.0000   39.2857  94.1429          6  106.9617
18286.0000   35.2857 103.4286          2  641.6850
18287.0000   26.8571  81.7143          4  583.1625
[2893 rows x 4 columns]
"""

##############################################################
# 2. BG-NBD Model
##############################################################

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])


################################################################
# Who are the 10 customers we expect the most to purchase in a week?
################################################################


bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency'],
                                                        cltv_df['T']).sort_values(ascending=False).head(10)
"""
Customer ID
15989.0000   0.0076
16720.0000   0.0072
14119.0000   0.0072
16204.0000   0.0072
17591.0000   0.0071
15169.0000   0.0070
17193.0000   0.0070
17251.0000   0.0070
17411.0000   0.0069
17530.0000   0.0068
dtype: float64
"""

cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                              cltv_df['frequency'],
                                              cltv_df['recency'],
                                              cltv_df['T'])

cltv_df
"""
             recency        T  frequency  monetary  expected_purc_1_week
Customer ID                                                             
12346.0000   28.0000 103.7143         11   33.8964                0.0000
12347.0000    5.2857  57.8571          2  661.6600                0.0004
12349.0000   25.8571  84.2857          3  765.0067                0.0014
12352.0000    2.2857  56.1429          2  171.9000                0.0002
12356.0000    6.2857  60.7143          3 1187.4167                0.0001
              ...      ...        ...       ...                   ...
18276.0000   48.0000 104.2857          5  264.1320                0.0023
18277.0000   13.8571  70.7143          4  256.6650                0.0002
18283.0000   39.2857  94.1429          6  106.9617                0.0009
18286.0000   35.2857 103.4286          2  641.6850                0.0026
18287.0000   26.8571  81.7143          4  583.1625                0.0010
[2893 rows x 5 columns]
"""

################################################################
# Who are the 10 customers we expect the most to purchase in 1 month?
################################################################


bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)
"""
Customer ID
15989.0000   0.0299
16720.0000   0.0285
14119.0000   0.0284
16204.0000   0.0282
17591.0000   0.0282
15169.0000   0.0276
17193.0000   0.0276
17251.0000   0.0275
17411.0000   0.0273
17530.0000   0.0268
dtype: float64
"""


cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])


cltv_df.sort_values("expected_purc_1_month", ascending=False).head(20)
"""
             recency        T  frequency  monetary  expected_purc_1_week  expected_purc_1_month
Customer ID                                                                                    
15989.0000   51.5714 105.4286          2  186.2550                0.0076                 0.0299
16720.0000   49.2857 103.2857          2  320.1200                0.0072                 0.0285
14119.0000   48.8571 102.7143          2   91.6850                0.0072                 0.0284
16204.0000   50.0000 104.5714          2  269.2650                0.0072                 0.0282
17591.0000   46.1429  98.8571          2  270.8450                0.0071                 0.0282
15169.0000   49.1429 103.8571          2  284.6100                0.0070                 0.0276
17193.0000   50.1429 105.4286          2  305.3700                0.0070                 0.0276
17251.0000   48.8571 103.5714          2  264.5750                0.0070                 0.0275
17411.0000   49.8571 105.2857          2  323.2200                0.0069                 0.0273
17530.0000   52.5714 104.8571          3  252.1700                0.0068                 0.0268
14479.0000   52.4286 104.8571          3  327.6200                0.0067                 0.0265
13807.0000   52.5714 105.1429          3  140.7333                0.0067                 0.0265
15184.0000   48.0000 103.4286          2  141.5950                0.0066                 0.0263
17776.0000   48.0000 103.4286          2  169.2800                0.0066                 0.0263
16596.0000   51.8571 104.2857          3  109.8267                0.0066                 0.0262
13403.0000   47.8571 103.4286          2   77.2500                0.0066                 0.0260
17816.0000   51.1429 103.5714          3   25.2533                0.0065                 0.0257
15713.0000   51.0000 103.4286          3  218.0300                0.0065                 0.0256
14865.0000   52.2857 105.5714          3   25.8400                0.0064                 0.0254
14986.0000   43.8571  97.8571          2  283.4500                0.0064                 0.0254
"""

################################################################
# What is the Expected Number of Sales of the Whole Company in 1 Month?
################################################################


bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()
# 11.608238398752142


################################################################
# What is the Expected Sales Number of the Whole Company in 3 Months?
################################################################


bgf.predict(4 * 3,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()
# 33.450223014701265


################################################################
# Visualization of the prediction results.
################################################################

plot_period_transactions(bgf)
plt.show()

##############################################################
# 3. GAMMA-GAMMA Model
##############################################################


ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])


ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).head(10)
"""
Customer ID
12346.0000     34.7855
12347.0000    726.7538
12349.0000    813.4300
12352.0000    190.3223
12356.0000   1261.8495
12357.0000   6172.3459
12358.0000    958.0482
12359.0000    436.1604
12360.0000    325.9717
12361.0000    115.0311
dtype: float64
"""


ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).sort_values(ascending=False).head(10)
"""
Customer ID
12357.0000   6172.3459
17450.0000   5457.7883
14091.0000   5022.9216
14088.0000   4792.1915
18102.0000   3533.3926
12409.0000   3228.0113
14646.0000   3115.7658
12454.0000   2940.2847
16684.0000   2866.3781
12415.0000   2851.5073
dtype: float64
"""


cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])

cltv_df
"""
             recency        T  frequency  monetary  expected_purc_1_week  expected_purc_1_month  expected_average_profit
Customer ID                                                                                                             
12346.0000   28.0000 103.7143         11   33.8964                0.0000                 0.0000                  34.7855
12347.0000    5.2857  57.8571          2  661.6600                0.0004                 0.0015                 726.7538
12349.0000   25.8571  84.2857          3  765.0067                0.0014                 0.0056                 813.4300
12352.0000    2.2857  56.1429          2  171.9000                0.0002                 0.0007                 190.3223
12356.0000    6.2857  60.7143          3 1187.4167                0.0001                 0.0005                1261.8495
              ...      ...        ...       ...                   ...                    ...                      ...
18276.0000   48.0000 104.2857          5  264.1320                0.0023                 0.0092                 274.4279
18277.0000   13.8571  70.7143          4  256.6650                0.0002                 0.0007                 269.3124
18283.0000   39.2857  94.1429          6  106.9617                0.0009                 0.0036                 110.7961
18286.0000   35.2857 103.4286          2  641.6850                0.0026                 0.0103                 704.8753
18287.0000   26.8571  81.7143          4  583.1625                0.0010                 0.0041                 610.6591
[2893 rows x 7 columns]
"""

##############################################################
# 4. Calculation of CLTV with BG-NBD and GG model.
##############################################################



cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=3,  # 3 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv.head()
"""
Customer ID
12346.0000    0.0002
12347.0000    3.2751
12349.0000   13.7464
12352.0000    0.3763
12356.0000    1.9081
Name: clv, dtype: float64
"""


cltv = cltv.reset_index()
cltv.sort_values(by="clv", ascending=False).head(50)
"""
      Customer ID      clv
67     12497.0000 142.8872
1671   15823.0000 103.1798
83     12557.0000  96.9996
1059   14564.0000  76.9371
6      12358.0000  70.4795
449    13373.0000  69.8940
58     12477.0000  68.8879
2127   16732.0000  66.8987
42     12435.0000  63.6585
221    12873.0000  57.2302
2143   16772.0000  51.9884
399    13253.0000  51.5580
78     12539.0000  51.2560
787    14038.0000  50.4523
2157   16795.0000  49.5489
2708   17857.0000  49.3889
26     12406.0000  49.0067
93     12586.0000  47.1510
16     12377.0000  45.7769
451    13377.0000  44.7087
985    14431.0000  43.8847
1688   15865.0000  43.6524
1741   15978.0000  39.3476
224    12876.0000  39.1203
822    14105.0000  38.7997
1723   15944.0000  37.9455
1303   15067.0000  37.5690
916    14286.0000  37.4592
2318   17107.0000  37.2771
157    12733.0000  37.1580
917    14290.0000  37.0077
775    14014.0000  36.2090
2722   17876.0000  36.1364
834    14130.0000  36.0861
1458   15382.0000  35.6605
1459   15384.0000  35.6283
1907   16281.0000  35.1354
671    13800.0000  35.0209
1310   15077.0000  34.7668
1315   15084.0000  34.6163
358    13154.0000  32.9788
205    12842.0000  31.9943
1113   14679.0000  31.1972
2121   16720.0000  30.9106
2030   16532.0000  30.8892
2186   16854.0000  30.8745
1776   16036.0000  30.8108
737    13946.0000  30.4955
928    14304.0000  30.3534
2512   17478.0000  30.0598
"""

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")

cltv_final.sort_values(by="clv", ascending=False).head(10)
"""
      Customer ID  recency        T  frequency  monetary  expected_purc_1_week  expected_purc_1_month  expected_average_profit      clv
67     12497.0000  34.8571  92.1429          2 2563.8200                0.0042                 0.0166                2810.1798 142.8872
1671   15823.0000  51.1429 104.5714          3 1292.0500                0.0062                 0.0245                1372.9255 103.1798
83     12557.0000  50.1429 103.2857          4 1681.2950                0.0046                 0.0180                1758.7351  96.9996
1059   14564.0000  42.8571 105.1429          2 1302.8100                0.0044                 0.0175                1429.0021  76.9371
6      12358.0000  50.8571 104.5714          3  901.2367                0.0061                 0.0240                 958.0482  70.4795
449    13373.0000  45.2857 104.8571          3 1424.9800                0.0038                 0.0150                1514.0405  69.8940
58     12477.0000  31.8571  89.1429          3 2352.2100                0.0023                 0.0090                2498.3637  68.8879
2127   16732.0000  46.8571 104.5714          3 1175.2767                0.0044                 0.0175                1248.9619  66.8987
42     12435.0000  48.5714 105.4286          4 1460.2225                0.0035                 0.0136                1527.6081  63.6585
221    12873.0000  40.5714  97.4286          2  826.6500                0.0052                 0.0205                 907.4665  57.2302
"""


scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_final[["clv"]])
cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])


cltv_final.sort_values(by="scaled_clv", ascending=False).head()
"""
      Customer ID  recency        T  frequency  monetary  expected_purc_1_week  expected_purc_1_month  expected_average_profit      clv  scaled_clv
67     12497.0000  34.8571  92.1429          2 2563.8200                0.0042                 0.0166                2810.1798 142.8872      1.0000
1671   15823.0000  51.1429 104.5714          3 1292.0500                0.0062                 0.0245                1372.9255 103.1798      0.7221
83     12557.0000  50.1429 103.2857          4 1681.2950                0.0046                 0.0180                1758.7351  96.9996      0.6789
1059   14564.0000  42.8571 105.1429          2 1302.8100                0.0044                 0.0175                1429.0021  76.9371      0.5384
6      12358.0000  50.8571 104.5714          3  901.2367                0.0061                 0.0240                 958.0482  70.4795      0.4933
"""


##############################################################
# 5. Creating Segments according to CLTV.
##############################################################


cltv_final["segment"] = pd.qcut(cltv_final["scaled_clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.head()
"""
   Customer ID  recency        T  frequency  monetary  expected_purc_1_week  expected_purc_1_month  expected_average_profit     clv  scaled_clv segment
0   12346.0000  28.0000 103.7143         11   33.8964                0.0000                 0.0000                  34.7855  0.0002      0.0000       D
1   12347.0000   5.2857  57.8571          2  661.6600                0.0004                 0.0015                 726.7538  3.2751      0.0229       B
2   12349.0000  25.8571  84.2857          3  765.0067                0.0014                 0.0056                 813.4300 13.7464      0.0962       A
3   12352.0000   2.2857  56.1429          2  171.9000                0.0002                 0.0007                 190.3223  0.3763      0.0026       C
4   12356.0000   6.2857  60.7143          3 1187.4167                0.0001                 0.0005                1261.8495  1.9081      0.0134       B
"""

cltv_final.sort_values(by="scaled_clv", ascending=False).head(50)
"""
      Customer ID  recency        T  frequency  monetary  expected_purc_1_week  expected_purc_1_month  expected_average_profit      clv  scaled_clv segment
67     12497.0000  34.8571  92.1429          2 2563.8200                0.0042                 0.0166                2810.1798 142.8872      1.0000       A
1671   15823.0000  51.1429 104.5714          3 1292.0500                0.0062                 0.0245                1372.9255 103.1798      0.7221       A
83     12557.0000  50.1429 103.2857          4 1681.2950                0.0046                 0.0180                1758.7351  96.9996      0.6789       A
1059   14564.0000  42.8571 105.1429          2 1302.8100                0.0044                 0.0175                1429.0021  76.9371      0.5384       A
6      12358.0000  50.8571 104.5714          3  901.2367                0.0061                 0.0240                 958.0482  70.4795      0.4933       A
449    13373.0000  45.2857 104.8571          3 1424.9800                0.0038                 0.0150                1514.0405  69.8940      0.4892       A
58     12477.0000  31.8571  89.1429          3 2352.2100                0.0023                 0.0090                2498.3637  68.8879      0.4821       A
2127   16732.0000  46.8571 104.5714          3 1175.2767                0.0044                 0.0175                1248.9619  66.8987      0.4682       A
42     12435.0000  48.5714 105.4286          4 1460.2225                0.0035                 0.0136                1527.6081  63.6585      0.4455       A
221    12873.0000  40.5714  97.4286          2  826.6500                0.0052                 0.0205                 907.4665  57.2302      0.4005       A
2143   16772.0000  42.5714 103.1429          3 1241.1833                0.0033                 0.0129                1318.9267  51.9884      0.3638       A
399    13253.0000  39.0000 100.5714          2  960.3550                0.0040                 0.0159                1053.9129  51.5580      0.3608       A
78     12539.0000  37.8571 102.4286          3 1882.1333                0.0021                 0.0084                1999.3426  51.2560      0.3587       A
787    14038.0000  39.4286  96.2857          2  754.7500                0.0050                 0.0198                 828.7148  50.4523      0.3531       A
2157   16795.0000  45.4286  97.7143          3  692.5033                0.0056                 0.0220                 736.4623  49.5489      0.3468       A
2708   17857.0000  39.2857  91.7143          2  609.3675                0.0061                 0.0241                 669.4781  49.3889      0.3456       A
26     12406.0000  42.4286  99.7143          3  991.6867                0.0039                 0.0152                1054.0676  49.0067      0.3430       A
93     12586.0000  49.7143 103.4286          3  621.9100                0.0059                 0.0232                 661.5223  47.1510      0.3300       A
16     12377.0000  29.7143  84.2857          2  896.0400                0.0039                 0.0152                 983.4690  45.7769      0.3204       A
451    13377.0000  35.5714  92.2857          2  758.8750                0.0044                 0.0175                 833.2329  44.7087      0.3129       A
985    14431.0000  44.0000  98.2857          3  714.6133                0.0048                 0.0189                 759.9337  43.8847      0.3071       A
1688   15865.0000  41.4286 100.5714          2  676.8950                0.0048                 0.0191                 743.4407  43.6524      0.3055       A
1741   15978.0000  30.1429  92.1429          2 1090.3200                0.0027                 0.0107                1196.2629  39.3476      0.2754       A
224    12876.0000  27.8571  84.4286          2  931.8850                0.0032                 0.0125                1022.7299  39.1203      0.2738       A
822    14105.0000  44.5714 103.1429          4 1180.6500                0.0026                 0.0103                1235.3205  38.7997      0.2715       A
1723   15944.0000  47.4286 103.8571          3  613.0467                0.0048                 0.0190                 652.1132  37.9455      0.2656       A
1303   15067.0000  27.5714  86.8571          2 1045.0550                0.0027                 0.0107                1146.6844  37.5690      0.2629       A
916    14286.0000  47.2857 105.1429          4  962.7525                0.0031                 0.0121                1007.5130  37.4592      0.2622       A
2318   17107.0000  39.8571  96.8571          4 1314.9975                0.0023                 0.0089                1375.7782  37.2771      0.2609       A
157    12733.0000  42.8571  98.5714          3  680.3067                0.0043                 0.0168                 723.5147  37.1580      0.2601       A
917    14290.0000  45.4286 104.5714          3  733.9700                0.0039                 0.0155                 780.4823  37.0077      0.2590       A
775    14014.0000  28.5714  82.4286          2  721.5150                0.0038                 0.0150                 792.3127  36.2090      0.2534       A
2722   17876.0000  39.2857  96.5714          3  835.3050                0.0034                 0.0133                 888.0569  36.1364      0.2529       A
834    14130.0000  50.4286 104.1429          3  466.2700                0.0060                 0.0237                 496.2989  36.0861      0.2525       A
1458   15382.0000  51.4286 105.1429          6 1296.3067                0.0022                 0.0088                1335.6638  35.6605      0.2496       A
1459   15384.0000  30.0000  84.7143          2  692.6900                0.0039                 0.0153                 760.7408  35.6283      0.2493       A
1907   16281.0000  36.8571  97.5714          2  680.9800                0.0039                 0.0153                 747.9149  35.1354      0.2459       A
671    13800.0000  43.7143 105.2857          2  560.6200                0.0047                 0.0185                 616.0853  35.0209      0.2451       A
1310   15077.0000  33.5714 104.4286          2 1214.0400                0.0021                 0.0085                1331.7728  34.7668      0.2433       A
1315   15084.0000  38.0000  92.2857          2  482.6000                0.0054                 0.0213                 530.6304  34.6163      0.2423       A
358    13154.0000  48.0000 105.1429          3  541.6133                0.0047                 0.0186                 576.2815  32.9788      0.2308       A
205    12842.0000  47.5714 105.4286          2  399.5000                0.0060                 0.0236                 439.6114  31.9943      0.2239       A
1113   14679.0000  49.8571 104.5714          3  429.8133                0.0056                 0.0222                 457.5975  31.1972      0.2183       A
2121   16720.0000  49.2857 103.2857          2  320.1200                0.0072                 0.0285                 352.6669  30.9106      0.2163       A
2030   16532.0000  27.0000  88.7143          2 1003.9200                0.0023                 0.0092                1101.6294  30.8892      0.2162       A
2186   16854.0000  31.8571  91.5714          2  704.4800                0.0033                 0.0130                 773.6544  30.8745      0.2161       A
1776   16036.0000  28.0000  88.7143          2  898.9600                0.0026                 0.0102                 986.6673  30.8108      0.2156       A
737    13946.0000  51.2857 105.1429          3  387.4183                0.0061                 0.0241                 412.5921  30.4955      0.2134       A
928    14304.0000  27.1429  81.5714          2  670.7800                0.0034                 0.0135                 736.7429  30.3534      0.2124       A
2512   17478.0000  32.1429  90.5714          2  636.4000                0.0036                 0.0140                 699.0867  30.0598      0.2104       A

"""

cltv_final.groupby("segment").agg(
    {"count", "mean", "sum"})
"""
        Customer ID                          recency                        T                    frequency               monetary                      expected_purc_1_week               expected_purc_1_month               expected_average_profit                        clv                    scaled_clv               
              count       mean           sum   count    mean        sum count    mean        sum     count    mean   sum    count     mean         sum                count   mean    sum                 count   mean    sum                   count     mean         sum count    mean        sum      count   mean     sum
segment                                                                                                                                                                                                                                                                                                                      
D               724 15506.7265 11226870.0000     724 22.7170 16447.1429   724 87.9594 63682.5714       724 11.6064  8403      724 326.9102 236682.9841                  724 0.0000 0.0252                   724 0.0001 0.0991                     724 338.0324 244735.4256   724  0.0819    59.2836        724 0.0006  0.4149
C               723 15325.2503 11080156.0000     723 23.3592 16888.7143   723 84.9972 61453.0000       723  5.0913  3681      723 333.4657 241095.7126                  723 0.0003 0.1993                   723 0.0011 0.7830                     723 351.7963 254348.7239   723  0.7924   572.9409        723 0.0055  4.0097
B               723 15349.1895 11097464.0000     723 30.2146 21845.1429   723 89.8484 64960.4286       723  4.4191  3195      723 368.0842 266124.8549                  723 0.0010 0.6958                   723 0.0038 2.7373                     723 389.1784 281376.0014   723  3.1319  2264.3624        723 0.0219 15.8472
A               723 15222.0069 11005511.0000     723 37.8453 27362.1429   723 95.3408 68931.4286       723  3.4786  2515      723 468.3474 338615.1853                  723 0.0028 2.0273                   723 0.0110 7.9888                     723 498.2589 360241.1576   723 14.4013 10412.1718        723 0.1008 72.8698
"""


##############################################################
# 6. Function of the project
##############################################################

def create_cltv_p(dataframe, month=3):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    today_date = dt.datetime(2011, 12, 11)

    cltv_df = dataframe.groupby('Customer ID').agg(
        {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                         lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
         'Invoice': lambda Invoice: Invoice.nunique(),
         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

    cltv_df.columns = cltv_df.columns.droplevel(0)
    cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
    cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
    cltv_df["recency"] = cltv_df["recency"] / 7
    cltv_df["T"] = cltv_df["T"] / 7

    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T'])

    cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                                  cltv_df['frequency'],
                                                  cltv_df['recency'],
                                                  cltv_df['T'])

    cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])

    cltv_df["expected_purc_3_month"] = bgf.predict(12,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])

    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
    cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                                 cltv_df['monetary'])

    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency'],
                                       cltv_df['T'],
                                       cltv_df['monetary'],
                                       time=month,  # 3 aylık
                                       freq="W",  # T'nin frekans bilgisi.
                                       discount_rate=0.01)

    cltv = cltv.reset_index()
    cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
    cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

    return cltv_final

df = df_.copy()


cltv_final2 = create_cltv_p(df)

