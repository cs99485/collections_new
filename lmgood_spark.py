#./pyspark --executor-cores 1 --total-executor-cores 1 --conf spark.default.parallelism=1 lmgood_spark.py
from __future__ import division
from pyspark import SparkContext
import os
import time
import datetime
header = ['risk_class' , 'item_family_name', 'item_product_type', 'bill_location_country_cd', 'ship_type','order_type_description' , 'route_description','sales_rep_name','biczincotrm', 'reason_coding' ,'invc_rjctn_sts' ,'invc_rjctn_desc','trx_type', 'amnt_categ', 'product_category']
def reg_alg(customer, total_data, predict_file):
    lm_list = []
    df = DataFrame(total_data,columns = header)
    train_dict = df.T.to_dict().values()
    vec_x_cat_train = vectorizer.fit_transform(train_dict)
    x_train = vec_x_cat_train
    x_train[np.isnan(x_train)]=0
    from sklearn import preprocessing
    le_sex = preprocessing.LabelEncoder()
    df.days_to_pay= le_sex.fit_transform(df.days_to_pay)
    X = vec_x_cat_train
    y = df1.days_to_pay
    from sklearn.linear_model import LinearRegression 
    lm = LinearRegression()
    lm.fit(X, y)   
    lm_list.append([customer]+zip(vectorizer.get_feature_names(), lm.coef_))
    with open(predict_file,"wb") as file:
    writer = csv.writer(file)
    for line in lm_list:
        writer.writerow(line)
        
def store(cust, coeff_map, d, write_header):
    typ ='good'
    f = open(opfile, 'a')
    for k in coeff_map:
        v = coeff_map.get(k)
        f.write(cust + d + k + d + v + d + typ +'\n')                
    f.close()
        
intmfile = 'lmoutput_good_intm.csv'
opfile = 'lmoutput_good_transposed.txt'
try:
    os.remove(predict)
except OSError:
    pass
sc = SparkContext(master='spark://paittvtst03.isus.emc.com:7077',appName='SVM')
sc = SparkContext(appName='GBS')
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')
print 'full-load is running'
data = sc.textFile("ccfulldata.csv",use_unicode=False)
totalgood_data = data.map(lambda s: s.split(',') and s[14] == 'Good').keyBy(lambda x: x[0]).mapValues(lambda x:(x[:])) 
customer_total = totalgood_data.groupByKey().sortByKey()
run_reg_good = customer_total.foreach(lambda x: reg_alg(x[0],list(x[1]),intmfile))
with open(intmfile, 'r') as f:
    lines = f.readlines()
    first = True
    for line in lines:
        cols = line.strip().split(',"')
        cust = cols[0]
        coeff_map = {}
        for i in range(1, len(cols)):
            if '=' in cols[i]:
                kv = cols[i].split('\'')
                if (len(kv)) == 3:
                    k = kv[1]
                    v = kv[2].replace(',', '').replace(')"', '').strip()
                    coeff_map[k] = v
                else:
                    kv = cols[i].split('"')
                    k = kv[2]
                    v = kv[4].replace(',', '').replace(')', '').strip()
                    coeff_map[k] = v
        store(cust, coeff_map, '|', first)
        if first:
            first = False