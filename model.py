import pandas as pd
import datetime
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def DealData(data, data_label):
    data_x = pd.read_csv(data)
    data_y = pd.read_csv(data_label)
    """
    "meal" contains values "Undefined", which is equal to SC.
    "agent" If no agency is given, booking was most likely made without one.
    "company" If none given, it was most likely private.
    Some rows contain entreis with 0 adults, 0 children and 0 babies. 
    """
    nan_replacements = {"children:": 0.0,"country": "Unknown", "agent": 0, "company": 0}
    full_data_cln = data_x.fillna(nan_replacements)
    full_data_cln["meal"].replace("Undefined", "SC", inplace=True)
    zero_guests = list(full_data_cln.loc[full_data_cln["adults"]
                + full_data_cln["children"]
                + full_data_cln["babies"]==0].index)
    full_data_cln.drop(full_data_cln.index[zero_guests], inplace=True)
    df_x = full_data_cln.copy()
    # df_x = full_data_cln[full_data_cln.is_canceled == False].copy()       # is_canceled preparation
    df_x['arrival_date'] = df_x['arrival_date_year'].astype(str) + '-' + pd.to_datetime(df_x['arrival_date_month'], format='%B').dt.month.astype(str).apply(lambda x: x.zfill(2)) + '-' + df_x['arrival_date_day_of_month'].astype(str).apply(lambda x: x.zfill(2))
    
    # df_x['revenue'] = df_x.adr*(df_x.stays_in_week_nights+df_x.stays_in_weekend_nights)
    # day_revenue = df_x.groupby('arrival_date')['revenue'].sum()
    # day_revenue = day_revenue.to_frame()
    # day_revenue = day_revenue.reset_index()
    # day_revenue_with_label = data_y.merge(day_revenue, on='arrival_date')
    df_new_x = df_x.groupby('arrival_date')['ID'].count()
    df_new_x = df_new_x.to_frame()
    
    df_x['stay_nights'] = df_x['stays_in_weekend_nights']+df_x['stays_in_week_nights']
    df_new_x['stay_nights_sum'] = df_x.groupby('arrival_date')['stay_nights'].sum()
    df_new_x['stays_in_weekend_nights_sum'] = df_x.groupby('arrival_date')['stays_in_weekend_nights'].sum()
    df_new_x['stays_in_week_nights_sum'] = df_x.groupby('arrival_date')['stays_in_week_nights'].sum()
    df_new_x['month'] = pd.to_datetime(df_new_x.index).month
    # df_x['reserved_room_type'].value_counts()
    return df_new_x, data_y




if __name__ == "__main__":
    new_train_x = DealData("./dataset/train.csv", "./dataset/train_label.csv")[0]
    train_y = DealData("./dataset/train.csv", "./dataset/train_label.csv")[1]
    test_x = DealData("./dataset/test.csv", "./dataset/test_nolabel.csv")[0]
    test_y = DealData("./dataset/test.csv", "./dataset/test_nolabel.csv")[1]

    clf = RandomForestClassifier(n_estimators=300, max_depth=2, random_state=0, oob_score=True)
    clf.fit(new_train_x, train_y['label'])
    clf_score = clf.score(new_train_x, train_y['label'])

    xgbc = XGBClassifier(criterion = 'giny', learning_rate = 0.01, max_depth = 5, n_estimators = 1000,
                          objective ='binary:logistic', subsample = 1.0)
    xgbc.fit(new_train_x, train_y['label'])
    xgbc_score = xgbc.score(new_train_x, train_y['label'])

    print(clf_score, xgbc_score)
    filename = "job.joblib"
    dump(xgbc, open(filename, 'wb'))

    from joblib import dump, load
    xgbc = load('job.joblib')
    test_y['label'] = xgbc.predict(test_x)
    test_y.to_csv('baseline_0106.csv', index=False) 



