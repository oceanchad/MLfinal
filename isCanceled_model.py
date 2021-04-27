import pandas as pd
import datetime
import calendar
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


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
    # df_x = full_data_cln.copy()
    df_x = full_data_cln[full_data_cln.is_canceled == False].copy()       # is_canceled preparation
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


"""
choose lead_time, total_of_special_requests, agent, previous_cancellations, customer_type_Transient
try to predict test is_canceled
"""
num_features = ["lead_time", "total_of_special_requests", "previous_cancellations", "agent"]

cat_features = ["customer_type"]

full_data = pd.read_csv("./dataset/train.csv")
full_test_data = pd.read_csv("./dataset/test.csv")

# Separate features and predicted value
features = num_features + cat_features
X = full_data.drop(["is_canceled"], axis=1)[features]
y = full_data["is_canceled"]
test_x = full_test_data[features]
test_y = full_test_data

# preprocess numerical feats:
# for most num cols, except the dates, 0 is the most logical choice as fill value
# and here no dates are missing.
num_transformer = SimpleImputer(strategy="constant")

# Preprocessing for categorical features:
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("onehot", OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing for numerical and categorical features:
preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_features),
                                               ("cat", cat_transformer, cat_features)])

clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=2000, random_state=42,n_jobs=-1))])
clf.fit(X, y)
print("model score: %.3f" % clf.score(X, y))
filename = "is_canceled.joblib"
dump(clf, open(filename, 'wb'))
clf_ = load('is_canceled.joblib')
full_test_data['is_canceled'] = clf_.predict(test_x)
full_test_data.to_csv('./dataset/test_1.csv', index=False) 


"""
main for submit
"""
if __name__ == "__main__":
    
    new_train_x = DealData("./dataset/train.csv", "./dataset/train_label.csv")[0]
    train_y = DealData("./dataset/train.csv", "./dataset/train_label.csv")[1]
    test_x = DealData("./dataset/test_1.csv", "./dataset/test_nolabel.csv")[0]
    test_y = DealData("./dataset/test_1.csv", "./dataset/test_nolabel.csv")[1]

    clf = RandomForestClassifier(n_estimators=300, max_depth=2, random_state=0, oob_score=True)
    clf.fit(new_train_x, train_y['label'])
    clf_score = clf.score(new_train_x, train_y['label'])

    xgbc = XGBClassifier(criterion = 'giny', learning_rate = 0.01, max_depth = 5, n_estimators = 1500,
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
