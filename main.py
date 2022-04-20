# library loading
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score # to split the data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score #To evaluate our model
from dataset import read_dataframe, check_head
from normal_preprocessing import preprocessing
from fairness_preprocessing import f_preprocessing
from fairness_measures import anti_classification_age, anti_classification_gender
from fairness_measures import separation, group_fairness
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
# Algorithmns models to be compared
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fairness", type = int, default = 0)
    parser.add_argument("--model_type", type = str, default = 'XGB')
    parser.add_argument("--test_size", type = float, default = 0.1)
    parser.add_argument("--fmeasure", type = str, default = 'default')
    args = parser.parse_args()

    print(args.fairness, "args.fairness")
    if args.fairness == 0:
        df, labels = preprocessing(read_dataframe())
    else:
        df, labels = f_preprocessing(read_dataframe())


    if args.fmeasure in ['default', 'separation', 'grpfairness'] :
        X = df
    elif args.fmeasure == 'ACG':
        X = anti_classification_gender(df)
    elif args.fmeasure == 'ACA':
        X = anti_classification_age(df)
    else:
        X = df

    y = labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = args.test_size, random_state=42)

    if args.model_type == 'RF':
        model = RandomForestClassifier()
    elif args.model_type == 'LR':
        model = LogisticRegression()
    elif args.model_type == 'KNN':
        model =  KNeighborsClassifier()
    elif args.model_type == 'DT':
        model = DecisionTreeClassifier()
    elif args.model_type == 'NB':
        model = GaussianNB()
    elif args.model_type == 'SVM':
        model = SVC(gamma = 'auto')
    elif args.model_type == 'XGB':
        model = XGBClassifier()
    else:
        model = XGBClassifier()


    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    model_acc = accuracy_score(y_test, predictions)
    roc_score = roc_auc_score(y_test, predictions)
    y_pred_df = pd.DataFrame(predictions, index=X_test.index,columns=["predictions"])
    df_pred=pd.concat([X_test, y_pred_df, y_test], axis=1)
    df_pred_global=df_pred
    msg = "Model %s has: %0.3f accuracy and %0.3f roc_auc_score" % (args.model_type, model_acc, roc_score)
    print(msg)

    if args.fmeasure == 'separation':
        (age_up, age_down), (class_female, class_male) = separation(df_pred_global)
        print(age_up, "age_up", age_down, "age_down")
        print(class_female, "class_female", class_male, "class_male")

    if args.fmeasure == 'grpfairness':
        conf_matrix, (age_up, age_down), (class_female, class_male) = group_fairness(df_pred_global)
        print(conf_matrix, "conf_matrix")
        print(age_up, "age_up", age_down, "age_down")
        print(class_female, "class_female", class_male, "class_male")




