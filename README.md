# Machine-Learning-Fairness

### Title  Individual Assignment 4 : Fairness

### Course 11-695/17-445/17-645 Machine Learning in Production / AI Engineering

### Authors Tushar Vatsa(andrewid : tvatsa), Ayush Agarwal(andrewid : ayushag2)

### Created 19th April, 2022

### Reference https://www.kaggle.com/datasets/uciml/german-credit/code

Arguments list:
  > fairness : 0 means Normal Preprocessing; 1 means Preprocessing by mitigating the bias
  
  > model_type : {'RF' : Random Forest Classifier, 'LR' : Logistic Regression, 'KNN' : K Nearest Neighbors,
  >               'DT' : Decision Tree Classifier, 'NB' : GaussianNB, 'SVM' : SVC(gamma = 'auto'), 'XGB' : XGBClassifier }
  
  >test_size : The size of the test dataset the user wants to choose

  >f_measure : {'ACA' : Anti-Classification Age, 'ACG' : Anti-Classification Gender, 'separation' : Separation measure, 'grpfairness : Group Fairness}

Steps to run:
 > git clone https://github.com/tusharvatsa32/Machine-Learning-Fairness.git
 
 > cd Machine-Learning-Fairness
 
 > python3 setup.py
 
