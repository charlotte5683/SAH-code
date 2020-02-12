# coding: utf-8
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,RepeatedKFold
from sklearn.metrics import precision_score, f1_score,confusion_matrix,accuracy_score,recall_score
from sklearn.naive_bayes import MultinomialNB,GaussianNB



data = pd.read_table('result3.txt')
data = data.T 
data.index = data.index.str.replace('.CEL', '') 
data.head() 
labels = pd.read_table('phenotype.txt', index_col=0)
warnings.filterwarnings('ignore')
labels['Disease'] = labels['Disease'].str.replace('SAH', '1')
labels['Disease'] = labels['Disease'].str.replace('none', '0')
labels['Disease'] = labels['Disease'].astype(int) # 转换数据类型
labels.head()
data = pd.merge(data, labels, left_index=True, right_index=True)
data.head()


def result(data,clf):
    p = 0.0
    a = 0.0
    r = 0.0
    s= 0.0
    f= 0.0
    num=48
    kf=RepeatedKFold(n_splits=2,n_repeats=num, random_state=0)
    test_y_all = []
    res_all = []
    for train_index, test_index in kf.split(data):  	
        train_data = data.iloc[train_index, :]
        test_data = data.iloc[test_index, :]
        dc = clf
        dc.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1]) # 训练
        res = dc.predict(test_data.iloc[:, :-1]) # 预测
        test_y = test_data.iloc[:, -1].tolist()
        res_all.extend(res)
        test_y_all.extend(test_y)
        p += precision_score(test_y, res)    
        a += accuracy_score(test_y, res)   
        r += recall_score(test_y, res)
        m = confusion_matrix(test_y, res)
        s += m[0,0]/(m[0,0]+m[0,1])

    p /= 2*num
    a /= 2*num
    r /= 2*num
    s /= 2*num
    f = 2*p*r/(p+r)

    print('Accuracy:%f' % a)
    print('Percision:%f' % p)
    print('Sensitivity:%f' % r)
    print('Specificity:%f' % s)
    print('f1:%f' % f)


def std(data,clf):
    p = 0.0
    a = 0.0
    r = 0.0
    s = 0.0
    f = 0.0
    num = 48
    kf=RepeatedKFold(n_splits=2, n_repeats=48, random_state=0)
    a_list = []
    p_list = []
    r_list = []
    s_list = []
    f_list = []
    for train_index, test_index in kf.split(data):  	
        train_data = data.iloc[train_index, :]
        test_data = data.iloc[test_index, :]
        dc = clf
        dc.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1]) # 训练
        res = dc.predict(test_data.iloc[:, :-1]) # 预测
        test_y = test_data.iloc[:, -1].tolist()
        a = accuracy_score(test_y, res)
        p = precision_score(test_y, res) 
        r = recall_score(test_y, res)
        f = f1_score(test_y, res)
        m = confusion_matrix(test_y, res)
        s = m[0,0]/(m[0,0]+m[0,1])


        a_list.append(a)
        p_list.append(p)
        r_list.append(r)
        s_list.append(s) 
        f_list.append(f)

    a_s = np.cov(np.array(a_list))
    p_s = np.cov(np.array(p_list))
    r_s = np.cov(np.array(r_list))
    s_s = np.cov(np.array(s_list))
    f_s = np.cov(np.array(f_list))

    print('Accuracy_std:%f' % a_s)
    print('Percision_std:%f' % p_s)
    print('Sensitivity_std:%f' % r_s)
    print('Specificity_std:%f' % s_s)
    print('f1_std:%f' % f_s)


if __name__ == '__main__':
# clf_lr = MLPClassifier(activation='relu',alpha=0.005)
# clf_lr.fit(data.iloc[:, :-1], data.iloc[:, -1])
	clf_lr =LogisticRegression(C=0.07)
	ada_lr = AdaBoostClassifier(clf_lr,n_estimators=20)
	clf_svm=SVC(gamma=30,probability=True) 
	ada_svm = AdaBoostClassifier(clf_svm,n_estimators=20,algorithm="SAMME")
	clf_nb= MultinomialNB(alpha=10)  
	ada_nb= AdaBoostClassifier(clf_nb,n_estimators=20)
#	clf_dt=DecisionTreeClassifier()
#	ada_dt=AdaBoostClassifier(clf_dt,n_estimators=20, learning_rate=0.5)
	voting_clf = VotingClassifier( estimators=[("ada_lr", ada_lr), ("ada_svm", ada_svm), ("ada_nb", ada_nb)],voting='soft') #,weights=[1.2,1,1]

	for clf in [("lr", clf_lr), ("svm", clf_svm), ("NB", clf_nb),('ensemble',voting_clf)]:
	    print(clf[0])
	    result(data,clf[1])

	for clf in [("lr", clf_lr), ("svm", clf_svm), ("NB", clf_nb),('ensemble',voting_clf)]:
   		print(clf[0])
   		std(data,clf[1])
