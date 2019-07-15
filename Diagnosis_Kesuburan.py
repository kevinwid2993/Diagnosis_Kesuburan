import numpy as np 
import pandas as pd 

data = pd.read_csv('fertility.csv')

#labeling
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
data['Childish diseases'] = label.fit_transform(data['Childish diseases'])
#print(label.classes_) ['no' 'yes']
data['Accident or serious trauma'] = label.fit_transform(data['Accident or serious trauma'])
#print(label.classes_) ['no' 'yes']
data['Surgical intervention'] = label.fit_transform(data['Surgical intervention'])
#print(label.classes_) ['no' 'yes']
data['High fevers in the last year'] = label.fit_transform(data['High fevers in the last year'])
#print(label.classes_) ['less than 3 months ago' 'more than 3 months ago' 'no']
data['Frequency of alcohol consumption'] = label.fit_transform(data['Frequency of alcohol consumption'])
#print(label.classes_) ['every day' 'hardly ever or never' 'once a week' 'several times a day' 'several times a week']
data['Smoking habit'] = label.fit_transform(data['Smoking habit'])
#print(label.classes_) ['daily' 'never' 'occasional']

data = data.drop(['Season'],axis = 1)

print(data)
x = data.drop(['Diagnosis'],axis = 1)
y = data['Diagnosis']

#from one hot encoder
from sklearn.preprocessing import OneHotEncoder #yang dilabelin jadi maju paling pertama
from sklearn.compose import ColumnTransformer

coltrans = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'),[4,5,6])],
    remainder='passthrough'
)

x = np.array(coltrans.fit_transform(x),dtype=np.float64)
#print(x[0])
'''
HV<3 >3  no  fac nev 1/w s/d s/w smk nvr occ age CD  AC/ SI  dduk 
             all                 dly                 ST
[ 0.  1.  0.  0.  0.  1.  0.  0.  0.  0.  1. 30.  0.  1.  1. 16.]
'''

#splitting
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(
    x,
    y,
    test_size = .1
)

#ml logistic
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(solver='liblinear')
logmodel.fit(xtrain,ytrain)
print(round(logmodel.score(xtest,ytest)*100,2),'%')

#knn
from sklearn.neighbors import KNeighborsClassifier
knnmodel = KNeighborsClassifier(n_neighbors=10)
knnmodel.fit(xtrain,ytrain)
print(round(knnmodel.score(xtest,ytest)*100,2),'%')

#svm
from sklearn.svm import SVC
svcmodel = SVC()
svcmodel.fit(xtrain,ytrain)
print(round(svcmodel.score(xtest,ytest)*100,2),'%')

'''
HF<3 >3  no  fac nev 1/w s/d s/w smk nvr occ age CD  AC/ SI  dduk 
            all/d                dly                 ST

HF= HIGH FEVER
FAC= ALCOHOL
SMK= MEROKOK
age
child disease
accident
surgical intervention
duduk
'''

arin = [0,0,1,1,0,0,0,0,1,0,0,29,0,0,0,5]
bebi = [0,0,1,0,0,0,0,1,0,1,0,31,0,1,1,16]
caca = [1,0,0,0,1,0,0,0,0,1,0,25,1,0,0,7]
dini = [0,0,1,0,1,0,0,0,1,0,0,28,0,1,1,16]
enno = [0,0,1,0,1,0,0,0,0,1,0,42,1,0,0,8]

print('Arin, prediksi kesuburan:',(logmodel.predict([arin])),'(Logistic Regression)')
print('Arin, prediksi kesuburan:',(knnmodel.predict([arin])),'(K Neighbors)')
print('Arin, prediksi kesuburan:',(svcmodel.predict([arin])),'(SVC)')

print('Bebi, prediksi kesuburan:',(logmodel.predict([bebi])),'(Logistic Regression)')
print('Bebi, prediksi kesuburan:',(knnmodel.predict([bebi])),'(K Neighbors)')
print('Bebi, prediksi kesuburan:',(svcmodel.predict([bebi])),'(SVC)')

print('Caca, prediksi kesuburan:',(logmodel.predict([caca])),'(Logistic Regression)')
print('Caca, prediksi kesuburan:',(knnmodel.predict([caca])),'(K Neighbors)')
print('Caca, prediksi kesuburan:',(svcmodel.predict([caca])),'(SVC)')

print('Dini, prediksi kesuburan:',(logmodel.predict([dini])),'(Logistic Regression)')
print('Dini, prediksi kesuburan:',(knnmodel.predict([dini])),'(K Neighbors)')
print('Dini, prediksi kesuburan:',(svcmodel.predict([dini])),'(SVC)')

print('Enno, prediksi kesuburan:',(logmodel.predict([enno])),'(Logistic Regression)')
print('Enno, prediksi kesuburan:',(knnmodel.predict([enno])),'(K Neighbors)')
print('Enno, prediksi kesuburan:',(svcmodel.predict([enno])),'(SVC)')


