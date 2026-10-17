import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


#Read file
df=pd.read_csv('titanic.csv')
df=df[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

#Missing values
df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])

#Encoding
df=pd.get_dummies(df,columns=['Sex','Embarked'])

X=df.drop('Survived',axis='columns')
Y=df['Survived']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

#Scaling
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#Regression
model1=LogisticRegression()
model1.fit(X_train,Y_train)


model2=tree.DecisionTreeClassifier()
model2.fit(X_train,Y_train)

model3=RandomForestClassifier()
model3.fit(X_train,Y_train)

#Accuracy
accuracy1=model1.score(X_test,Y_test)
accuracy2=model2.score(X_test,Y_test)
accuracy3=model3.score(X_test,Y_test)
print(accuracy1)
print(accuracy2)
print(accuracy3)



#Prediction_exemple
new_passenger=[[1,24,0,0,18,1,0,0,0,1]]
new_passenger=scaler.transform(new_passenger)
prediction1=model1.predict(new_passenger)
perdiction2=model2.predict(new_passenger)
perdiction3=model3.predict(new_passenger)
print(prediction1)
print(perdiction2)
print(perdiction3)
