import numpy as np
import pandas as pd 
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint
from sklearn.metrics import brier_score_loss
import warnings
warnings.filterwarnings("ignore")


def call_model(a, b, c, d, e, f, g, h, i, j, k, l, m):
    #pull data
    trainData = pd.read_csv("./heart.csv") 

    #PRERPOCESSING
    #delete duplicates
    trainData = trainData.drop_duplicates()
    #split train
    X = trainData.drop(columns=['target'], axis=1)  # Replace 'target' with your actual target column name
    Y = trainData['target']
    # Split the data into training and test sets 80-20
    #dataSetWithoutTargetTrain, dataSetWithoutTargetTest, targetTrain, targetTest
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.2, stratify=Y)
    #standardizing features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #train models
    LRmodel = LogisticRegression(C = 0.1, class_weight= 'balanced', fit_intercept= True, l1_ratio= 0.1, max_iter=100, penalty='elasticnet', solver = 'saga')
    LRmodel.fit(X_train, Y_train) #tries to find a pattern between all medical stats and whether or not the person has heart disease

    KNNmodel = KNeighborsClassifier(algorithm='auto', leaf_size = 10, metric = 'manhattan',  n_neighbors=16, p =1, weights='uniform')
    KNNmodel.fit(X_train, Y_train)

    RFmodel = RandomForestClassifier(
        n_estimators = 75,  # Reduce number of trees
        min_samples_split = 30,  # Increase minimum samples for split
        min_samples_leaf = 10,   # Increase minimum samples for leaf
        max_features = 'sqrt',   # Limit features to sqrt of total
        max_depth = 15,  # Shallower trees
        class_weight = 'balanced', 
        bootstrap = True
    )
    RFmodel.fit(X_train, Y_train)

    SVMmodel = SVC(C = 0.1, class_weight = 'balanced', degree = 2, gamma = 'scale', kernel='linear', max_iter = 200, probability= True)
    SVMmodel.fit(X_train, Y_train)

    MLPmodel = MLPClassifier(learning_rate= 'adaptive', learning_rate_init=0.1, activation = 'logistic', alpha = 0.0001, batch_size = 32, early_stopping = False, hidden_layer_sizes = 100, max_iter = 500, solver= 'sgd')
    MLPmodel.fit(X_train, Y_train)

    VCmodel = VotingClassifier(
        estimators=[('lr', LRmodel), ('knn', KNNmodel), ('rf', RFmodel), ('svm', SVMmodel), ('mlp', MLPmodel)],
        voting='hard'
    )
    VCmodel.fit(X_train, Y_train)
    
    inputtedData = (a, b, c, d, e, f, g, h, i, j, k, l, m)
    reshapedNumPyArr = scaler.transform(np.asarray(inputtedData).reshape(1,-1))
    prediction = VCmodel.predict(reshapedNumPyArr)[0]
    if(prediction == 0):
        return "Your results indicate low risk of having heart disease."
    else:
        return "Your results indicate a high chance of having heart disease."
