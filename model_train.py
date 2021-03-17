import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

import tensorflow as tf
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, confusion_matrix, balanced_accuracy_score,make_scorer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.gaussian_process.kernels import RBF, DotProduct
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
import xgboost as xgb

import multiprocessing as mp
import tqdm
import sys, os
import time
from datetime import datetime
sns.set()

def _dropSparseCol(df,thresh):
    """
    Return DataFrame with subset of columns that a proportion of nonNaN entries to total entries above `thresh`
    
    @param thresh: float defining the threshold for the proportion of filled values in a column. 
    Keep all columns above this values
    
    @param df: pandas DataFrame
    """
    isNotSparse = (df.describe().loc[ 'count' ,:].values)/len(df.index) >= thresh
    df_dropCol = df.iloc[:,isNotSparse]
    return df_dropCol

def _getResponseCol(df,cols,m=2):
    """
    Return np array with boolean values indicating the class depending on the `cols` param
    `options` defines the methods used:
    1. 'use2D': take the element-wise logical or of v1 ... vn with n:=len(cols)
        Let vi be be a boolean array where each element represents whether that observation is outside the 
        2 standard deviations of of the mean of that col
    
    @param df: pandas DataFrame
    @param cols: list of column names for `df` deliminating what columns to create a classification response from. Here, we expect WET cols
    """
    options = {'use2SD':True}
    
    for opt in options:
        if opt == 'use2SD' and options[opt]:
            prev = pd.Series([False]*df.shape[0])
            for col in cols: 
                prev = np.logical_or(prev.values, np.abs(df[col] - np.mean(df[col])) > m * np.std(df[col]))
    return prev

def _removeSmallSE(df, thresh):
    """
    Return a DataFrame that drops columns whose mean/sd is below threshold (Not exactly SE, but close enough)
    
    @param df: pandas DataFrame
    @param thresh: float defining the threshold for the proportion of filled values in a column. 
    Keep all columns above this values
    """
    colsToRemove =[]
    counter=0
    for col in df.columns.values:
        try:
            colSE = np.mean(df[col])/np.std(df[col])
        except:
            # Catch a divide by zero Exception
            colsToRemove.append(col)
            
        if np.abs(colSE) <= thresh:
            colsToRemove.append(col)
    if colsToRemove != []:
        return df.drop(columns=colsToRemove)
    else: 
        return df

def runPCA(data):
    """
    Return numpy array of transform PCA `data` and the associated sklearn model
    The reduced dimension taken will be the first k whose explained variance is above 1
    
    @param data: pandas DataFrame
    """
    print('Running PCA')
    pcaModel = PCA()
    pcaModel.fit(data)
    cummalVarRatio = [np.sum(pcaModel.explained_variance_ratio_[:i+1]) for i,_ in enumerate(pcaModel.explained_variance_ratio_)]

    eigenvalues = pcaModel.explained_variance_
    kProp = np.argmin(pcaModel.explained_variance_ >= 1)
    # kProp = np.argmax(np.array(cummalVarRatio) > proportion)   MAKE SURE TO SET PROPORTION
    print('PCA reduced to {}. Original shape: {}'.format(str(kProp), str(data.shape)))
    pcaProportionModel = PCA(kProp)
    pcaProportionModel.fit(data)
    return pcaProportionModel.transform(data), pcaProportionModel

def writeToLogFile(CURRENT_MODEL_FOLDER,s):
    """
    Write str `s` or list `s` to `CURRENT_MODEL_FOLDER` + logs.txt
    Also adds timestamp
    
    @param CURRENT_MODEL_FOLDER: file path to current train folder
    @param s: string or list of strings 
    """
    if type(s) == list:
        s = map(str, s)
        s = ' '.join(s)
    else:
        s = str(s)
    f = open(CURRENT_MODEL_FOLDER + 'logs.txt', 'a+')
    f.write('[{}]: {}\n'.format(datetime.now().strftime("%m/%d/%Y %H:%M:%S"), s ))
    f.close()

def get_data(train, wet_cols):
    """
    Return train-test split on train
    
    @param train: pandas DataFrame with columns `wet_cols` as response variable
    @param wet_cols: list of response columns in `wet_cols`
    """
    X_train, X_test,y_train,y_test = train_test_split(train.drop(wet_cols, axis=1), train[wet_cols] , test_size = .3, random_state=909)
    return X_train, X_test, y_train, y_test

def _scatter_preds(preds, truth, CURRENT_MODEL_FOLDER, col, method=''):
    """
    Plot a scatter plot of preds abd truth
    
    @param preds: array like 
    @param truth: array like
    @param CURRENT_MODEL_FOLDER: path to save folder to
    @param col: name of folder (for wet parm of interest) under images
    @param method: string denoting the algorithm used
    """
    fig = plt.figure(figsize=(10,8))
    sns.regplot(preds, truth, marker="+")
    top = max(max(preds),max(truth))
    bottom = min(min(preds),min(truth))
    x = np.linspace(bottom,top,100)
    y102 = x * 1.02
    y105 = x * 1.05
    y98 = x * 0.98
    y95 = x * 0.95
    plt.plot(x,x, 'g', linewidth=3)
    plt.plot(x,y102, 'y', linewidth=3)
    plt.plot(x,y98, 'y', linewidth=3)
    plt.plot(x,y105, 'r', linewidth=3)
    plt.plot(x,y95, 'r', linewidth=3)
    plt.ylim((bottom, top))
    plt.xlim((bottom, top))
    plt.xlabel("predictions")
    plt.ylabel("truth")
    plt.title("Pred. vs. Truth "+ method)
    plt.savefig(CURRENT_MODEL_FOLDER +"/images/"+col+"/pred_scatter_"+method +".png")
    plt.show()

def _plotROC(preds, truth,CURRENT_MODEL_FOLDER,method = ''):
    """
    Plot a ROC plot of preds and truth
    
    @param preds: array like 
    @param truth: array like
    @param CURRENT_MODEL_FOLDER: path to save folder to
    @param method: string denoting the algorithm used
    """
    fpr,tpr, thresh = roc_curve(truth, preds)
    fig, ax = plt.subplots(figsize=(6,6))
    plt.plot(fpr, tpr)
    plt.plot(np.linspace(0,1,30),np.linspace(0,1,30))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC curve ' + method)
    if method != "":
        filename = method+'ROC'
    else:
        filename = 'modelROC'
    plt.savefig(CURRENT_MODEL_FOLDER +'images/'+ filename + '.png')
    return fpr,tpr,thresh

def _getConfusionMatrix(preds, truth, thresh,fpr, tpr, CURRENT_MODEL_FOLDER,method = '', figsize=(6,6)):
    """
    Plot a confusion matrix of preds and truth
    
    @param preds: array like 
    @param truth: array like
    @param thresh: array like
    @param fpr: array like
    @param tpr: array like
    @param CURRENT_MODEL_FOLDER: path to save folder to
    @param method: string denoting the algorithm used
    @param figsize: tuple
    """
    # Get Threshold val
    # First filter all thresholds to those with fpr less than `arbitrary_fpr_max`
    # Next take all indices with maximal tpr
    # Finally take the threshold of those indices with minimal fpr
    arbitrary_fpr_max = 0.45
    idx_fpr = [i for i,val in enumerate(fpr) if val <= arbitrary_fpr_max]
    tpr_with_fpr_max = [tpr[i] for i in idx_fpr]
    max_tpr = max(tpr_with_fpr_max)
    idx_max_tpr = [(i,fpr[i]) for i, j in enumerate(tpr_with_fpr_max) if j == max_tpr]
    threshold_i = idx_max_tpr[0][0] 
    threshold_v = idx_max_tpr[0][1]
    for i,v in idx_max_tpr[1:]:
        if v < threshold_v:
            threshold_i = i
            threshold_v = v
    threshold_val = thresh[threshold_i]
    
    # Round and make judgement
    preds = [val >= threshold_val for val in preds]
    cm = confusion_matrix(truth, preds, labels=np.unique(truth))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(truth), columns=np.unique(truth))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    snsCM = sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
    
    if method != "":
        filename = method+'CM'
    else:
        filename = 'modelCM'
    fig.savefig(CURRENT_MODEL_FOLDER +'/images/'+ filename + '.png')
    
    return threshold_val
    
def _runClassificationModel(X_train, X_test, y_train, y_test, pipe, pipe_params,modelMethod,CURRENT_MODEL_FOLDER):
    """
    Taking in data, conduct a gridsearch over `pipe_params` using the `pipe` and return the best model found
    
    @param X_train: array like
    @param X_test: array like
    @param y_train: vector like
    @param y_test: vector like
    @param pipe: sklearn Pipeline containing algorithm to use
    @param pipe_params: dict for pipe
    @param modelMethod: string for algorithm used
    param CURRENT_MODEL_FOLDER: string for path
    """
    # create scorer
    balanced_accuracy_scorer = make_scorer(balanced_accuracy_score, )
    
    # Fit model. Ravel done to avert warnings in some algorithms
    gs = GridSearchCV(pipe, param_grid=pipe_params, cv=StratifiedKFold() ,n_jobs=10, scoring = balanced_accuracy_scorer)
    if type(y_train) == pd.Series or type(y_train) == pd.DataFrame:
        y_train = y_train.values.ravel()
    gs.fit(X_train, y_train)
    
    # Get quick model stats
    print('cvs:', gs.best_score_)
    writeToLogFile(CURRENT_MODEL_FOLDER, ['cvs:', gs.best_score_])
    print('train score:', gs.score(X_train, y_train))
    writeToLogFile(CURRENT_MODEL_FOLDER, ['train score:', gs.score(X_train, y_train)])
    print('test score:', gs.score(X_test, y_test))
    writeToLogFile(CURRENT_MODEL_FOLDER, ['test score:', gs.score(X_test, y_test)])
    writeToLogFile(CURRENT_MODEL_FOLDER, gs.best_params_)
    writeToLogFile(CURRENT_MODEL_FOLDER, gs.best_estimator_)
    print(gs.best_params_)
    print(gs.best_estimator_)

    # Predict
    preds_train = gs.best_estimator_.predict_proba(X_train)[:,1]
    preds_test = gs.best_estimator_.predict_proba(X_test)[:,1]
    
    # Convert and create output csvs
    convert_to_class = lambda lst, x: [val >= x for val in lst]
    
    # Get ROC and confustion matric
    fpr,tpr,thresh = _plotROC(preds_test, y_test,CURRENT_MODEL_FOLDER,method = modelMethod )
    threshold_val = _getConfusionMatrix(preds_test,y_test, thresh, fpr, tpr, CURRENT_MODEL_FOLDER,method = modelMethod)
    
    preds_train_df = pd.DataFrame(convert_to_class(preds_train, threshold_val), columns = ['PRED_TRAIN_CLASS'], index = X_train.index)
    preds_train_df['PRED_TRAIN'] = preds_train
    preds_train_df['ACT_TRAIN'] = y_train
    preds_test_df = pd.DataFrame(convert_to_class(preds_test, threshold_val), columns = ['PRED_TEST_CLASS'], index = X_test.index)
    preds_test_df['PRED_TEST'] = preds_test
    preds_test_df['ACT_TEST'] = y_test
    preds_test_df['THRESH'] = pd.Series([threshold_val]*len(y_test), index=X_test.index)
    
    preds_train_df.to_csv(CURRENT_MODEL_FOLDER + 'models/' + modelMethod + 'TRAINpredVsAct.csv')
    preds_test_df.to_csv(CURRENT_MODEL_FOLDER + 'models/' + modelMethod + 'TESTpredVsAct.csv')
    roc_df = pd.DataFrame(thresh, columns = ['THRESHOLD'])
    roc_df['FPR'] = fpr
    roc_df['TPR'] = tpr
    roc_df.to_csv(CURRENT_MODEL_FOLDER + 'models/' + modelMethod + 'roc.csv')
    
    return gs.best_estimator_

def _runRegressionModel(X_train, X_test, y_train, y_test, pipe, pipe_params,scalar,PCAmodel,i_wet, tot_wet, modelMethod, model_type,CURRENT_MODEL_FOLDER):
    """
    Taking in data, conduct a gridsearch over `pipe_params` using the `pipe` and return the best model found
    
    @param X_train: array like
    @param X_test: array like
    @param y_train: vector like
    @param y_test: vector like
    @param pipe: sklearn Pipeline containing algorithm to use
    @param pipe_params: dict for pipe
    @param scalar: sklearn scalar
    @param PCAmodel: sklearn PCAmodel. Simply to extrac the number of features used before scalar
    @param i_wet: int corresponding to index of the wet column of interest. Used for the scalar
    @param tot_wet: int used to construct the empty array for the scalar
    @param modelMethod: string for algorithm used
    @param CURRENT_MODEL_FOLDER: string for path
    """
    # Get the column name (wet) of interest for folder acquisition
    col = y_train.name.split('/')[0]
    
    # Fit model
    gs = GridSearchCV(pipe, param_grid=pipe_params, cv=5,n_jobs=10, scoring ='r2')
    gs.fit(X_train, y_train)
    
    # Get quick model stats
    print('cvs:', gs.best_score_)
    writeToLogFile(CURRENT_MODEL_FOLDER, ['cvs:', gs.best_score_])
    print('train score:', gs.score(X_train, y_train))
    writeToLogFile(CURRENT_MODEL_FOLDER, ['train score:', gs.score(X_train, y_train)])
    print('test score:', gs.score(X_test, y_test))
    writeToLogFile(CURRENT_MODEL_FOLDER, ['test score:', gs.score(X_test, y_test)])
    
    writeToLogFile(CURRENT_MODEL_FOLDER, gs.best_params_)
    writeToLogFile(CURRENT_MODEL_FOLDER, gs.best_estimator_)
    print(gs.best_params_)
    print(gs.best_estimator_)
    
    # Invert the predicted output if model_type is 'regression'
    def inverseScalarWET(wet_col):
        aug_df = pd.DataFrame(np.zeros((len(wet_col), PCAmodel.n_features_+tot_wet)))
        aug_df.iloc[:, i_wet] = wet_col
        pred_inv = scalar.inverse_transform(aug_df)
        return pred_inv[:, i_wet]
    preds_train = inverseScalarWET(gs.best_estimator_.predict(X_train)) if model_type == 'regression' else gs.best_estimator_.predict(X_train)
    preds_test = inverseScalarWET(gs.best_estimator_.predict(X_test)) if model_type == 'regression' else gs.best_estimator_.predict(X_test)
    y_train_inv = inverseScalarWET(y_train.values) if model_type == 'regression' else y_train.values
    y_test_inv = inverseScalarWET(y_test.values) if model_type == 'regression' else y_test.values
    
    # Convert and create output csvs
    preds_train_df = pd.DataFrame(preds_train, columns = ['PRED_TRAIN'], index=X_train.index)
    preds_train_df['ACT_TRAIN'] = y_train_inv
    preds_test_df = pd.DataFrame(preds_test, columns = ['PRED_TEST'],index=X_test.index)
    preds_test_df['ACT_TEST'] = y_test_inv
    preds_train_df.to_csv(CURRENT_MODEL_FOLDER +'/models/'+col+'/' + modelMethod + 'TRAINpredVsAct.csv')
    preds_test_df.to_csv(CURRENT_MODEL_FOLDER +'/models/'+col+'/' + modelMethod + 'TESTpredVsAct.csv')
    
    # Create preds vs. truth plots
    _scatter_preds(preds_train,  y_train_inv,CURRENT_MODEL_FOLDER, col, modelMethod+'_train')
    _scatter_preds(preds_test, y_test_inv,CURRENT_MODEL_FOLDER, col, modelMethod+'_test')
    
    return gs.best_estimator_

def run_model(CURRENT_MODEL_FOLDER, model_type, X_train, X_test, y_train, y_test, scalar, modelPCA):
    """
    Taking in data, Aggregate the model types and call the associated model function
    
    @param CURRENT_MODEL_FOLDER: string for path
    @param model_type: string denoting classification or regression
    @param X_train: array like
    @param X_test: array like
    @param y_train: vector like
    @param y_test: vector like
    @param scalar: sklearn scalar
    @param PCAmodel: sklearn PCAmodel. Simply to extrac the number of features used before scalar
    """
    # Set up variables for all algorithms
    pipelines = []
    pipeline_params = []
    model_lim = -1
    if model_type == 'classification':
        pipelines.append( Pipeline([('lr', LogisticRegression(solver='liblinear')) ]) )
        pipeline_params.append( [{
                'lr':[LogisticRegression(penalty='l2', max_iter = 100000)],
                'lr__solver': ['newton-cg','sag','lbfgs'],
                'lr__C':[1,0.5,0.25,0.1,0.01,0.001],
                'lr__class_weight':['balanced']
            },
            {
                'lr':[LogisticRegression(penalty='l1',solver = 'liblinear', max_iter = 100000)],
                'lr__C':[1,0.5,0.25,0.1,0.01,0.001],
                'lr__class_weight':['balanced']
            },
            {
                'lr':[LogisticRegression(penalty = 'elasticnet', solver='saga', max_iter = 100000)],
                'lr__C':[1,0.1,0.01,0.001],
                'lr__l1_ratio':[0.1,0.5,0.9],
                'lr__class_weight':['balanced']
            }
        ])

        pipelines.append( Pipeline([('svc', SVC(probability = True))]) )
        pipeline_params.append( {
            'svc__kernel': ['linear','poly','rbf','sigmoid'],
            'svc__degree': [2,3,4,5],
            'svc__gamma':['scale','auto',0.01,0.1,1,3],
            'svc__coef0':[0.01,0.1,1,3],
            'svc__random_state':[100],
            'svc__C': [0.0001,0.001,0.01,0.1,1]
        })

        pipelines.append( Pipeline([('ada',AdaBoostClassifier())]) )
        pipeline_params.append( {
            'ada__n_estimators': [25,50,100,200],
            'ada__learning_rate': [0.001,0.01,0.1,1]
        })

        pipelines.append( Pipeline([('knn', KNeighborsClassifier())]) )
        pipeline_params.append( {
            'knn__n_neighbors': [3,5,8,12],
            'knn__weights': ['uniform', 'distance']
        })

        pipelines.append( Pipeline([('gp', GaussianProcessClassifier(max_iter_predict=5000, n_restarts_optimizer=25, warm_start = True))]) )
        pipeline_params.append( {
            'gp__kernel': [1.0 * RBF(length_scale=1.0), 1.0 * DotProduct(sigma_0=1.0)**2]
        })

        pipelines.append( Pipeline([('dtree', DecisionTreeClassifier() )]) )
        pipeline_params.append( {
            'dtree__criterion': ["gini", "entropy"],
            'dtree__splitter':['best', 'random'],
            'dtree__max_depth':[None, 3,5,8],
            'dtree__max_features':[None, 'sqrt', 'log2']
        })

        pipelines.append( Pipeline([('rf', RandomForestClassifier() )]) )
        pipeline_params.append( {
            'rf__criterion': ["gini", "entropy"],
            'rf__n_estimators':[50,100,200,500],
            'rf__max_depth':[None, 3,5,8],
            'rf__max_features':[None, 'sqrt', 'log2'],
            'rf__bootstrap':[True, False]
        })

        pipelines.append( Pipeline([('nb', GaussianNB() )]) )
        pipeline_params.append( {
        })

        pipelines.append( Pipeline([('qda', QuadraticDiscriminantAnalysis() )]) )
        pipeline_params.append( {
        })

        # THESE TAKE A WHILE
        pipelines.append( Pipeline([('nn', MLPClassifier(max_iter=5000) )]) )
        pipeline_params.append( {
            'nn__hidden_layer_sizes': [(8,),(16,),(32,),(64,),(32,64,),(32,64,64,),(32,64,64,64,)],
            'nn__activation':['logistic','tanh','relu'],
            'nn__solver':['lbfgs','sgd','adam'],
            'nn__alpha':[0.000001,0.00001,0.0001,0.001,0.01,0.1],
            'nn__learning_rate':['invscaling','adaptive']
        })
        
        pipelines.append( Pipeline([('xgb',xgb.XGBClassifier(objective='binary:logistic') )]) )
        pipeline_params.append({
            'xgb__learning_rate': [0.0001, 0.001,0.01,0.1],
            'xgb__max_depth': [3,4,5,7],
            'xgb__booster': ['gbtree','gblinear'],
            'xgb__gamma':[0,0.0001,0.001,0.01,0.1],
            'xgb__min_child_weight': [0.001,0.01,0.1,1,2,4],
            'xgb__min_delta_step': [0.001,0.01,0.1,1,2,4],
            'xgb__base_score':[0.05,0.1,0.2,0.5],
            'xgb__colsample_bytree': [0.7]
                 })

        model_names = ['bestLR','bestSVM','bestADA','bestKNN','bestGP','bestDTREE', 'bestRF','bestNB','bestQDA','bestNN','bestXGB']
        
        if model_lim != -1:
            model_names = model_names[:model_lim]
        
        # Run Classification models
        models = {}
        for i,model_name in enumerate(model_names):
            pipe = pipelines[i]
            pipe_param = pipeline_params[i]
            method = model_name.split('best')[-1]
            start = time.time()
            models[model_name] = _runClassificationModel(X_train, X_test, y_train, y_test, pipe,pipe_param,method,CURRENT_MODEL_FOLDER)
            end = time.time()
            writeToLogFile(CURRENT_MODEL_FOLDER, ['Total time taken to train model ',method,':',end-start])
    
    #======================== REGRESSION ========================
    elif model_type == 'regression' or model_type == 'regression_SD':
        pipelines.append( Pipeline([('lr', LinearRegression()) ]) )
        pipeline_params.append({})
        
        pipelines.append( Pipeline([('el', ElasticNet(max_iter=5000)) ]) )
        pipeline_params.append(
            {
                'el__alpha':[0.0001, 0.001,0.01,0.1,1],
                'el__l1_ratio':[0.1,0.5,0.9]
            })
        
        pipelines.append( Pipeline([('rf', RandomForestRegressor() )]) )
        pipeline_params.append( {
            'rf__criterion': ["mse", "mae"],
            'rf__n_estimators':[50,100,200,500],
            'rf__max_depth':[None, 3,5,8],
            'rf__max_features':[None, 'sqrt', 'log2'],
            'rf__bootstrap':[True, False]
        })
        

        pipelines.append( Pipeline([('svr', SVR())]) )
        pipeline_params.append( {
            'svr__kernel': ['linear','poly','rbf','sigmoid'],
            'svr__degree': [2,3,4,5],
            'svr__gamma':['scale','auto',0.01,0.1,1,3],
            'svr__coef0':[0.01,0.1,1,3],
            'svr__C': [0.0001,0.001,0.01,0.1,1]
        })
        
        pipelines.append( Pipeline([('xgb',xgb.XGBRegressor(objective='reg:squarederror') )]) )
        pipeline_params.append({
            'xgb__learning_rate': [0.0001, 0.001,0.01,0.1],
            'xgb__max_depth': [3,4,5,7],
            'xgb__booster': ['gbtree','gblinear'],
            'xgb__gamma':[0,0.0001,0.001,0.01,0.1],
            'xgb__min_child_weight': [0.001,0.01,0.1,1,2,4],
            'xgb__min_delta_step': [0.001,0.01,0.1,1,2,4],
            'xgb__base_score':[0.05,0.1,0.2,0.5],
            'xgb__colsample_bytree': [0.7]
        })

        model_names = ['bestLR','bestEL', 'bestRF', 'bestSVR', 'bestXGB']
        models = {}
        
        # Run through all model algos
        for i,model_name in enumerate(model_names):
            pipe = pipelines[i]
            pipe_param = pipeline_params[i]
            method = model_name.split('best')[-1]
            
            # Apply an algo to each wet params 
            for i_wet in range(len(y_train.columns)):
                i_wet_comb = i_wet + modelPCA.n_features_
                start = time.time()
                wet_col_param_name = y_train.columns.values[i_wet]
                y_train_i = y_train[wet_col_param_name]
                y_test_i = y_test[wet_col_param_name]
                models[model_name +'_'+wet_col_param_name.split('/')[0]] = _runRegressionModel(X_train, X_test, y_train_i, y_test_i, pipe,pipe_param,scalar, modelPCA,i_wet_comb, len(y_train.columns), method, model_type, CURRENT_MODEL_FOLDER)
                end = time.time()
                writeToLogFile(CURRENT_MODEL_FOLDER, ['Total time taken to train model ',method,wet_col_param_name,':',end-start])
    
    return models
    

def train_model(xy_df,features,predictors, model_type):
    """
    Pull data, filter columns, run model. Return train and wet columns used in this train cycle
    
    @param CURRENT_MODEL_FOLDER: string for path
    @param model_type: string denoting classification or regression
    """

    # Drop all columns with less than `drop_sparse_thresh` proportion of non empty values 
    drop_sparse_thresh = 0.75
    clean = _dropSparseCol(xy_df,drop_sparse_thresh)
    
    # Remove columns with a mean/sd < `se_thresh`
    se_thresh = 0.05
    clean = _removeSmallSE(clean,se_thresh)
    print('Removed Small SE & Drop Sparse to {}. Original shape: {}'.format(str(clean.shape), str(spc_df.shape)))
    train_cols = clean.columns
    all_cols = pd.concat([train_cols.to_series(), wet_cols.to_series()])
    
    # Standardize and Scale and then run PCA with `pc_to_run` Principle Components    
    scalar = StandardScaler()
    scalar.fit(pd.concat([clean, wet_df], axis=1, join="inner"))
    train = scalar.transform(pd.concat([clean, wet_df], axis=1, join="inner"))[:,~all_cols.isin(wet_cols)]
    cleanPCA, modelPCA = runPCA(train)
    
    # Create needed folders
    if not os.path.isdir(CURRENT_MODEL_FOLDER + 'images/'):
        os.mkdir(CURRENT_MODEL_FOLDER + 'images/')
    if not os.path.isdir(CURRENT_MODEL_FOLDER + 'models/'):
        os.mkdir(CURRENT_MODEL_FOLDER + 'models/')
    
    if model_type != 'classification':
        for col in wet_cols:
            col = col.split('/')[0]
            if not os.path.isdir(CURRENT_MODEL_FOLDER +'models/' +col):
                os.mkdir(CURRENT_MODEL_FOLDER + 'models/'+col)
            if not os.path.isdir(CURRENT_MODEL_FOLDER +'images/'+col):
                os.mkdir(CURRENT_MODEL_FOLDER +'images/'+ col)
            if not os.path.isdir(CURRENT_MODEL_FOLDER +'models/'+col):
                os.mkdir(CURRENT_MODEL_FOLDER + 'models/'+col)
    
    # Run models
    if model_type == 'classification':
        clean_label = _getResponseCol(wet_df,wet_cols)
        
        cleanPCA = pd.DataFrame(cleanPCA, index = clean.index)
        cleanPCA['isOutlier'] = clean_label

        # Fetch data
        X_train, X_test, y_train, y_test = get_data(cleanPCA, ['isOutlier'])        

        # Run the models
        models = run_model(CURRENT_MODEL_FOLDER, model_type, X_train, X_test, y_train, y_test, scalar, modelPCA)
        
    elif model_type == 'regression' or model_type == 'regression_SD':
        cleanPCA = pd.DataFrame(cleanPCA, index = clean.index)
        wet = scalar.transform(pd.concat([clean, wet_df], axis=1, join="inner"))[:,all_cols.isin(wet_cols)]
        clean_comb = np.concatenate((cleanPCA, wet), axis=1)
        clean_comb = pd.DataFrame(clean_comb, columns = list(range(cleanPCA.shape[1])) + wet_cols.to_list(), index= clean.index)
        
        # Fetch data
        X_train, X_test, y_train, y_test = get_data(clean_comb, wet_cols)  
        
        # Run the models
        models = run_model(CURRENT_MODEL_FOLDER, model_type, X_train, X_test, y_train, y_test, scalar, modelPCA)
 
    # Save the models
    for name,model in models.items():
        if model_type == 'classification':
            pickle.dump(model, open(CURRENT_MODEL_FOLDER + 'models/' + name + '.p', 'wb') )
        else:
            col = name.split('_')[0]
            pickle.dump(model, open(CURRENT_MODEL_FOLDER +'models/' + "_".join(name.split('_')[1:]) +'/' + col + '.p', 'wb') )
    pickle.dump(modelPCA, open(CURRENT_MODEL_FOLDER + 'models/PCAmodel.p', 'wb') )
    pickle.dump(scalar, open(CURRENT_MODEL_FOLDER + 'models/scalar.p', 'wb') )

    return True,train_cols, wet_cols