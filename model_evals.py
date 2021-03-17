import os, sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pylon.data import build_dataset, load_json
import json, pickle
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_today(CURRENT_TRAIN, CURRENT_EVAL, TODAY,model_options):
    model_type = model_options.loc['TYPE']
    with open(CURRENT_TRAIN + "model_json.json") as json_file:
        json_train = json.load(json_file)
    
    with open(CURRENT_EVAL + "eval_json.json", "w") as out_file:
        json.dump(json_eval, out_file)
    
    eval_files = os.listdir(CURRENT_EVAL)
            
    if (not have_hist) or (not have_spc):
        # TODO: get todays eval data

        have_data_today = True

    else:
        print("Already have today's evaluation data.")
        have_data_today = True
    
    have_eval = False
    
    if have_data_today:
        pred_today, spc_today = apply_model(CURRENT_TRAIN, CURRENT_EVAL,model_type)
        have_eval = True
    else:
        pred_today = None, None
        
    return pred_today, have_eval

def apply_model(CURRENT_TRAIN, CURRENT_EVAL,model_type):
    
    # import data
    
    # Fetch pickled models
    models = []
    wet_reg = []
    intermediaryModels = ['PCAmodel.p','scalar.p']
    if model_type == 'classification':
        model_files = os.listdir(CURRENT_TRAIN + '/models/')
        for file in model_files:
            if file[-2:] == '.p' and file not in intermediaryModels:
                with open(CURRENT_TRAIN + '/models/'+file, 'rb') as pickle_file:
                    models.append((file[:-2],pickle.load(pickle_file)))
                
    if model_type == 'regression' or model_type == 'regression_SD':
        hidden_files = ['__pycache__','.ipynb_checkpoints']
        model_files = next(os.walk(CURRENT_TRAIN + '/models/'))[1]
        model_files = [file for file in model_files if file not in hidden_files]
        for folder in model_files:
            new_files = os.listdir(CURRENT_TRAIN + '/models/' + folder)
            for file in new_files:
                if file[-2:] == '.p':
                    with open(CURRENT_TRAIN + '/models/'+folder+'/'+file, 'rb') as pickle_file:
                        models.append((file[:-2],pickle.load(pickle_file)))
                        wet_reg.append(folder)
    
    with open(CURRENT_TRAIN + 'models/PCAmodel.p', 'rb') as pickle_file:
        modelPCA = pickle.load(pickle_file)
    with open(CURRENT_TRAIN + 'models/scalar.p', 'rb') as pickle_file:
        modelScalar = pickle.load(pickle_file)
        
    def inverseScalar(pred_col,idx_pred):
        aug_df = pd.DataFrame(np.zeros((len(pred_col), modelPCA.n_features_+len(model_files) )))
        aug_df.iloc[:,idx_pred] = pred_col
        pred_inv = modelScalar.inverse_transform(aug_df)
        return pred_inv[:,idx_pred]
    
    # Impute Data
    # TODO: fix the dfs
    spc_train = spc_today.fillna(spc_mean)
    keep_train_cols = spc_train.columns[spc_train.columns.isin(keep_cols['TRAINING_PARAMS'])]
    spc_train = spc_train[keep_train_cols]
    spc_index = spc_train.index
    for col in keep_cols.loc[~keep_cols.isin(spc_train.columns).values].values.ravel():
        mu,s, new_df = calc_pseudo_stats(spc_model_train,3)
        spc_train[col] = mu[col]
    
    spc_train = spc_train[keep_cols['TRAINING_PARAMS'].values]
    
    # Perform Standardize, Scale & run PCA
    numb_response = len(wet_cols)
    spc_train_comb = np.concatenate((spc_train, np.zeros((spc_train.shape[0],numb_response))), axis=1)
    spc_train = modelScalar.transform(spc_train_comb)
    spc_train = spc_train[:,:-numb_response]
    spc_train = modelPCA.transform(spc_train)
    
    # Setup predicted results
    pred_columns = [file +'_'+wet_reg[i] for i,(file,m) in enumerate(models)] if model_type != 'classification' else [file for file,m in models] 
    pred_today_df = pd.DataFrame(np.zeros((len(spc_index),len(models))), index=spc_index, columns = pred_columns)
    pred_today_df.index.name = 'LOT_PARENT'
    
    # Loop over all models and predict output
    for i,(file,model) in enumerate(models):
        i_wet = modelPCA.n_features_+ i % len(model_files)
        if 'XGB' not in file:
            if model_type == 'classification':
                preds_today =  model.predict_proba(spc_train)[:,1]
            elif model_type == 'regression':
                preds_today = inverseScalarWET(model.predict(spc_train), i_wet )
            else:
                preds_today =  model.predict(spc_train)
        else:
            spc_train_dM = xgb.DMatrix(spc_train)
            if model_type == 'classification':
                preds_today =  model.predict_proba(spc_train_dM)[:,1]
            elif model_type == 'regression':
                preds_today = inverseScalarWET(model.predict(spc_train_dM), i_wet )
            else:
                preds_today =  model.predict(spc_train_dM)
        
        # Grab the preds from the Training folder to create a scatter plot + rugplot
        wet_path_filler = wet_reg[i] if model_type != 'classification' else ''
        PREDS_TRAIN_PATH = CURRENT_TRAIN + 'models/' + wet_path_filler + '/' + file.split('best')[-1] + 'TRAINpredVsAct.csv'  
        PREDS_TEST_PATH = CURRENT_TRAIN + 'models/' + wet_path_filler + '/' + file.split('best')[-1] + 'TESTpredVsAct.csv'
        preds_train_df = pd.read_csv(PREDS_TRAIN_PATH)
        preds_test_df = pd.read_csv(PREDS_TEST_PATH)
        
        # Create predicted class
        pred_dict_name = file + '_' + wet_reg[i] if model_type != 'classification' else file
        pred_today_df[pred_dict_name] = pd.Series(preds_today, index=spc_index)
        
        if model_type == 'classification':
            convert_to_class = lambda lst, x: [val >= x for val in lst]
            threshold_val = preds_test_df['THRESH'].iloc[0]
            pred_today_df[pred_dict_name + '_CLASS'] = pd.Series(convert_to_class(preds_today,threshold_val), index=spc_index)
        
        # Save current preds to eval folder
        if not os.path.isdir(CURRENT_EVAL + 'images/'):
            os.mkdir(CURRENT_EVAL + 'images/') 
        
        fig = plt.figure(figsize=(8,6))
        plt.scatter(preds_train_df['PRED_TRAIN'],preds_train_df['ACT_TRAIN'], color='b', label='train')
        plt.scatter(preds_test_df['PRED_TEST'],preds_test_df['ACT_TEST'], color='r', label='test')
        sns.rugplot(preds_today)
        
        if model_type != 'classification':
            xbottom, xtop = min(min(preds_train_df['PRED_TRAIN']), min(preds_test_df['PRED_TEST'])), max(max(preds_train_df['PRED_TRAIN']), max(preds_test_df['PRED_TEST']))
            xbottom,xtop = min(xbottom,min(preds_today)), max(xtop,max(preds_today))
            ybottom, ytop = min(min(preds_train_df['ACT_TRAIN']), min(preds_test_df['ACT_TEST'])), max(max(preds_train_df['ACT_TRAIN']), max(preds_test_df['ACT_TEST']))
            bottom,top = min(xbottom,ybottom), max(xtop, ytop)
            
            plt.plot(np.linspace(bottom,top,100),np.linspace(bottom,top,100))
            plt.xlim((xbottom, xtop))
            plt.ylim((ybottom, ytop))
        plt.legend(loc='best')
        plt.xlabel("predictions")
        plt.ylabel("truth")
        plt.title("Pred. vs. Truth "+ pred_dict_name)
        plt.savefig(CURRENT_EVAL + 'images/' + (wet_reg[i] + '_' if wet_path_filler != '' else '') + file + '_preds.png')
    
    # Output predicted csv
    pred_today_df = pred_today_df.join(ts_today, how="left").rename(columns={'DATE_STEP_STARTED': 'TRIGGER_TIME'})
    have_eval = True
    pred_today_df.to_csv(CURRENT_EVAL+'predsOutlier.csv')
    
    return pred_today_df, spc_today
  