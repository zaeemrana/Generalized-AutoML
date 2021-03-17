import os
import pandas as pd
from datetime import datetime, timedelta

from model_preps import prepare_model
from model_evals import evaluate_today

WORKING_DIR = os.getcwd() + "/"
CONFIG_FOLDER = WORKING_DIR + "config/"

def check_auto_dirs():
    """
    Create overall folders to contain models run
    """
    if not os.path.isdir(WORKING_DIR + 'auto/'):
        os.mkdir(WORKING_DIR + 'auto/')
    
    if not os.path.isfile(WORKING_DIR + 'logs.txt'):
        f = open('logs.txt', 'a+')
        f.close()
    
    return WORKING_DIR + 'auto/'


def writeToLogFile(WORKING_DIR, s):
    """
    Write str `s` to `WORKING_DIR` + logs.txt
    Also adds timestamp
    
    @param WORKING_DIR: file path to current working dir
    @param s: string
    """
    f = open(WORKING_DIR + 'logs.txt', 'a+')
    f.write('[{}]: {}'.format(datetime.now().strftime("%m/%d/%Y %H:%M:%S"), s+'\n' ))
    f.close()


def check_model_folders(WORKING_DIR, AUTO_FOLDER, auto_options):
    """
    Create model folders for one model setup
    
    @param WORKING_DIR: file path to current working dir. Just used for writeToLogFile function
    @param AUTO_FOLDER: file path that holds all model data
    @auto_options: pandas DataFrame of once column containing this specific model setup
    """
    model_name = auto_options['MODELNAME']
    NEW_PROJ_FOLDER = AUTO_FOLDER + model_name + '_' + str(auto_options.loc['VERSION']) + '/'
    if not os.path.isdir(NEW_PROJ_FOLDER):
        print('Creating new project at:', NEW_PROJ_FOLDER)
        writeToLogFile(WORKING_DIR, 'Creating new project at: '+ NEW_PROJ_FOLDER)
        os.mkdir(NEW_PROJ_FOLDER)
        os.mkdir(NEW_PROJ_FOLDER+ 'train/')
        os.mkdir(NEW_PROJ_FOLDER+ 'eval/')
    
    return NEW_PROJ_FOLDER
        
def main():
    """
    Loop through all models configs and run models.
    """
    TODAY = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d")
    
    # Fetch models to run and respective model options
    models_to_run = pd.read_csv(CONFIG_FOLDER + 'models_to_run.csv')
    
    # Fetch parameter data
    feature_parms_config = pd.read_csv(CONFIG_FOLDER + "feature_params.csv")
    predictor_parms_config = pd.read_csv(CONFIG_FOLDER + "predictor_params.csv")

    # Get Data
    data_df = getData()

    # Set up model directories and run main loop for models
    proj_folders = {}
    AUTO_FOLDER = check_auto_dirs()
    for idx in models_to_run.index:
        models_opts = models_to_run.loc[idx,:]
        PROJ_FOLDER = check_model_folders(WORKING_DIR, AUTO_FOLDER, models_to_run.loc[idx])

        model_name = models_opts['MODELNAME'] + '_' + str(models_opts.loc['VERSION'])
        
        # Identify feature parameter group and associated predictor parameter list
        feature_parm_gp = models_opts['FEATURE_SET']
        predictor_parm_gp = models_opts['PRED_SET']
        features = feature_parms_config.loc[feature_parms_config['SET'] == feature_parm_gp]['NAME'].values
        predictors = predictor_parms_config.loc[predictor_parms_config['SET'] == predictor_parm_gp]['NAME'].values

        # Construct desired data set
        xy_df = data_df[data_df.columns.isin(features+predictors)]

        # Create model config folder
        MODEL_CONFIG = CONFIG_FOLDER + model_name + '_' + str(models_opts.loc['VERSION']) + '/'
        if not os.path.isdir(MODEL_CONFIG):
            os.mkdir(MODEL_CONFIG)
        data_df[data_df.columns.isin(features)].to_csv(MODEL_CONFIG + 'feature_params.csv', index=False)
        data_df[data_df.columns.isin(predictors)].to_csv(MODEL_CONFIG + 'predictor_params.csv', index=False)
        xy_df.to_csv(MODEL_CONFIG + 'data.csv', index=False)

        print('Model Name: {} type: {}'.format(model_name, models_opts.loc['RESPONSE_GRP']))
        
        OLD_MODEL_DATE = (datetime.strptime(TODAY, "%Y-%m-%d") 
                      + timedelta(days=-int(models_opts.loc['DAYS_BACK_TO_PULL']) )).replace(hour=0, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d")
        
        # Train model if possible
        TRAIN_FOLDER, EVAL_FOLDER, CURRENT_MODEL_FOLDER, CURRENT_EVAL = prepare_model(PROJ_FOLDER,CONFIG_FOLDER, TODAY, OLD_MODEL_DATE, model_opts)
        
        # Eval model
        pred_today_out, have_eval = evaluate_today(CURRENT_MODEL_FOLDER, CURRENT_EVAL, TODAY, product_list, spc_param_list, spc_wafer_stat, wet_lot_stat,model_options,TRIGGER_STEP)

        
        folders_cleaned = clean_old_data(TRAIN_FOLDER, EVAL_FOLDER, TODAY, expire_days=model_options.loc['DAYS_TILL_EXPIRE'])
        
        if folders_cleaned:
            print("Expired data deleted and models archived.")
    
if __name__ == '__main__':
    main()