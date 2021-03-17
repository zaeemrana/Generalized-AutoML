import pandas as pd
import numpy as np
import os, sys
from datetime import datetime, timedelta
import json

from model_train import train_model

#from model_prep import make_json_train, check_model_dates

def check_model_dates(is_multi_day, days_to_update_after, TODAY):
    if is_multi_day:

        return datetime.strptime(TODAY, "%Y-%m-%d")  > timedelta(days_to_update_after) +
    else:
        return True

def prepare_model(PROJ_FOLDER,CONFIG_FOLDER,TODAY,OLD_MODEL_DATE,options):
    
    is_multi_day, days_back_to_pull, days_to_update_after, days_till_expire, model_type = options.values[[5,6,7,8,2]]

    TRAIN_FOLDER, EVAL_FOLDER, CURRENT_TRAIN, CURRENT_EVAL = get_dirs(PROJ_FOLDER, TODAY)
    
    CURRENT_MODEL_FOLDER, have_model = check_dir(TRAIN_FOLDER, EVAL_FOLDER, TODAY, days_to_update_after, model_type)

    check_model_dates(is_multi_day, days_to_update_after, TODAY)

    if not have_model:
        print("Preparing models.")
        if not os.path.isfile(CURRENT_MODEL_FOLDER + 'train_logs.txt'):
            f = open('train_logs.txt', 'a+')
            f.close()
    
        have_model = train_model(CURRENT_MODEL_FOLDER, model_type)
    else:
        print("Already have model.")
    
    return TRAIN_FOLDER, EVAL_FOLDER, CURRENT_MODEL_FOLDER, CURRENT_EVAL

def check_dir(CURRENT_TRAIN, CURRENT_EVAL, TODAY, days_to_update_after, model_type):
    
    have_model = False
    
    LATEST_MODEL = check_model_dates(CURRENT_TRAIN, TODAY, days_to_update_after)
    
    CURRENT_MODEL_FOLDER = CURRENT_TRAIN + LATEST_MODEL + "/"

    # Check for models
    if os.path.exists(CURRENT_MODEL_FOLDER + 'models/'):
        for f in os.listdir(CURRENT_MODEL_FOLDER + 'models/'):
            if f.find(".p") != -1:
                have_model = True
            
    else:
        os.mkdir(CURRENT_MODEL_FOLDER)
        
    return CURRENT_MODEL_FOLDER, have_model

def get_dirs(PROJ_FOLDER,TODAY):
    
    TRAIN_FOLDER = PROJ_FOLDER + 'train/'
    EVAL_FOLDER = PROJ_FOLDER + 'eval/'
    CURRENT_TRAIN = TRAIN_FOLDER + TODAY +'/'
    CURRENT_EVAL = EVAL_FOLDER + TODAY + '/'
    
    for dir_ in [TRAIN_FOLDER, EVAL_FOLDER, CURRENT_EVAL]:
        if not os.path.exists(dir_):
            os.mkdir(dir_)
    
    return TRAIN_FOLDER, EVAL_FOLDER, CURRENT_TRAIN, CURRENT_EVAL