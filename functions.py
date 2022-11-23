import numpy as np
import pandas as pd 
from math import sqrt
import os
import sys
import csv
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow 
from skimage import morphology, measure
from skimage.draw import polygon, polygon_perimeter
from scipy.spatial.distance import cdist
from scipy.stats import kurtosis

import pyefd
from pyefd import elliptic_fourier_descriptors, normalize_efd

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import feature_selection as fs
from sklearn import preprocessing

from datetime import datetime

# pay attention to capitalization below!
from spFSR import SpFSR
from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE, BorderlineSMOTE

from itertools import cycle
from random import randint
from random import sample

import xgboost as xgb 

#Funções para normalizar (todos os dados):
## Normaliza dados
def normalize(min, max, value):
    return (value-min)/(max - min)

def normalize_prop(prop, df):
    min = np.min(df[prop].values)
    max = np.max(df[prop].values)
    return (normalize(min, max, df[prop].values))

# Filtra/Normaliza dados
def normalize_dataset(df, n_efd_coeffs):
  dataset = df.copy()
   
  dataset.areaN = normalize_prop('areaN', df)
  dataset.eccenN = normalize_prop('eccenN', df) 
  dataset.extentN = normalize_prop('extentN', df)
  dataset.periN = normalize_prop('periN', df)
  dataset.maxAxN = normalize_prop('maxAxN', df)  
  dataset.minAxN = normalize_prop('minAxN', df)  
  dataset.compacN = normalize_prop('compacN', df)
  dataset.circuN = normalize_prop('circuN', df)
  dataset.convexN = normalize_prop('convexN', df)
  dataset.hAreaN = normalize_prop('hAreaN', df)
  dataset.solidN = normalize_prop('solidN', df) 
  dataset.equidiaN = normalize_prop('equidiaN', df) 
  dataset.elonN = normalize_prop('elonN', df)
  dataset.eN = normalize_prop('eN', df)  
  dataset.kN = normalize_prop('kN', df)  
  dataset.mrdN = normalize_prop('mrdN', df)  
  dataset.ardN = normalize_prop('ardN', df)  
  dataset.fdN = normalize_prop('fdN', df)       
  efds = ['efdN'+str(i) for i in range(1,(n_efd_coeffs*4 + 1 - 3))]
  for efd in efds: 
      dataset[efd] = normalize_prop(efd, df) 
    
  dataset.areaC = normalize_prop('areaC', df)
  dataset.eccenC = normalize_prop('eccenC', df) 
  dataset.extentC = normalize_prop('extentC', df)
  dataset.periC = normalize_prop('periC', df)
  dataset.maxAxC = normalize_prop('maxAxC', df)  
  dataset.minAxC = normalize_prop('minAxC', df)
  dataset.compacC = normalize_prop('compacC', df)
  dataset.circuC = normalize_prop('circuC', df)
  dataset.convexC = normalize_prop('convexC', df)
  dataset.hAreaC = normalize_prop('hAreaC', df)
  dataset.solidC = normalize_prop('solidC', df) 
  dataset.equidiaC = normalize_prop('equidiaC', df) 
  dataset.elonC = normalize_prop('elonC', df)
  dataset.eC = normalize_prop('eC', df)  
  dataset.kC = normalize_prop('kC', df)  
  dataset.mrdC = normalize_prop('mrdC', df)  
  dataset.ardC = normalize_prop('ardC', df)  
  dataset.fdC = normalize_prop('fdC', df)       
  efds = ['efdC'+str(i) for i in range(1,(n_efd_coeffs*4 + 1 - 3))]
  for efd in efds: 
      dataset[efd] = normalize_prop(efd, df)   

  #dataset.nucleus_position = normalize_prop('nucleus_position', df)
  return dataset

#Funções - organiza dados (x, y, ids) e target para diferentes classificadores:
#Monta base e retorna 3 dataframes: data (x), target(2,3 e 6 classes), image/cell_id
def get_database_data_targe_ids(data_normal, data_ascus, 
                       data_lsil, data_asch, data_hsil,data_car,
                       features_to_fit):
 
    data =  pd.DataFrame(data=np.vstack([
                          data_normal.values,
                          data_ascus.values,
                          data_asch.values,
                          data_lsil.values, 
                          data_hsil.values,
                          data_car.values]), 
                         columns = data_car.columns)
    
    ## ID's imagens e celulas
    image_cells_ids= data[['image_id', 'cell_id']].copy() 
    
    ##Ajusta y(target) para classificação binária, ternária além de bethesda
    y = np.array(data['bethesda'].values)
    y_bin = np.array(y)
    y_ter = np.array(y)
        
    for i in range(data.shape[0]):
         y_bin[i] = 0 if y_bin[i]==0 else 1
    
    for i in range(data.shape[0]):
          if y_ter[i] == 3:  ##Lsil
             y_ter[i] = 1
          elif (y_ter[i] == 4 or y_ter[i] == 5):  ##HSIl e Car
                y_ter[i] = 2
                
    target = pd.DataFrame(data = np.stack([y_bin,
                                           y_ter,
                                           y], axis=-1),
                          columns = ['binary', 'ternary', 'bethesda'])
    
    data = data[features_to_fit]      
    return (data, target, image_cells_ids)

## Prepara dados para tuning de parâmetros
## Valores para type: 1 (normal/anormal), 2(baixo/alto grau), 3(ASCUS/LSIL), 4(ASCH/HSIL/CAR)

def filter_dataXY(X, y, cls_type):
    # X e y devem ser do tipo dataframe 
    
    if cls_type == 1: # (normal/anormal)
        return (X, y['binary'])
    elif cls_type == 2: # (baixo/alto grau)
          lines = filter_lines(y['ternary'], [1,2])
          return (X.loc[lines], y['ternary'].loc[lines])  
    elif cls_type == 3: # (ASCUS/LSIL)
          lines = filter_lines(y['bethesda'], [1,3])
          return (X.loc[lines], y['bethesda'].loc[lines])  
    else: #(ASCH/HSIL/Car)    
          lines = filter_lines(y['bethesda'], [2,4,5])
          return (X.loc[lines], y['bethesda'].loc[lines])                           

## Filtra linhas
def filter_lines(y, cls):
   lines = []
   for idx, value in zip(y.index, y.values):
        if (value in cls):
            lines.append(idx)
   return lines

## Filtra dados para teste do classificador 2 com base nas predições do classificador 1:
def filter_Xy_from_cls1_to_cls2(data, target, predics_bin, idx_test):
    lines = []
    for i in idx_test:
        if predics_bin[i] == 1: #Anormal  
             lines.append(i)
            
    X = data.loc[lines]
    y = target['ternary'].loc[lines]
    return (lines, X, y)

## Filtra dados para teste do classificador 3 com base nas predições do classificador 2:
def filter_Xy_from_cls1_to_cls3(data, target, predics_ter, idx_test):
    lines = []
    for i in idx_test:
        if predics_ter[i] == 1:  #lesão de baixo grau
             lines.append(i)
            
    X = data.loc[lines]
    y = target['bethesda'].loc[lines]
    return (lines, X, y)

## Filtra dados para teste do classificador 4 com base nas predições do classificador 2:
def filter_Xy_from_cls2_to_cls4(data, target, predics_ter, idx_test):
    lines = []
    for i in idx_test:
        if predics_ter[i] == 2:  #lesão de baixo grau
             lines.append(i)
            
    X = data.loc[lines]
    y = target['bethesda'].loc[lines]
    return (lines, X, y)

## Seleciona indices com pred = cls de idx_test          
def index_pred_from_class(idx_test, pred_y, cls=0):
    idx = []
    for i, pred in zip(idx_test, pred_y):
        if (pred == cls):
            idx.append(i)
    return idx
 
 
## Contabiliza tempo:
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

# Lista features (N+C, N, C e EFD)
def list_all_features(n_efd_coeffs):
   # n_efd_coeffs: número de coefficientes a considerar (série Eliptica de fourier - EFD) para N e C
     
   feature_labels=['areaN', 'eccenN', 'extentN', 'periN', 'maxAxN', 'minAxN',
                   'compacN', 'circuN', 'convexN', 'hAreaN', 'solidN', 'equidiaN', 
                   'elonN', 'sdnrlN', 'raN', 'riN', 'eN', 'kN', 'mrdN', 'ardN', 'fdN'] 
   
   efdNs = ['efdN'+str(i) for i in range(1, (n_efd_coeffs*4+1 - 3))]  
   for name_f in efdNs:
       feature_labels.append(name_f) 
   
   aux=['areaC', 'eccenC', 'extentC', 'periC', 'maxAxC', 'minAxC',
         'compacC', 'circuC', 'convexC', 'hAreaC', 'solidC', 'equidiaC', 
          'elonC', 'sdnrlC', 'raC', 'riC', 'eC', 'kC', 'mrdC', 'ardC', 'fdC'] 
   for name_f in aux:
       feature_labels.append(name_f)

   efdCs = ['efdC'+str(i) for i in range(1, (n_efd_coeffs*4+1 - 3))]  
   for name_f in efdCs:
       feature_labels.append(name_f)
    
   aux = ['ratio_NC', 'ratio_NC_per', 'ratio_NC_hArea', 'nucleus_position']

   for name_f in aux:
       feature_labels.append(name_f)
   return feature_labels   

def list_all_EFD_features(n_efd_coeffs):
   # n_efd_coeffs: número de coefficientes a considerar (série Eliptica de fourier - EFD) para N e C

   feature_labels = ['efdN'+str(i) for i in range(1, (n_efd_coeffs*4+1 - 3))]      
   efdCs = ['efdC'+str(i) for i in range(1, (n_efd_coeffs*4+1 - 3))]  
   for name_f in efdCs:
       feature_labels.append(name_f)
   return feature_labels   


def list_all_nucleus_features(n_efd_coeffs):
   # n_efd_coeffs: número de coefficientes a considerar (série Eliptica de fourier - EFD) para N e C
     
   feature_labels=['areaN', 'eccenN', 'extentN', 'periN', 'maxAxN', 'minAxN',
                   'compacN', 'circuN', 'convexN', 'hAreaN', 'solidN', 'equidiaN', 
                   'elonN', 'sdnrlN', 'raN', 'riN', 'eN', 'kN', 'mrdN', 'ardN', 'fdN'] 
   
   efdNs = ['efdN'+str(i) for i in range(1, (n_efd_coeffs*4+1 - 3))]  
   for name_f in efdNs:
       feature_labels.append(name_f) 
    
   #TODO: incluir features abaixo? 
   #aux = ['ratio_NC', 'ratio_NC_per', 'ratio_NC_hArea', 'nucleus_position']
   #for name_f in aux:
   #    feature_labels.append(name_f)

   return feature_labels   

def list_all_cyto_features(n_efd_coeffs):
   # n_efd_coeffs: número de coefficientes a considerar (série Eliptica de fourier - EFD) para N e C
     
   feature_labels =['areaC', 'eccenC', 'extentC', 'periC', 'maxAxC', 'minAxC',
         'compacC', 'circuC', 'convexC', 'hAreaC', 'solidC', 'equidiaC', 
          'elonC', 'sdnrlC', 'raC', 'riC', 'eC', 'kC', 'mrdC', 'ardC', 'fdC'] 

   efdCs = ['efdC'+str(i) for i in range(1, (n_efd_coeffs*4+1 - 3))]  
   for name_f in efdCs:
       feature_labels.append(name_f)
    
   #aux = ['ratio_NC', 'ratio_NC_per', 'ratio_NC_hArea', 'nucleus_position']
   #for name_f in aux:
   #    feature_labels.append(name_f)
    
   return feature_labels 

### FEATURES SELECTION: "Simultaneous Perturbation Stochastic Approximation (SPSA) for feature selection and ranking" 
# Fonte: An implementation of feature selection and ranking via SPSA based on the article "K-best feature selection and ranking via stochastic approximation"(https://www.sciencedirect.com/science/article/abs/pii/S0957417422018826) 
# Código: https://github.com/akmand/spFSR.git
def features_selection_spfsr(X_train, y_train, N_features = None):        
    #Atenção: X_train contem apenas colunas de features!
    
    # pred_type needs to be 'c' for classification and 'r' for regression datasets
    sp_engine = SpFSR(x=X_train.values, y=y_train.values, pred_type='c', wrapper=None, scoring='accuracy')
    
    np.random.seed(999)

    if N_features is not None:
        sp_output = sp_engine.run(num_features= N_features, print_freq = 9000000, n_jobs=4).results    
    else:
        sp_output = sp_engine.run(num_features=0, print_freq = 9000000, n_jobs=4).results    

    fs_indices_spfsr = sp_output.get('selected_features')
    best_features_spfsr = X_train.columns[fs_indices_spfsr].values
    feature_importances_spfsr = sp_output.get('selected_ft_importance')
    return(best_features_spfsr, feature_importances_spfsr)

                             
### FEATURES SELECTION: método Mutual Information
def features_selection_mi(X_train, y_train, N_features = 20):    
    
    ## Feature Selection using Mutual Info  
    fs_fit_mutual_info = fs.SelectKBest(fs.mutual_info_classif, k=N_features)
    fs_fit_mutual_info.fit_transform(X_train, y_train)

    # ordena extrai do maior score para o menor entre as n_features mais importantes
    fs_indices_mutual_info = np.argsort(fs_fit_mutual_info.scores_)[::-1][0:N_features] # extrai do maior score para o menor entre as 10 features mais importantes
    best_features_mutual_info = X_train.columns[fs_indices_mutual_info].values  
    feature_importances_mutual_info = fs_fit_mutual_info.scores_[fs_indices_mutual_info]

    best_features_MI = np.asarray(best_features_mutual_info)
    feature_importances_MI = np.asarray(feature_importances_mutual_info, dtype = np.float32)    
    return (best_features_MI, feature_importances_MI)

## Plota gráfico de ganho para features selecionadas: 
def plot_imp(best_features_1, scores_1, method_name_1,
            best_features_2, scores_2, method_name_2):   
    
    plt.style.use("bmh")
    #plt.rcParams.update({'font.size': 12})
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(15, 7))
    axs[0].tick_params(labelsize= 'small')
    axs[0].barh(best_features_1, scores_1, color= 'blue', height=0.75)    
    axs[0].set(xlim=[min(0, (np.min(scores_1))), max(0.8, np.max(scores_1)+0.1)], xlabel='Score', ylabel='Feature', title= method_name_1 + ' Scores')
    axs[1].tick_params(labelsize= 'small')
    axs[1].set(xlim=[min(0, np.min(scores_2)), max(0.8, np.max(scores_2)+0.1)], xlabel='Score', ylabel='Feature', title=method_name_2 + ' Scores')
    axs[1].barh(best_features_2, scores_2, color= 'green')    
    
    #fig.suptitle('Feature Selection') 
    fig.subplots_adjust(left=0.1, right=0.9, wspace=0.3)
    plt.show()
 
 # contabiliza estatísticas (feature scores):
def acum_feature_importances(best_features, best_features_score, acum_dict):    
    for feature, score in zip(best_features, best_features_score):
        acum_dict[feature] = (acum_dict[feature] + score)
    return acum_dict

# Conjunto final de features e scores em ordem decrescente de importância
def resume_feature_importance(acum_dict, N_iter, N_features = None):
    acum_dict_tuples = sorted(acum_dict.items(), key=lambda item:item[1], reverse=True)
    best_features = []
    feature_importances = []
    for i in list(range(len(acum_dict_tuples))):
        if acum_dict_tuples[i][1] == 0:
                break
        if N_features is not None:
            if i == N_features:
                break
        best_features.append(acum_dict_tuples[i][0])
        feature_importances.append(acum_dict_tuples[i][1]/N_iter)

    best_features = np.asarray(best_features)
    feature_importances = np.asarray(feature_importances, dtype = np.float32)    
    return (best_features, feature_importances)
    
#Funções tuning/gridsearch (SVM, RF, XGBoost)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report

def grid_search_SVM(model, params, X, y):
    grid_search = GridSearchCV(
        model, params, scoring= ['accuracy', 'f1_weighted'], refit='f1_weighted'
    )
    grid_search.fit(X, y)
    y_pred = grid_search.predict(X)
    print(classification_report(y, y_pred))
    return (grid_search.best_params_)

def grid_search_RF(model, params, X, y):
    grid_search = GridSearchCV(model, param_grid = params, 
        scoring= 'accuracy')
    grid_search.fit(X, y)
    y_pred = grid_search.predict(X)
    print(classification_report(y, y_pred))
    return (grid_search.best_params_)

def grid_search_XGB(model, params, X, y):
    folds = 5
    param_comb = 150
    skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)
    
    random_search = RandomizedSearchCV(model, param_distributions= params,
                        n_iter=param_comb, scoring='accuracy', n_jobs=-1, 
                        cv=skf.split(X,y), random_state=1001)

    start_time = timer(None) # Tempo inicial
    random_search.fit(X, y)
    timer(start_time)
    
    print('\n Best estimator:')
    print(random_search.best_estimator_)
    print('\n Best score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
    print(random_search.best_score_ )
    results = pd.DataFrame(random_search.cv_results_) 
    
    return random_search.best_params_, results
    

#Funções para classificadores e métricas:
# Gera modelos 
def getModel(params, classifier = 'SVM', class_type = 'binary'):
    if classifier == 'SVM':
          model = SVC(probability = True, random_state=27).set_params(**params)
    elif classifier == 'RF':
          model = RandomForestClassifier(oob_score=True, random_state=27).set_params(**params)
    elif classifier == 'XGBoost':
        if class_type == 'binary':
            model = xgb.XGBClassifier(learning_rate=0.2, n_estimators=200, objective= 'binary:logistic',
                     scale_pos_weight=1, seed=27).set_params(**params)
        else:    # multiclass  
            model = xgb.XGBClassifier(learning_rate=0.2, n_estimators=200, objective= 'multi:softprob',
                     seed=27).set_params(**params) 
    else:
        model = None # 'MLP toDo'    
    return model    
    
 
## Ajusta modelos com Cross validation nos dados de treino com aumento de dados em cada fold (retorna métrica de treino)
def fit_model(X, y, model, cls_type= 1, cv=None):
    """
    Faz upsamples dos dados de teste
    Returns array de métricas de treino
    """
    le = preprocessing.LabelEncoder()
    if cls_type == 1:  ## normal/anormal
            cls = [0,1]
            class_type = 'binary'
            label = 1
    elif cls_type == 2:   ## baixo/alto grau
            cls = [1,2]
            le.fit(cls)
            class_type = 'binary'
            label = 2
    elif cls_type == 3:  ## asc-us/lsil
            cls = [1,3]
            le.fit(cls)
            class_type = 'binary'
            label = 3
    else:              ## asch/hsil/car
            cls = [2,4,5]
            le.fit(cls)
            label= None
            class_type = 'ternary'
 
    smoter = SMOTE(random_state=42)
    accs = precs = recs = specs = f1_scores = aucs = 0 
     
    # Upsample apenas nos dados de treinamento
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=45)
    #X_train, y_train = X,y
    X_train_upsample, y_train_upsample = smoter.fit_resample(X_train, y_train)
    
    ## codifica rótulos em y se classificadores 2, 3 e 4
    if (cls_type != 1):  
        y_train_upsample = le.transform(y_train_upsample.astype(np.int32))
    else:
        y_train_upsample = y_train_upsample.astype(np.int32)
            
    model = model.fit(X_train_upsample, y_train_upsample)            

    # Predição:
    pred_y = model.predict(X_val)
     
    ## decodifica rótulos em y se classificadores tipos 2, 3 e 4:
    if (cls_type!= 1):
        pred_y = le.inverse_transform(pred_y)

    # Calcula e registra métricas p/ fold:
    accs = calc_metric(y_val, pred_y, metric_type='acc', class_type = class_type, pos_label= label, classes=cls)
    precs = calc_metric(y_val, pred_y, metric_type='prec', class_type = class_type, pos_label= label, classes=cls)
    recs = calc_metric(y_val, pred_y, metric_type='rec', class_type = class_type, pos_label= label, classes=cls)
    specs = calc_metric(y_val, pred_y, metric_type='spec', class_type = class_type, pos_label= label, classes=cls)
    f1_scores = calc_metric(y_val, pred_y, metric_type='f1_score', class_type = class_type, pos_label= label, classes=cls)
       
    ## Registra resultados (dataframe):
    metrics = {'acc': accs, 'prec': precs, 'rec': recs, 'spec': specs, 'f1_score': f1_scores}      
    return metrics, model

## Ajusta modelos com Cross validation nos dados de treino com aumento de dados em cada fold (retorna métrica de treino)
def fit_model_old(X, y, model, cls_type= 1, cv=None):
    """
    Cria folds e upsamples dentro de cada fold.
    Returns array de métricas de validação
    """
    le = preprocessing.LabelEncoder()
    if cls_type == 1:  ## normal/anormal
            cls = [0,1]
            class_type = 'binary'
            label = 1
    elif cls_type == 2:   ## baixo/alto grau
            cls = [1,2]
            le.fit(cls)
            class_type = 'binary'
            label = 2
    elif cls_type == 3:  ## asc-us/lsil
            cls = [1,3]
            le.fit(cls)
            class_type = 'binary'
            label = 3
    else:              ## asch/hsil/car
            cls = [2,4,5]
            le.fit(cls)
            label= None
            class_type = 'ternary'
 
    N_SPLITS = 5
    if cv is None:
        cv = StratifiedKFold(n_splits=N_SPLITS, random_state=None)

    smoter = BorderlineSMOTE(random_state=42)
    accs = precs = recs = specs = f1_scores = aucs = np.zeros((N_SPLITS), dtype = np.float64)
    for i, (train_fold_index, val_fold_index) in enumerate(cv.split(X, y)):
 
        # Dados de treinamento
        X_train_fold, y_train_fold = X[train_fold_index], y[train_fold_index]
        # Dados de validação
        X_val_fold, y_val_fold = X[val_fold_index], y[val_fold_index]

        # Upsample apenas nos dados de treinamento
        X_train_fold_upsample, y_train_fold_upsample = smoter.fit_resample(X_train_fold,
                                                                           y_train_fold)
        ## codifica rótulos em y se classificadores 2, 3 e 4
        if (cls_type != 1):  
            y_train_fold_upsample = le.transform(y_train_fold_upsample.astype(np.int32))
        else:
            y_train_fold_upsample = y_train_fold_upsample.astype(np.int32)
            
        # Ajusta parâmetros:
        #if params == None:
            ### fazer aqui?
        model = model.fit(X_train_fold_upsample, y_train_fold_upsample)            
         
        # Predição:
        pred_y = model.predict(X_val_fold)
        
        ## decodifica rótulos em y se classificadores tipos 2, 3 e 4:
        if (cls_type!= 1):
            pred_y = le.inverse_transform(pred_y)
            
        # Calcula e registra métricas p/ fold:
        accs[i] = calc_metric(y_val_fold, pred_y, metric_type='acc', class_type = class_type, pos_label= label, classes=cls)
        precs[i] = calc_metric(y_val_fold, pred_y, metric_type='prec', class_type = class_type, pos_label= label, classes=cls)
        recs[i] = calc_metric(y_val_fold, pred_y, metric_type='rec', class_type = class_type, pos_label= label, classes=cls)
        specs[i] = calc_metric(y_val_fold, pred_y, metric_type='spec', class_type = class_type, pos_label= label, classes=cls)
        f1_scores[i] = calc_metric(y_val_fold, pred_y, metric_type='f1_score', class_type = class_type, pos_label= label, classes=cls)
       
    ## Registra resultados (dataframe):
    metrics = {'acc': np.mean(accs), 'prec': np.mean(precs), 'rec': np.mean(recs), 
                   'spec': np.mean(specs), 'f1_score': np.mean(f1_scores)}      
    return metrics, model

 
# Calcula métricas: (vide metrics_type e classifiers_type)
def calc_metric(target_test, target_predict, metric_type='acc', class_type ='binary', pos_label=1, classes=[0,1]):   
    if (metric_type == 'acc'):
        return accuracy_score(target_test, target_predict)
    elif (metric_type == 'prec'):
         if (class_type == 'binary'):  ## caso classificadores binário
            return  precision_score(target_test, target_predict, pos_label= pos_label, zero_division=0)  
         else:  ## multiclasses
            return precision_score(target_test, target_predict, average='weighted', zero_division=0)
    elif (metric_type == 'rec'):
        if (class_type == 'binary'):  ## classificadores binários
            return recall_score(target_test, target_predict, pos_label= pos_label, zero_division=0)
        else:  ## multiclasses
            return  recall_score(target_test, target_predict, average ='weighted', zero_division=0)
    elif (metric_type == 'spec'):   
         if (class_type == 'binary'):  ## classificadores binários
            tn, fp, fn, tp = confusion_matrix(target_test, target_predict).ravel()
            return tn/(tn + fp)
         else:  ##  multiclasses - média aritmética  
            spec = 0
            for l in classes:
                tn, fp, fn, tp = confusion_matrix((np.array(target_test)==l), (np.array(target_predict)==l)).ravel()
                spec += tn/(tn + fp)
            return spec/len(classes)  
    elif (metric_type == 'f1_score'):      
         if (class_type == 'binary'):  ## classificadores binários
            f1 = f1_score(target_test, target_predict, pos_label= pos_label)
            return f1
         else:  ## multiclasses
            f1 = f1_score(target_test, target_predict, average= 'weighted')
            return f1 
    else:
        return None


def fill_line_metrics_CV(model_name, featur, line_results, metrics, results, class_type='binary'):
    line = pd.Series(data = np.array([class_type, model_name, featur,
             '{:.4f}'.format(metrics['acc']), '{:.4f}'.format(metrics['prec']),
             '{:.4f}'.format(metrics['rec']),'{:.4f}'.format((1- metrics['spec'])), 
             '{:.4f}'.format(metrics['spec']), '{:.4f}'.format(metrics['f1_score'])], dtype = object), 
              index=['Tipo', 'Model', 'Features', 'Acurácia', 'Precisão', 'Sensibil' , 
                     'Falso Pos', 'Especif', 'F1_measure']) 
    results.loc[line_results] = line
    
# Exibe curva ROC para classificadores binários 
def plot_roc_curve_CV(roc_curve_list, labels_list, title = "ROC Curve"):
    fig, ax = plt.subplots(figsize=(9,5))
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "deeppink", "navy", "darkorange"])
    plt.style.use("bmh")
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    for i,color in zip(range(len(roc_curve_list)), colors):
        ax.plot(
            roc_curve_list[i][0],
            roc_curve_list[i][1],
            color=color,
            label=labels_list[i],
            lw=2,
            alpha=0.8,
        )
 
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title= title
    )
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    plt.show()

# Gera grafico matriz confusao  
def make_confusionMatrixDisplay(test, pred, labels, title):
    cm = confusion_matrix(test, pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    return (disp, title)

# Exibe 3 matrizes de confusão uma para cada classificador
def plot_conf_matrix(preds_to_conf_matrix, lbls=[0,1], disp_lbls=['normal', 'anormal']):
    fig, axes = plt.subplots(nrows=1, ncols= 3, figsize=(15,9))

    for i,ax in enumerate(axes.flatten()):
          ConfusionMatrixDisplay.from_predictions(preds_to_conf_matrix[i][0], preds_to_conf_matrix[i][1], 
                                    labels= lbls, cmap='Blues', colorbar=False, ax=ax, display_labels=disp_lbls)
          ax.title.set_text(preds_to_conf_matrix[i][2])
    plt.tight_layout()  
    plt.show()