import numpy as np
import pandas as pd 
from math import sqrt
import os
import sys
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
from itertools import cycle
from random import randint
from random import sample
import xgboost as xgb 
import csv

### DEFINIÇÃO de constantes:
class const:
    Bethesda_classes = {'Normal':0, 'ASC-US':1, 'ASC-H':2, 'LSIL':3,'HSIL':4, 'Invasive Carcinoma':5} 
    Bethesda_idx_classes = {0: 'Normal', 1:'ASC-US', 2:'ASC-H', 3:'LSIL',4: 'HSIL', 5:'Invasive Carcinoma'} 
    IMG_W = 1376
    IMG_H = 1020
 
### LEITURA DA BASE CRIC: cell_id de núcleo/citoplasma
# Gera lista de imagens e células segmentadas base CRIC(image_id, cell_id) 
def list_cells(nucleos_csv, cyto_csv): 
   df_nucleos_full = pd.read_csv(nucleos_csv, header=0)
   df_cyto_full = pd.read_csv(cyto_csv, header = 0)
   
   # dataframe of unique cells (nucleos)
   df_nucleos = df_nucleos_full[['image_id', 'cell_id']]
   df_nucleos = df_nucleos.sort_values(by=['image_id', 'cell_id']) 
   df_nucleos = df_nucleos.drop_duplicates(subset=['image_id', 'cell_id'], keep='first', inplace=False) 
    
   # dataframe of unique cells (cytoplams)
   df_cyto = df_cyto_full[['image_id', 'cell_id']]
   df_cyto = df_cyto.sort_values(by=['image_id', 'cell_id']) 
   df_cyto= df_cyto.drop_duplicates(keep='first', inplace=False) 
    
   return (df_nucleos, df_cyto,df_nucleos_full, df_cyto_full)

### FEATURE EXTRACTION  

# Calcula localização relativa do núcleo dentro do citoplasma (versão Daniela):
'''def nucleus_to_cyto_border_dist(cent_N, cent_C, border_coords_C, border_coords_N):
    dists_bound_C= cdist(border_coords_C, list([cent_N]), metric='euclidean') 
    dist_bound_C = np.min(dists_bound_C)
    nearest_point = border_coords_C[np.argmin(dists_bound_C)]
    
    dists_bounds_N_to_C = cdist(border_coords_N, list([nearest_point]), metric='euclidean')
    dist_bounds_N_to_C = np.min(dists_bounds_N_to_C)
    
    #print(nearest_point)
    normal_dist = np.sqrt((nearest_point[0]- cent_C[0])**2 + (nearest_point[1]- cent_C[1])**2)
    #print('dist C: ', normal_dist)
    return dist_bounds_N_to_C/normal_dist
    '''

# Calcula localização relativa do núcleo dentro do citoplasma (versão \cite{Mariarputham2015}):
def nucleus_position(cent_N, cent_C, minAxC):
    d = np.sqrt((cent_N[0]-cent_C[0])**2 + (cent_N[1]-cent_C[1])**2)
    if d == 0:
        return 0
    else: 
        return (d/minAxC)  #distância entre os centroides (N - C)/eixo menor do citoplasma

## Para chamada ao metodo que approxima o contorno com a série de Fourier series, como descrito em (https://www.sci.utah.edu/~gerig/CS7960-S2010/handouts/Kuhl-Giardina-CGIP1982.pdf)
## Fonte: https://pyefd.readthedocs.io/en/latest/#second
def efd_feature(contour):
    coeffs = elliptic_fourier_descriptors(contour, order=10, normalize=True)
    return coeffs.flatten()[3:]

# Calcula Dimensão Fractal
# From: https://github.com/jankaWIS/fractal_dimension_analysis/blob/main/fractal_analysis_fxns.py
def fractal_dimension(Z, threshold=0.9):
    """
    calculate fractal dimension of an object in an array defined to be above certain threshold as a count of squares
    with both black and white pixels for a sequence of square sizes. The dimension is the a coefficient to a poly fit
    to log(count) vs log(size) as defined in the sources.
    :param Z: np.array, must be 2D
    :param threshold: float, a thr to distinguish background from foreground and pick up the shape, originally from
    (0, 1) for a scaled arr but can be any number, generates boolean array
    :return: coefficients to the poly fit, fractal dimension of a shape in the given arr
    """
    # Only for 2d image
    assert (len(Z.shape) == 2)

    # Transform Z into a binary array
    Z = (Z < threshold)

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2 ** np.floor(np.log(p) / np.log(2))

    # Extract the exponent
    n = int(np.log(n) / np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2 ** np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def boxcount(Z, k):
    """
    returns a count of squares of size kxk in which there are both colours (black and white), ie. the sum of numbers
    in those squares is not 0 or k^2
    Z: np.array, matrix to be checked, needs to be 2D
    k: int, size of a square
    """
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
        np.arange(0, Z.shape[1], k), axis=1)  # jumps by powers of 2 squares

    # We count non-empty (0) and non-full boxes (k*k)
    #return len(np.where((S > 0) & (S < k * k))[0])
    return len(np.where(S > 0)[0])

# Calcula o alongamento da célula a partir da sua bounding box:
def elongation(bbox):
     l = (bbox[2] - bbox[0])  # maxRow - minRow 
     w = (bbox[3] - bbox[1])  # maxCol - minCol
     return (1 - (w/l))


#Calcula as distâncias radiais (algumas métricas de \cite{Po-HsiangTsui2010a, e \cite{Chiang2001} para CA de mama})
def radial_distances_stats(centroid, border_coords):
    ''' Retorna:
        desvio padrão da distancia radial (SDNRL)
        tx de área: porcentagem de area fora da distância radial média
        RI: indice de rugosidade
        E : entropia do histograma do comprimento radial normalizado representa a redondeza e a rugosidade.
        k: kurtosis do histograma das 
        MRD: comprimento radial máximo (maximum length from center of gravity to perimeter) - valores absolutos (não normalizados)
        ARD: comprimento radial médio (average length from center of gravity to perimeter) - valores absolutos (não normalizados)
    '''    
    dis= cdist(border_coords, list([centroid]), metric='euclidean') 
    dis_Norm = dis/np.max(dis)
    mean = np.mean(dis_Norm)
    SDNRL= np.std(dis_Norm, ddof=1) 
    
    # Calcula taxa de area, soma di - dis[i+1]:
    N = dis_Norm.shape[0]
    area_out = 0
    sum = 0
    for i in range(N):
        if dis_Norm[i] >=mean: 
           area_out +=(dis_Norm[i] - mean)   
        if i != (N-1):
            sum += np.abs(dis_Norm[i] - dis_Norm[i+1])
    RA = area_out /(N*mean)
    
    # Calcula índice de rugosidade(RI):
    RI = sum/N
    
    # Calcula a entropia (E) do histograma dis_Norm
    hist, _ = np.histogram(dis, bins=100, density=True, )
    E = 0
    for p in hist: 
       if p!= 0:
          E+=(p*np.log(p))
    K = kurtosis(dis)   
    return ({'SDNRL': SDNRL, 'RA': RA[0], 'RI': RI[0], 'E':-E, 'K': K, 'MRD': np.max(dis), 'ARD': np.mean(dis)})


### Funções para rótular as 120 features e gerar dicionário p/ Extração:
def get_list_feature_labels():
   feature_labels=['areaN', 'eccenN', 'extentN', 'periN', 'maxAxN', 'minAxN',
                   'compacN', 'circuN', 'convexN', 'hAreaN', 'solidN', 'equidiaN', 
                   'elonN', 'sdnrlN', 'raN', 'riN', 'eN', 'kN', 'mrdN', 'ardN', 'fdN'] 
   efdNs = ['efdN'+str(i) for i in range(1,38)]  
   
   # 37 coefficientes da série Eliptica de fourier (vide EFD)  para borda N
   for name_f in efdNs:
       feature_labels.append(name_f) 
   
   aux=['areaC', 'eccenC', 'extentC', 'periC', 'maxAxC', 'minAxC',
         'compacC', 'circuC', 'convexC', 'hAreaC', 'solidC', 'equidiaC', 
          'elonC', 'sdnrlC', 'raC', 'riC', 'eC', 'kC', 'mrdC', 'ardC', 'fdC'] 
   for name_f in aux:
       feature_labels.append(name_f)
   efdCs = ['efdC'+str(i) for i in range(1,38)]  

   # 37 coefficientes da série Eliptica de fourier (vide EFD) para borda C
   for name_f in efdCs:
       feature_labels.append(name_f)
    
   aux = ['ratio_NC', 'ratio_NC_per', 'ratio_NC_hArea', 'nucleus_position']

   for name_f in aux:
       feature_labels.append(name_f)
   return feature_labels   

def create_dictionary_features():
   feature_labels=['image_id', 'cell_id']
   features = get_list_feature_labels()
   for name in features:
      feature_labels.append(name)
        
   feature_labels.append('bethesda')     

   aux = [[] for i in range(len(feature_labels))]
   return dict(zip(feature_labels, aux))

# Preenche dictio com os coeficientes 'coeffs' conforme nome (gerados em ordem):
def set_efd_coeff(coeffs, dictio, efd='N'):
    if efd == 'N':
       efds = ['efdN'+str(i) for i in range(1,38)] 
    else:
       efds = ['efdC'+str(i) for i in range(1,38)] 
    for feat, value in zip(efds, coeffs):
        dictio[feat].append(value)

# Calcula, registra e retorna dataframe de 120 features de formato e contorno para 
# nucleo (N) e citoplasma (C)
def make_stats(df_nucleos, df_cyto, df_nucleos_full, df_cyto_full, const):
   """ 
     Features (N e C):  
           area, excentricidade, extenção, perímetro, major Axis, minor Axis,
           compacidade, circularidade, convexidade, area do convex hull', solidicidade, 
           diâmetro equivalente, alongamento, SDNRL, RA, RI, entropia das distâncias radiais (RD),
           kurtosis das distâncias radiais, maior RD, average RD, FD (dimensão fractal), 
           37 coeficientes EF (elliptic fourier descriptor)
     Features relação N/C: razão area N/C, razão perimetro N/C,razão area convex hull N/C, posição relativa do nucleo
   """ 
   count_cells = np.zeros(6, dtype = int)
    
   data = create_dictionary_features()
   for image_id, cell_id in df_nucleos.values:   
 
        cell = f'{image_id:05d}_{cell_id:05d}_'
        
        points_nucleos = df_nucleos_full.query('image_id == '+str(image_id) + ' and cell_id == '+str(cell_id))[['x', 'y']].values
        points_cyto = df_cyto_full.query('image_id == '+str(image_id) + ' and cell_id == '+str(cell_id))[['x', 'y']].values
        bethesda = const.Bethesda_classes[df_nucleos_full.query('image_id == '+str(image_id) + ' and cell_id == '+str(cell_id))['bethesda_system'].values[0]]
             
        mask_nucleo =  np.zeros((const.IMG_H, const.IMG_W), dtype=np.uint8)
        mask_cyto =  np.zeros((const.IMG_H, const.IMG_W), dtype=np.uint8)
        
        # Nucleos mask
        rrN, ccN = polygon(points_nucleos[:,1], points_nucleos[:,0])
        mask_nucleo[rrN, ccN] = 1    
        
        # Cytoplasm mask
        rrC, ccC = polygon(points_cyto[:,1], points_cyto[:,0])
        mask_cyto[rrC, ccC] = 1    
        
        # Pontos das bordas
        border_coords_N = [[ri, ci] for ri,ci in zip(rrN, ccN)]
        border_coords_C = [[ri, ci] for ri,ci in zip(rrC, ccC)]
        
        ## Exclui pontos duplicados das bordas para calcular EFD
        points_N = np.array(border_coords_N, dtype=np.float64)
        points_N = np.unique(points_N, axis=0)
        
        points_C = np.array(border_coords_C, dtype=np.float64)
        points_C = np.unique(points_C, axis=0)
           
        # Calc metrics from Nucleos mask:
        m_N = measure.regionprops(mask_nucleo)  
        
        # Calc metrics from Cyto mask:
        m_C = measure.regionprops(mask_cyto)
 
        # Get metrics:
        area_N = m_N[0].area
        eccenN = m_N[0].eccentricity 
        extentN = m_N[0].extent   # area / area da bounding box (bbox)
        per_N = m_N[0].perimeter
        max_N = m_N[0].axis_major_length
        min_N = m_N[0].axis_minor_length 
        cent_N = m_N[0].centroid
        compacN = np.power(per_N,2)/area_N    # perimetro^2/area
        circulN = (4*np.pi*area_N)/np.power(per_N,2)    # 4pi*area/perimetro^2
        convexN = measure.perimeter(m_N[0].image_convex)/per_N  #  perimetro do convex hull/perimetro
        hAreaN = m_N[0].area_convex 
        solidN =  m_N[0].solidity  # area / area do convex hull
        equidiaN  = m_N[0].equivalent_diameter_area # diametro equivalente de um círculo de mesma area
        elonN = elongation(m_N[0].bbox)  # (1 - w/L), onde w é a largura em colunas da bbox e L a altura da bbox
        rdN= radial_distances_stats(cent_N, points_N) # estatísticas de distancia radial (centroid borda) p/ Nucleo
        
        area_C = (m_C[0].area - area_N)
        eccenC = m_C[0].eccentricity
        extentC = m_C[0].extent
        per_C = m_C[0].perimeter
        max_C = m_C[0].axis_major_length
        min_C = m_C[0].axis_minor_length 
        cent_C = m_C[0].centroid
        compacC = np.power(per_C,2)/area_C
        convexC = measure.perimeter(m_C[0].image_convex)/per_C
        circulC = (4*np.pi*area_C)/np.power(per_C,2)
        hAreaC = m_C[0].area_convex
        solidC =  m_C[0].solidity
        equidiaC  = m_C[0].equivalent_diameter_area
        elonC = elongation(m_C[0].bbox)
        rdC= radial_distances_stats(cent_C, points_C) # estatísticas de distancia radial (centroid borda) p/ Cito
    
        # Calcula posição relativa do núcleo no citoplasma
        nucleus_pos= nucleus_position(cent_N, cent_C, min_C)
    
        ratio_NC = area_N/area_C
        ratio_NC_per = per_N/per_C
        ratio_NC_hArea = hAreaN/hAreaC

        # Adiciona métricas:
        data['image_id'].append(image_id)
        data['cell_id'].append(cell_id)
        data['areaN'].append(area_N)
        data['eccenN'].append(eccenN)
        data['extentN'].append(extentN) 
        data['periN'].append(per_N)
        data['maxAxN'].append(max_N)
        data['minAxN'].append(min_N)
        data['compacN'].append(compacN)
        data['circuN'].append(circulN)
        data['convexN'].append(convexN)
        data['hAreaN'].append(hAreaN)
        data['solidN'].append(solidN) 
        data['equidiaN'].append(equidiaN)
        data['elonN'].append(elonN) 
        data['sdnrlN'].append(rdN['SDNRL'])
        data['raN'].append(rdN['RA'])
        data['riN'].append(rdN['RI'])
        data['eN'].append(rdN['E'])
        data['kN'].append(rdN['K'])
        data['mrdN'].append(rdN['MRD'])
        data['ardN'].append(rdN['ARD'])
        data['fdN'].append(fractal_dimension(mask_nucleo))
        efd_coeffs_N = efd_feature(points_N)
        set_efd_coeff(efd_coeffs_N, data)
            
        data['areaC'].append(area_C)
        data['eccenC'].append(eccenC)
        data['extentC'].append(extentC) 
        data['periC'].append(per_C)
        data['maxAxC'].append(max_C)
        data['minAxC'].append(min_C)
        data['compacC'].append(compacC)
        data['circuC'].append(circulC)
        data['convexC'].append(convexC)
        data['hAreaC'].append(hAreaC)
        data['solidC'].append(solidC)
        data['equidiaC'].append(equidiaC)
        data['elonC'].append(elonC)
        data['sdnrlC'].append(rdC['SDNRL'])
        data['raC'].append(rdC['RA'])
        data['riC'].append(rdC['RI'])
        data['eC'].append(rdC['E'])
        data['kC'].append(rdC['K'])
        data['mrdC'].append(rdC['MRD'])
        data['ardC'].append(rdC['ARD'])
        data['fdC'].append(fractal_dimension(mask_cyto))        
        efd_coeffs_C = efd_feature(points_C)
        set_efd_coeff(efd_coeffs_C, data, efd='C')         
        
        data['ratio_NC'].append(ratio_NC)
        data['ratio_NC_per'].append(ratio_NC_per)
        data['ratio_NC_hArea'].append(ratio_NC_hArea)
        data['nucleus_position'].append(nucleus_pos)
    
        data['bethesda'].append(bethesda)
        
        count_cells[bethesda]+=1
        
   df = pd.DataFrame(data)
   return (count_cells, df)  
 

 ## Normaliza dados
def normalize(min, max, value):
    return (value-min)/(max - min)

def normalize_prop(prop, df):
    min = np.min(df[prop].values)
    max = np.max(df[prop].values)
    return (normalize(min, max, df[prop].values))

# Filtra/normaliza dados
def normalize_dataset(df):
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
  efds = ['efdN'+str(i) for i in range(1,38)]
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
  efds = ['efdC'+str(i) for i in range(1,38)]
  for efd in efds: 
      dataset[efd] = normalize_prop(efd, df)   

  #dataset.nucleus_position = normalize_prop('nucleus_position', df)
  return dataset


  # Executa e retorna a predição do modelo KNN:
def execute_KNN(data_train, target_train, data_test, n_nbors):     
        model_knn = KNeighborsClassifier(n_neighbors= n_nbors)      ## instancia
        model_knn.fit(data_train, target_train)                     ## treino 
        target_prediction = model_knn.predict(data_test)            ## teste  
        return target_prediction
    
# Executa e retorna a predição para SVM:
def execute_SVM(data_train, target_train, data_test, kn='rbf', degr=3):     
        model_svm = SVC(kernel= kn, degree=degr) 
        model_svm.fit(data_train, target_train)
        target_pred = model_svm.predict(data_test)
        return target_pred
    
# Executa e retorna a predição para Naive:
def execute_NB(data_train, target_train, data_test): 
       model_nb = GaussianNB()
       model_nb.fit(data_train, target_train)
       return model_nb.predict(data_test)
              
# Gera modelo RandomForest
def fit_get_RF(data_train, target_train): 
    model =RandomForestClassifier()
    model.fit(data_train, target_train)
    return model

def getModel(classifier = 'SVM', class_type = 'binary'):
    if classifier == 'SVM':
        model = SVC(kernel= 'rbf', degree=3, probability=True) 
    elif classifier == 'RF':
        model =RandomForestClassifier()
    elif classifier == 'XGBoost':
        if class_type == 'binary':
            model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
        else:    # multiclass  
            model = xgb.XGBClassifier(objective="multi:softprob", random_state=42)
    else:
        model = None # 'MLP toDo'    
    return model    
    

# Calcula métricas: (vide metrics_type e classifiers_type)
def calc_metric(target_test, target_predict, metric_type='acc', class_type ='binary', classes=[0,1]):   
    if (metric_type == 'acc'):
        return accuracy_score(target_test, target_predict)
    elif (metric_type == 'prec'):
         if (class_type == 'binary'):  ## caso classificadores binário
            return  precision_score(target_test, target_predict)  
         else:  ## multiclasses
            return precision_score(target_test, target_predict, average='weighted')
    elif (metric_type == 'rec'):
        if (class_type == 'binary'):  ## classificadores binários
            return recall_score(target_test, target_predict)
        else:  ## multiclasses
            return  recall_score(target_test, target_predict, average ='weighted')
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
            f1 = f1_score(target_test, target_predict)
            return f1
         else:  ## multiclasses
            f1 = f1_score(target_test, target_predict, average= 'weighted')
            return f1 
    else:
        return None

def fill_line_metrics(model_name, featur, target, pred, line_results, class_type='binary', pos_lb=1):
    accur = calc_metric(target, pred, 'acc', class_type)
    prec = calc_metric(target, pred, 'prec', class_type, pos_lbl= pos_lb)
    sens = calc_metric(target, pred, 'sens', class_type)
    espe = calc_metric(target, pred, 'espec', class_type)
    f1Mes = calc_metric(target, pred, 'f1_mesure', class_type, pos_lbl= pos_lb)

    line = pd.Series(data = np.array([class_type, model_name, featur,
                        '{:.4f}'.format(accur), '{:.4f}'.format(prec), '{:.4f}'.format(sens),
                        '{:.4f}'.format((1- espe)), '{:.4f}'.format(espe), '{:.4f}'.format(f1Mes)], dtype = object), 
                index=['Tipo', 'Model', 'Features', 'Acurácia', 'Precisão', 'Sensibil' , 'Falso Pos', 'Especif', 'F1_measure']) 
    results.loc[line_results] = line

def fill_line_metrics_CV(model_name, featur, line_results, metrics, results, class_type='binary'):
    line = pd.Series(data = np.array([class_type, model_name, featur,
             '{:.4f}'.format(metrics['acc']), '{:.4f}'.format(metrics['prec']),
             '{:.4f}'.format(metrics['rec']),'{:.4f}'.format((1- metrics['spec'])), 
             '{:.4f}'.format(metrics['spec']), '{:.4f}'.format(metrics['f1_score'])], dtype = object), 
              index=['Tipo', 'Model', 'Features', 'Acurácia', 'Precisão', 'Sensibil' , 
                     'Falso Pos', 'Especif', 'F1_measure']) 
    results.loc[line_results] = line
    
    
def plot_roc_curve(roc_curve_list, labels_list):
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "deeppink", "navy", "darkorange"])
    plt.style.use("bmh")
    for i,color in zip(range(len(roc_curve_list)), colors):
        plt.plot(
            roc_curve_list[i][0],
            roc_curve_list[i][1],
            color=color,
            lw=2,
            label= labels_list[i],
        )
    plt.plot([0, 1] , c=".7", ls="--")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Normal vs anormal Random Forest) ')
    plt.legend(fontsize= 'medium')
    plt.show()
 

## Plota gráfico de ganho para features selecionadas: 
def plot_imp(best_features_1, scores_1, method_name_1,
            best_features_2, scores_2, method_name_2):   
    plt.style.use("bmh")
    #plt.rcParams.update({'font.size': 12})
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(15, 7))
    axs[0].tick_params(labelsize= 'small')
    axs[0].barh(best_features_1, scores_1, color= 'blue', height=0.75)    
    axs[0].set(xlim=[0,0.9], xlabel='Score', ylabel='Feature', title= method_name_1 + ' Scores')
    axs[1].tick_params(labelsize= 'small')
    axs[1].set(xlim=[0,0.2], xlabel='Score', ylabel='Feature', title=method_name_2 + ' Scores')
    axs[1].barh(best_features_2, scores_2, color= 'green')    
    
    #fig.suptitle('Feature Selection') 
    fig.subplots_adjust(left=0.1, right=0.9, wspace=0.3)
    plt.show()

# Gera grafico matriz confusao  
def make_confusionMatrixDisplay(test, pred, labels, title):
    cm = confusion_matrix(test, pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    return (disp, title)

### FEATURES SELECTION: método Mutual Information
def features_selection(data_normal, data_ascus, data_lsil, data_asch, data_hsil,data_car, n_normal=(77*5), n_ascus=77, n_lsil=77, n_asch=77, n_hsil=77, n_car=77, n_features = 20):

    feat = get_list_feature_labels()
    
    aux = [0.0 for i in feat]
    features_importances = dict(zip(feat, aux))
    
    for i in list(range(30)):
        sorted_idx_Normal = sample(range(0, data_normal.shape[0]), n_normal)
        sorted_idx_Normal.sort()

        data_cl0  = pd.DataFrame(data=data_normal.loc[sorted_idx_Normal].values, columns = data_normal.columns)

        data_cl1 =  pd.DataFrame(data=np.vstack([data_ascus.loc[sample(range(0, data_ascus.shape[0]), n_ascus)].values,
                                   data_asch.loc[sample(range(0, data_asch.shape[0]), n_asch)].values,
                                   data_lsil.loc[sample(range(0, data_lsil.shape[0]), n_lsil)].values, 
                                   data_hsil.loc[sample(range(0, data_hsil.shape[0]), n_hsil)].values,
                                   data_car.loc[sample(range(0, data_car.shape[0]), n_car)].values]), 
                                 columns = data_car.columns)
        
        # Normaliza dados
        data_cl0 = normalize_dataset(data_cl0)
        data_cl1 = normalize_dataset(data_cl1)

        data_x = np.concatenate([data_cl0[feat].values, 
                                 data_cl1[feat].values], axis=0)
         
        data_y = np.zeros(data_x.shape[0], dtype = np.int32)
        #data_y[385:(385*2)] = 1
        data_y[385:(385*2)] = data_cl1['bethesda'].values
        
        ## Feature Selection using Mutual Info  
        fs_fit_mutual_info = fs.SelectKBest(fs.mutual_info_classif, k=n_features)
        fs_fit_mutual_info.fit_transform(data_x, data_y)
        # ordena extrai do maior score para o menor entre as n_features mais importantes
        fs_indices_mutual_info = np.argsort(fs_fit_mutual_info.scores_)[::-1][0:n_features] 
        best_features_mutual_info = data_cl0[feat].columns[fs_indices_mutual_info]
        feature_importances_mutual_info = fs_fit_mutual_info.scores_[fs_indices_mutual_info]
        
        #contabiliza estatísticas:
        for feature, score in zip(best_features_mutual_info,              
                                  feature_importances_mutual_info):
            #features_importances[feature] =+ score
            features_importances[feature] =+ 1

    return  features_importances

## Feature Selection using Random Forest:
#model_rfi = RandomForestClassifier(n_estimators=100)
#model_rfi.fit(data_x, data_y)
#fs_indices_rfi = np.argsort(model_rfi.feature_importances_)[::-1][0:N_FEATURES]
#best_features_rfi = df.columns[fs_indices_rfi].values
#feature_importances_rfi = model_rfi.feature_importances_[fs_indices_rfi]
#best_features_rfi, feature_importances_rfi