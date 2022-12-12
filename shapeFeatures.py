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
from itertools import cycle
from random import randint
from random import sample
import xgboost as xgb 
import csv

#TODO: Fazer acesso do atributos da classe abaixo nas funções
class CRIC_images:
    def __init__(self):
        self.IMG_W = 1376
        self.IMG_H = 1020
        self.Bethesda_classes = {'Normal':0, 'ASC-US':1, 'ASC-H':2, 'LSIL':3,'HSIL':4, 'Invasive Carcinoma':5} 

# Monta dataframe de features por célula (núcleo e citoplasma):
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

## Extração de features:
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
def efd_feature(contour, n_coeffs):
    '''  contour: pontos de borda
         n_coeffs: nº de coeficientes da serie de fourier (X(sen e cos) para Y(sen e cos))
         retorno: tupla (coeficientes, número de coeficientes)
    ''' 
    coeffs = elliptic_fourier_descriptors(contour, order= n_coeffs, normalize=True)
    return (coeffs.flatten()[3:(n_coeffs*4+1)], (n_coeffs*4+1 - 3))

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
        
def get_list_feature_labels(n_efd_coeffs):
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

def create_dictionary_features(n_efd_coeffs):
   # n_efd_coeffs: número de coefficientes a considerar (série Eliptica de fourier - EFD) para N e C
   feature_labels=['image_id', 'cell_id']
   features = get_list_feature_labels(n_efd_coeffs)
   for name in features:
      feature_labels.append(name)
        
   feature_labels.append('bethesda')     

   aux = [[] for i in range(len(feature_labels))]
   return dict(zip(feature_labels, aux))

def set_efd_coeff(coeffs, dictio, efd='N'):
    if efd == 'N':
       efds = ['efdN'+str(i) for i in range(1, coeffs.shape[0]+ 1)] 
    else:
       efds = ['efdC'+str(i) for i in range(1, coeffs.shape[0]+ 1)] 
    for feat, value in zip(efds, coeffs):
        dictio[feat].append(value)

## Gera dataframe de características (1 linha por célula):
# Para cada célula (identificação e features)
# Calula, registra e retorna dataframe de 120 features de formato e contorno para nucleo (N) e citoplasma (C)
def make_stats(df_nucleos, df_cyto, df_nucleos_full, df_cyto_full, n_efd_coeffs= 10):
   """ 
     Features (N e C):  
           area, excentricidade, extenção, perímetro, major Axis, minor Axis,
           compacidade, circularidade, convexidade, area do convex hull', solidicidade, 
           diâmetro equivalente, alongamento, SDNRL, RA, RI, entropia das distâncias radiais (RD),
           kurtosis das distâncias radiais, maior RD, average RD, FD (dimensão fractal), 
           'n_efd_coeffs' 1º coeficientes da série EF (elliptic fourier descriptor)
     Features relação N/C: razão area N/C, razão perimetro N/C,razão area convex hull N/C, posição relativa do nucleo
     
   """ 
   img = CRIC_images()
 
   count_cells = np.zeros(6, dtype = int)
    
   data = create_dictionary_features(n_efd_coeffs)
   for image_id, cell_id in df_nucleos.values:   
        cell = f'{image_id:05d}_{cell_id:05d}_'
        
        points_nucleos = df_nucleos_full.query('image_id == '+str(image_id) + ' and cell_id == '+str(cell_id))[['x', 'y']].values
        points_cyto = df_cyto_full.query('image_id == '+str(image_id) + ' and cell_id == '+str(cell_id))[['x', 'y']].values
        bethesda = img.Bethesda_classes[df_nucleos_full.query('image_id == '+str(image_id) + ' and cell_id == '+str(cell_id))['bethesda_system'].values[0]]
                
        # Nucleos mask
        mask_nucleo =  np.zeros((img.IMG_H, img.IMG_W), dtype=np.uint8)
        # Cytoplasma mask
        mask_cyto =  np.zeros((img.IMG_H, img.IMG_W), dtype=np.uint8)
        
        # Nucleos contour points
        rrN, ccN = polygon(points_nucleos[:,1], points_nucleos[:,0])
        mask_nucleo[rrN, ccN] = 1    
        
        # Cytoplasm contour points
        rrC, ccC = polygon(points_cyto[:,1], points_cyto[:,0])
        mask_cyto[rrC, ccC] = 1    
        
        # Eliminate duplicate contour points: 
        border_coords_N = [[ri, ci] for ri,ci in zip(rrN, ccN)]
        border_coords_C = [[ri, ci] for ri,ci in zip(rrC, ccC)]
        points_N = np.array(border_coords_N, dtype=np.float64)
        points_N = np.unique(points_N, axis=0)
        points_C = np.array(border_coords_C, dtype=np.float64)
        points_C = np.unique(points_C, axis=0)
           
        # Calc regionprops metrics from Nucleos mask:
        m_N = measure.regionprops(mask_nucleo)  
        
        # Calc regionprops metrics from Cyto mask:
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
    
        #Calc rations to nucleus and cytoplasm areas 
        ratio_NC = area_N/area_C
        ratio_NC_per = per_N/per_C
        ratio_NC_hArea = hAreaN/hAreaC

        # Registry metrics on data dict:
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
        efd_coeffs_N, _ = efd_feature(points_N, n_efd_coeffs)
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
        efd_coeffs_C, _ = efd_feature(points_C, n_efd_coeffs)
        set_efd_coeff(efd_coeffs_C, data, efd='C')         
        
        data['ratio_NC'].append(ratio_NC)
        data['ratio_NC_per'].append(ratio_NC_per)
        data['ratio_NC_hArea'].append(ratio_NC_hArea)
        data['nucleus_position'].append(nucleus_pos)
    
        data['bethesda'].append(bethesda)
        
        count_cells[bethesda]+=1
        
   df = pd.DataFrame(data)
   return (count_cells, df)  

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
