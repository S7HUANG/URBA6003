# -*- coding: utf-8 -*-
"""
Created on Thu April 13 21:30:46 2022

"""

import os
import time
import math
import numpy as np
from scipy import misc
from osgeo import gdal, gdalconst
from sklearn.decomposition import PCA
from os.path import join

base_dir = os.path.dirname(os.path.abspath(__file__))

def get_ndvi(red, nir, threshold):
    r, c = red.shape
    ndvi = np.zeros((r,c))
    for i in range(r):
        for j in range(c):
            if red[i,j] == threshold:
                ndvi[i,j] = threshold  
            else:
                ndvi[i,j] = (nir[i,j] - red[i,j]) / (nir[i,j] + red[i,j] + 1e-6)  
                if ndvi[i,j]>1:
                    ndvi[i,j] = 1
                elif ndvi[i,j] < -1:
                    ndvi[i,j] = -1
    return ndvi

def get_wet(blue, green, red, nir, swir1, swir2, year, threshold):
    r, c = blue.shape
    wet = np.zeros((r,c))
    if year>=2013:    #OLI data from 2013 onwards and TM data before that, using two separate calculation methods
        print('use oli data!')
        for i in range(r):
            for j in range(c):
                if blue[i,j] == threshold:
                    wet[i,j] = threshold
                else:
                    wet[i,j] = (0.1511*blue[i,j] + 0.1973*green[i,j] + 0.3283*red[i,j] + 
                                0.3407*nir[i,j] - 0.7117*swir1[i,j] - 0.4559*swir2[i,j]) / 10000
    else:
        print('use TM data!')
        for i in range(r):
            for j in range(c):
                if blue[i,j] == threshold:
                    wet[i,j] = threshold
                else:
                    wet[i,j] = (0.0315*blue[i,j] + 0.2021*green[i,j] + 0.3012*red[i,j] + 
                                0.1594*nir[i,j] - 0.6806*swir1[i,j] -0.6109*swir2[i,j]) / 10000
    return wet

def get_si(blue, red, nir, swir1, threshold):
    r, c = blue.shape
    si = np.zeros((r,c))
    for i in range(r):
        for j in range(c):
            if blue[i,j] == threshold:
                si[i,j] = threshold
            else:
                si[i,j] = ((swir1[i,j] + red[i,j]) - (nir[i,j] + blue[i,j])) / ((swir1[i,j] + red[i,j]) + (nir[i,j] + blue[i,j]) + 1e-6)
    return si

def get_ibi(green, red, nir, swir1, threshold):
    r, c = green.shape
    ibi = np.zeros((r,c))
    for i in range(r):
        for j in range(c):
            if green[i,j] == threshold:
                ibi[i,j] = threshold
            else:
                left_part = 2*swir1[i,j] / (swir1[i,j] + nir[i,j] + 1e-6)
                right_part = (nir[i,j] / (nir[i,j] + red[i,j] + 1e-6)) + (green[i,j] / (green[i,j] + swir1[i,j] + 1e-6))
                ibi[i,j] = (left_part - right_part) / (left_part + right_part + 1e-6)
    return ibi

def get_ndbsi(si, ibi, threshold):
    r, c = si.shape
    ndbsi = np.zeros((r,c))
    for i in range(r):
        for j in range(c):
            if si[i,j] == threshold:
                ndbsi[i,j] = threshold
            else:
                ndbsi[i,j] = (si[i,j] + ibi[i,j]) / 2.0
    return ndbsi

def get_epsilon(NDVI, threshold):
    r, c = NDVI.shape
    fv = np.zeros((r,c))
    for i in range(r):
        for j in range(c):
            if NDVI[i,j] == threshold:
                fv[i,j] = threshold
            if NDVI[i,j]>0.7:
                fv[i,j] = 1
            elif NDVI[i,j]<0.05:
                fv[i,j] = 0
            else:
                fv[i,j] = (NDVI[i,j] - 0.05) / (0.7 - 0.05)

    epsilon_surface = 0.9625 + 0.0614*fv - 0.0461*fv*fv
    epsilon_building = 0.9589 + 0.086*fv - 0.0671*fv*fv
    
    epsilon = np.zeros((r,c))
    data = []
    for i in range(r):
        for j in range(c):
            if NDVI[i,j] == threshold:
                epsilon[i,j] = threshold
            if ((NDVI[i,j]>=0.7) & (NDVI[i,j] < threshold)):
                epsilon[i,j] = epsilon_surface[i,j]
                data.append(epsilon[i,j])
            elif NDVI[i,j]<=0:
                epsilon[i,j] = 0.995
                data.append(epsilon[i,j])
            elif ((NDVI[i,j]>0) & (NDVI[i,j]<0.7)):
                epsilon[i,j] = epsilon_building[i,j]
                data.append(epsilon[i,j])

    return epsilon

def get_temperature(img, epsilon, year, threshold):
    r, c = img.shape
    lst = np.zeros((r,c))
    
    if year<2013:  
        gain = 0.055   #Image headers
        bias = 1.18243  
        K1 = 607.76   
        K2 = 1260.56  
        L_up = 2.19  #NASA search  https://atmcorr.gsfc.nasa.gov/
        L_down = 3.50
        Tao = 0.72

    else:
        gain = 0.0003342 
        bias = 0.1
        K1 = 774.88530
        K2 = 1321.0789
        L_up =2.14
        L_down = 3.49
        Tao = 0.75

    print("Tao: ", Tao)   #comfirm year
    for i in range(r):
        for j in range(c):
            if img[i,j] == threshold:
                lst[i,j] = threshold
            else:
# =============================================================================
#                 temp_bias_img = gain * img[i,j] + bias
#                 temp_T = (temp_bias_img - L_down - Tao*(1-epsilon[i,j])*L_up) / (Tao*(epsilon[i,j]))
#                 lst[i,j] = K2 / (1 + (K1 / temp_T))
# =============================================================================
                temp_T = (img[i,j] - L_down - Tao*(1-epsilon[i,j])*L_up) / (Tao*(epsilon[i,j]))  #两种计算方式都可以
                lst[i,j] = K2 / math.log(1 + (K1 / temp_T)) - 273

    return lst

def adjustData(data, threshold):  #index normalization, choose 95% confidence interval
    r, c = data.shape
    temp_data = []
    for i in range(r):
        for j in range(c):
            if data[i,j] == threshold:
                continue
            else:
                temp_data.append(data[i,j])
    data_max = np.max(np.array(temp_data))
    data_min = np.min(np.array(temp_data))
    
    #choose 95% confidence interval
    data1 = sorted(temp_data) 
    l = data1[1000]
    u = data1[len(data1)-1000]
    print("Max: {}     Max: {}".format(l,u))
    
    for i in range(r):
        for j in range(c):
            if data[i,j] == threshold:
                continue
            else:
                temp = l*(data[i,j]<l) + data[i,j] * ((data[i,j] >= l) & (data[i,j] <=u)) + u * (data[i,j] > u)
                data[i,j] = (temp - data_min) / (data_max - data_min)
    
    return data

def adjustRSEI(data, threshold): #RSEI Normalization
    r, c = data.shape
    temp_data = []
    for i in range(r):
        for j in range(c):
            if data[i,j] == threshold:
                continue
            else:
                temp_data.append(data[i,j])
    data_max = np.max(np.array(temp_data))
    data_min = np.min(np.array(temp_data))
    
    for i in range(r):
        for j in range(c):
            if data[i,j] == threshold:
                continue
            else:
                temp = -1*(data[i,j]<-1) + data[i,j] * ((data[i,j] >= -1) & (data[i,j] <=1)) + 1 * (data[i,j] > 1)
                data[i,j] = (temp - data_min) / (data_max - data_min)
    
    return data

def process_data(data, threshold):  
    r, c = data.shape
    data_pro = []
    for i in range(r):
        for j in range(c):
            if data[i,j] == threshold:
                continue
            else:
                data_pro.append(data[i,j])
    data_pro = np.array(data_pro)
    return data_pro.reshape((1, -1))

def get_rsei(rsei, data, threshold):
    RSEI = np.zeros((data.shape[0], data.shape[1])) + threshold
    valid_region = np.loadtxt(join(base_dir, 'valid_region.txt'))
    for i in range(valid_region.shape[0]):
        idx = int(valid_region[i,0])
        idy = int(valid_region[i,1])
        RSEI[idx, idy] = rsei[i,0]
    return RSEI

def writeRSData(filename, im_data, im_geotrans, im_proj): #write tif
    path, name = os.path.split(filename)
    if not os.path.exists(path):
        os.makedirs(path)
        
    if 'int8' in im_data.dtype.name:  
            datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
            
    if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1,im_data.shape  
        
    driver = gdal.GetDriverByName("GTiff")        
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
    dataset.SetGeoTransform(im_geotrans)        
    dataset.SetProjection(im_proj)               
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del dataset

"""build mask"""
year = 2010
TiaoDaiHao = 12840  #Landsat image path and row

imgPath = join(base_dir, '%d'%year)
if not os.path.exists(imgPath):
    os.makedirs(imgPath)
    
dataset = gdal.Open(join(imgPath, 'TIRS_%d.tif'%TiaoDaiHao), gdalconst.GA_ReadOnly)
img_width = dataset.RasterXSize
img_height = dataset.RasterYSize
img_data = np.array(dataset.ReadAsArray(0, 0, img_width, img_height), dtype=float)

print("background value: ", img_data[0,0])
data = []
valid = []
for i in range(img_data.shape[0]):
    for j in range(img_data.shape[1]):
        if img_data[i,j]==img_data[0,0]:
            data.append([i,j])
        else:
            valid.append([i,j])
data = np.array(data)
valid = np.array(valid)
print(data.shape)
with open(join(base_dir, 'mask.txt'), 'w') as f:
    for k in range(data.shape[0]):
        f.write('%d %d\n' %(data[k,0], data[k,1]))   

with open(join(base_dir, 'valid_region.txt'), 'w') as f1:
    for l in range(valid.shape[0]):
        f1.write('%d %d\n' %(valid[l,0], valid[l,1])) 
        
print('begin calculate RSEI of year %d\n'%year)
dataset = gdal.Open(join(imgPath, 'multispec_%d.tif'%TiaoDaiHao), gdalconst.GA_ReadOnly)

img_width = dataset.RasterXSize
img_height = dataset.RasterYSize
print('multispec load...')
print('width: {0}, highth: {1}\n'.format(img_width, img_height))
im_proj = dataset.GetProjection()
im_geotrans = dataset.GetGeoTransform()  # Affine Matrix
img_data = np.array(dataset.ReadAsArray(0, 0, img_width, img_height), dtype=float) 

if year>=2013:  
    blue = img_data[1,...]
    green = img_data[2,...]
    red = img_data[3,...]
    nir = img_data[4,...]
    swir1 = img_data[5,...]
    swir2 = img_data[6,...]
else:
    blue = img_data[0,...]
    green = img_data[1,...]
    red = img_data[2,...]
    nir = img_data[3,...]
    swir1 = img_data[4,...]
    swir2 = img_data[5,...]


if blue[0,0]!=32767.0:
    invalid = np.loadtxt(join(base_dir, 'mask.txt'))
    r, c = invalid.shape
    for i in range(r):
        a, b = int(invalid[i,0]), int(invalid[i,1])
        blue[a, b] = 32767.0
        green[a, b] = 32767.0
        red[a, b] = 32767.0
        nir[a, b] = 32767.0
        swir1[a, b] = 32767.0
        swir2[a, b] = 32767.0
else:
    thresh = blue[0,0]

NDVI = get_ndvi(red, nir, thresh)
NDVI[NDVI<0] = 0


WET = get_wet(blue, green, red, nir, swir1, swir2, int(year), thresh)
SI = get_si(blue, red, nir, swir1, thresh)
IBI = get_ibi(green, red, nir, swir1, thresh)
NDBSI = get_ndbsi(SI, IBI, thresh)

epsilon = get_epsilon(NDVI, thresh)

dataset_tirs = gdal.Open(join(imgPath, 'TIRS_%d.tif'%TiaoDaiHao), gdalconst.GA_ReadOnly)
img_width_tirs = dataset_tirs.RasterXSize
img_height_tirs = dataset_tirs.RasterYSize
print('TIRS load...')
print('TIRSimgewidth: {0}, highth: {1}\n'.format(img_width_tirs, img_height_tirs))
im_proj_tirs = dataset_tirs.GetProjection()
im_geotrans_tirs = dataset_tirs.GetGeoTransform() 
TIRS_data = np.array(dataset_tirs.ReadAsArray(0, 0, img_width_tirs, img_height_tirs), dtype=float)  
a = TIRS_data[0,0]
TIRS_data[TIRS_data==a] = thresh  

LST = get_temperature(TIRS_data, epsilon, int(year), thresh)



LST = adjustData(LST, thresh)

NDVI = adjustData(NDVI, thresh)

NDBSI = adjustData(NDBSI, thresh)

WET = adjustData(WET, thresh)


writeRSData(join(base_dir, 'result\\%d\\LST.tif'%year), LST, im_geotrans_tirs, im_proj_tirs)
writeRSData(join(base_dir, 'result\\%d\\NDVI.tif'%year), NDVI, im_geotrans, im_proj)
writeRSData(join(base_dir, 'result\\%d\\NDBSI.tif'%year), NDBSI, im_geotrans, im_proj)
writeRSData(join(base_dir, 'result\\%d\\WET.tif'%year), WET, im_geotrans, im_proj)

ndvi = process_data(NDVI, thresh)
wet = process_data(WET, thresh)
ndbsi = process_data(NDBSI, thresh)
lst = process_data(LST, thresh)

data = np.concatenate((ndvi, wet, ndbsi, lst), axis=0)
matrix = np.cov(data) #Calculate Covariance matrix

print(matrix)
a, b=np.linalg.eig(matrix) 

print(a)

print(b)

for num in range(4):
    print(a[num] / np.sum(a))

pca = PCA(n_components=1)  #choose PCA1
data = data.T
rsei = pca.fit_transform(data)  #choose PCA1 to recalculate RSEI
RSEI = get_rsei(rsei, NDVI, thresh)
RSEI = adjustRSEI(RSEI, thresh)
writeRSData(join(base_dir, 'result\\%d\\RSEI.tif'%year), RSEI, im_geotrans_tirs, im_proj_tirs)
