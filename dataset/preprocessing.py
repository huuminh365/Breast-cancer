import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
    CLAHE image
'''

plt.rcParams.update({'font.size': 15})
plt.rcParams['figure.figsize'] = (30, 10)
def show_hist(gray, equ, categori, cdf1 = [], cdf2 = []):
    m = int(np.max(img))
    f = plt.figure(figsize=(15, 15))
    f.add_subplot(2, 2, 1, xticks = [], yticks=[]).set_title('Gray Image')
    plt.imshow(gray, cmap = 'gray')
    f.add_subplot(2, 2, 2).set_title('Histogram Gray Image')
    histogram, bin_edges = np.histogram(gray, bins=m+1, range=(0, m+1))
    plt.plot(bin_edges[:-1], histogram/np.max(histogram))
    if len(cdf1) > 0:
        plt.plot(cdf1, color = 'red')
    plt.xlabel('pixel')
    plt.ylabel('percent')


    f.add_subplot(2, 2, 3, xticks = [], yticks=[]).set_title(f'{categori}')
    plt.imshow(equ, cmap = 'gray')
    f.add_subplot(2, 2, 4).set_title(f'Histogram {categori}')
    histogram, bin_edges = np.histogram(equ, bins=256, range=(0, 256))
    plt.plot(bin_edges[:-1], histogram/np.max(histogram))
    if len(cdf2) > 0:
        plt.plot(cdf2, color = 'red')
    plt.xlabel('pixel')
    plt.ylabel('percent')
    plt.show()
    
def get_cdf(img):
    m = int(np.max(img))
    hist = np.histogram(img, bins=m+1, range=(0, m+1))[0]
    hist = hist/img.size
    cdf = np.cumsum(hist)
    return cdf

# numpy
def histogram_equalization(img):
    m = int(np.max(img))
    hist = np.histogram(img, bins=m+1, range=(0, m+1))[0]
    # bước 1: tính pdf
    hist = hist/img.size
    # bước 2: tính cdf
    cdf = np.cumsum(hist)
    # bước 3: lập bảng thay thế
    s_k = (255 * cdf)
    # ảnh mới
    img_new = np.array([s_k[i] for i in img.ravel()]).reshape(img.shape)
    return img_new


#test
gray2 = img
img_equ= histogram_equalization(gray2)
cdf1 = get_cdf(gray2)
cdf2 = get_cdf(img_equ)
show_hist(gray2, img_equ, 'Equalized with numpy', cdf1=cdf1, cdf2=cdf2)

# ----------------------------------------------------------------------------------

'''
    Crop image:
     - find threshold using sauvola
     - CLAHE image
     - erode image
'''

