import os
import sys
import cv2
import shutil
import pydicom 
import numpy as np
import pandas as pd
import pydicom as pdc
import SimpleITK as sitk
from requests import patch
from tensorflow import keras
import matplotlib.pyplot as plt
import segmentation_models as sm
from sklearn.utils import shuffle
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def slice_with_overlay(img_name, stride_x, stride_y, out, path_to_save='./', path_to_read='./'):
    """
        stride_x kadar enine stride_y kadar boyuna bindirme yapar. 
    """

    os.chdir(path_to_read)
    print(os.getcwd())
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>{img_name} görüntüsü parçalanıyor<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

    k = img_name
    is_mask = True
    if img_name.endswith('.dcm'):
        is_mask = False
    else:
        is_mask = True

    if is_mask:
        img = cv2.imread(k)  # etiket dosyaları için
    else:
        img = pdc.dcmread(k).pixel_array
        # img=pdc.read_file(dicoms[0],force=True)
        # img.file_meta.TransferSyntaxUID = pdc.uid.ImplicitVRLittleEndian  # or whatever is the correct transfer syntax for the file
        # img.SamplesPerPixel = 1
        # img = img.pixel_array

        # img = pd.dcmread(k).pixel_array #dicom dosyaları için
  
    os.chdir(path_to_save)
    if os.path.exists(k[:-4]):
        os.chdir(k[:-4])
    else:
        os.mkdir(k[:-4])
        os.chdir(k[:-4])


    row = img.shape[0]
    col = img.shape[1]

    outer = (row // out) * out
    inner = (col // out) * out

    for r in range(0, outer, stride_y):
        for c in range(0, inner + 1, stride_x):
            #satır boyutundan büyük çıkarsa satır boyutunu sınır alarak geriye out kadar geriden alır
            if r + out > row:

                #sütun boyutundan büyük çıkarsa sütun boyutunu sınır alarak geriye out kadar geriden alır
                if c + out > col:

                    # print(row - out, row, col - out, col)

                    if is_mask:
                        # #etiketleri kayıtederken
                        cv2.imwrite(str(row-out) + '_' + str(row) + '_' + str(col-out) +'_' + str(col) + '_' + k, img[row-out:row, col - out:col, :])

                    else:
                        # dicom dosyalarını kayıt ederken
                        temp_dcm = img[row-out:row, col - out:col]
                        temp_dcm = sitk.GetImageFromArray(temp_dcm)
                        sitk.WriteImage(temp_dcm, str(row-out) + '_' + str(row) + '_' + str(col-out) + '_' + str(col) + '_' + k)

                    break
                    
                else:
                    #print(row - out, row, c, c + out)

                    if is_mask:
                        # #etiketleri kayıtederken
                        cv2.imwrite(str(row-out) + '_' + str(row) + '_' + str(c) + '_' + str(c + out) + '_' + k, img[row - out : row, c : c + out, :])

                    else:
                        # dicom dosyalarını kayıt ederken
                        temp_dcm = img[row - out:row, c : c + out]
                        temp_dcm = sitk.GetImageFromArray(temp_dcm)
                        sitk.WriteImage(temp_dcm, str(row-out) + '_' + str(row) + '_' + str(col-out) + '_' + str(col) + '_' + k)

            elif c + out > col:

                #print(r, r + out, col - out, col)
                if is_mask:
                        # #etiketleri kayıtederken
                        cv2.imwrite(str(r) + '_' + str(r + out) + '_' + str(col - out) + '_' + str(col) + '_' + k, img[r : r + out, col - out : col, :])

                else:
                    # dicom dosyalarını kayıt ederken
                    temp_dcm = img[r : r + out, col - out : col]
                    temp_dcm = sitk.GetImageFromArray(temp_dcm)
                    sitk.WriteImage(temp_dcm, str(r) + '_' + str(r + out) + '_' + str(col-out) + '_' + str(col) + '_' + k)

                break
            else:
                #print(r , r + out, c, c + out)

                if is_mask:
                        # #etiketleri kayıtederken
                        cv2.imwrite(str(r) + '_' + str(r + out) + '_' + str(c) + '_' + str(c + out) + '_' + k, img[r : r + out, c : c + out, :])

                else:
                    # dicom dosyalarını kayıt ederken
                    temp_dcm = img[r : r + out, c : c + out]
                    temp_dcm = sitk.GetImageFromArray(temp_dcm)
                    sitk.WriteImage(temp_dcm, str(r) + '_' + str(r + out) + '_' + str(c) + '_' + str(c + out) + '_' + k)

               
    os.chdir(path_to_read)


def preprocess_data(img, mask, num_class, scaler, preprocess_input):
    #Scale images
    img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
    img = preprocess_input(img)  #Preprocess based on the pretrained backbone...
    #Convert mask to one-hot
    if mask.max() > 2:
        mask = mask / 255
    mask = to_categorical(mask, num_class)
      
    return (img,mask)





def trainGenerator(train_img_path, train_mask_path, num_class, target_size, batch_size, seed, scaler, preprocess_input):
    
    img_data_gen_args = dict(fill_mode='reflect')
    
    image_datagen = ImageDataGenerator(**img_data_gen_args)
    mask_datagen = ImageDataGenerator(**img_data_gen_args)
    print('Target Shape: ', target_size)
    
    image_generator = image_datagen.flow_from_directory(
        train_img_path,
        class_mode = None,
        batch_size = batch_size,
        target_size=target_size,
        seed = seed,
        shuffle=False)
    
    mask_generator = mask_datagen.flow_from_directory(
        train_mask_path,
        class_mode = None,
        color_mode = 'grayscale',
        batch_size = batch_size,
        target_size=target_size,
        seed = seed,
        shuffle=False)
    
    train_generator = zip(image_generator, image_generator.filenames, mask_generator, mask_generator.filenames)
    
    for (img, img_file, mask, mask_file) in train_generator:

        img, mask = preprocess_data(img, mask, num_class, scaler, preprocess_input)
        yield (img, img_file, mask, mask_file)



def concat(path, is_dcm:bool, model_name):
    '''
    img_list : birleştirilecek resimlerin listesi, klasörün içinde bulunan resimler,
    img_size : birleştirilecek görüntünün orijinal boyutu,
    configs : resimlerin boyutlarını içeren sözlük
    '''
    row = 0
    column = 0
    if is_dcm:
        img_list = [i for i in os.listdir(path) if i.endswith('.dcm')]
    else:
        img_list = [i for i in os.listdir(path) if i.endswith('.png')]

    os.chdir(path)

    for i in img_list:

        i_ = i.split('_')

        if row < int(i_[1]):
            row = int(i_[1])
        
        if column < int(i_[3]):
            column = int(i_[3])

    

    if img_list[0].endswith('.dcm'):
        concat_img = np.zeros((row, column), dtype='uint16')
    elif img_list[0].endswith('.png'):
        concat_img = np.zeros((row, column,3), dtype='uint8')

    for i in img_list:
        if i.endswith('.png'):
            img = cv2.imread(i)
        
            img = cv2.resize(img, (1024, 1024), cv2.INTER_AREA)
            name = i.split('_')
            concat_img[int(name[0]):int(name[1]), int(name[2]):int(name[3]), :] = img
        elif i.endswith('.dcm'):
            img = pdc.dcmread(i).pixel_array
            img = cv2.resize(img, (1024, 1024), cv2.INTER_AREA)
            name = i.split('_')
            concat_img[int(name[0]):int(name[1]), int(name[2]):int(name[3])] = img

            
        name = img_list[0].split('_')
        name = '_'.join(name[4:])
        s_path = '\\'.join(path.split('\\')[:-2])
        print(s_path)
        print(concat_img.shape)
        if img_list[0].endswith('.dcm'):
            temp_dcm = sitk.GetImageFromArray(concat_img)
            sitk.WriteImage(temp_dcm, name)
        else:
            if not os.path.exists(s_path + '/' + model_name + '_full_size_pred'):
                    os.mkdir(s_path + '/' + model_name + '_full_size_pred')
            cv2.imwrite(s_path + '/' + model_name +  '_full_size_pred/' + name, concat_img)

def visualize(title = '',**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(12, 9))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
#         plt.xticks([])
#         plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
        plt.savefig(title)



def predict(model_path, backbone, image_path, mask_path, save_path, patch_size_w, patch_size_h, will_sliced):
    
    
    
    
    
    BACKBONE = backbone #'senet154'
    model = load_model(model_path, compile=False)
    model_name = model_path.split('\\')[-1][22:-5]
    print(model_name)
    print('===================>>>>>>>>>>>>>>Model input size: ',model.input_shape)

    if not os.path.exists(model_name + '_preds'):
        print(f'{model_name} için klasör oluşturuluyor...')
        os.mkdir(model_name + '_preds')

    if will_sliced:
        
    
        print('===================>>>>>>>>>>>>>>Klasörler Oluşturuluyor...')
        try:
            os.mkdir('data')
            os.mkdir('data/patched_images')
            os.mkdir('data/patched_labels')
        except:
            pass
        print('===================>>>>>>>>>>>>>>Klasörler Oluşturuldu!!!')
        patched_img_path = os.path.abspath('data/patched_images')
        patched_msk_path  = os.path.abspath('data/patched_labels')
        save_path = os.path.abspath(model_name + '_preds/')
        seed=24
        batch_size= 1
        n_classes=2
        test_img_path = os.path.abspath('data/patched_images')
        test_msk_path = os.path.abspath('data/patched_labels')
        scaler = MinMaxScaler()

        
    




        img_read_path = '/'.join(image_path.split('\\'))
        for root, _, files in os.walk(img_read_path):
            for file in files:
                if file.endswith('.png') or file.endswith('jpeg') or file.endswith('jpg'):
                    os.chdir(root)
                    slice_with_overlay(file, patch_size_h, patch_size_w, patch_size_h, patch_size_w, patched_img_path, img_read_path )


        msk_read_path = '/'.join(mask_path.split('\\'))

        for root, _, files in os.walk(msk_read_path):
            for file in files:
                if file.endswith('.png') or file.endswith('jpeg') or file.endswith('jpg'):
                    os.chdir(root)
                    slice_with_overlay(file, patch_size_h, patch_size_w, patch_size_h, patch_size_w, patched_msk_path, msk_read_path )
        
    else:
        print('DOSYALAR ZATEN BÖLÜNMÜŞ')
        print('===================>>>>>>>>>>>>>>Klasörler Oluşturuldu!!!')
        patched_img_path = os.path.abspath('data/patched_images')
        patched_msk_path  = os.path.abspath('data/patched_labels')
        save_path = os.path.abspath(model_name + '_preds/')
        seed=24
        batch_size= 1
        n_classes=2
        test_img_path = os.path.abspath('data/patched_images')
        test_msk_path = os.path.abspath('data/patched_labels')
        scaler = MinMaxScaler()

    
    print('\n\n===================>>>>>>>>>>>>>>TAHMİN BAŞLIYOR...\n\n')
    preprocess_input = sm.get_preprocessing(BACKBONE)
    img_count = len([file for root, _, files in os.walk(patched_img_path) for file in files])
    print('===================>>>>>>>>>>>>>>Tahmin edilecek resim sayısı:',img_count)
    test_img_gen = trainGenerator(test_img_path, test_msk_path, num_class=n_classes, target_size = (model.input_shape[1], model.input_shape[1]), batch_size = batch_size, seed = seed, scaler=scaler, preprocess_input=preprocess_input)
    for i in range(0,img_count):
        test_image_batch, img_name, test_mask_batch, msk_name = test_img_gen.__next__()
        print('===================>>>>>>>>>>>>>>Test Image Size: ', test_image_batch.shape)
        test_mask_batch_argmax = np.argmax(test_mask_batch, axis=3)
        test_pred_batch = model.predict(test_image_batch)
        test_pred_batch_argmax = np.argmax(test_pred_batch, axis=3)
    

        a = test_pred_batch_argmax[0].copy()
        a = a.astype(np.float32)
        a = cv2.resize(a, (patch_size_w, patch_size_h), interpolation=cv2.INTER_AREA)
        a.resize(patch_size_w,patch_size_h,1)
        a = a * 255
        print('RESMİN MAKSİMUM DEĞERİ: ',a.max())
        print('RESMİN PİKSEL TOPLAMLARININ DEĞERİ: ',np.sum(a))
        img = np.concatenate((a,a,a), axis=2)

        img_name = img_name.split('\\')
        if os.path.exists(save_path + '/' + img_name[0]):
            cv2.imwrite(save_path + '/' + img_name[0] + '\\' + img_name[1], img)
        else:
            os.mkdir(save_path + '/' + img_name[0])
            cv2.imwrite(save_path + '/' + img_name[0] + '\\' + img_name[1], img)
    

    paths = []
    for root, _, files in os.walk(save_path):    
        for file in files:
            paths.append(os.path.abspath(root))
    paths = list(set(paths))
    for i in paths:
        print(i)
        concat(i, False, model_name)
    os.chdir(image_path)
    os.chdir('../')
    preds = [os.path.abspath(root + '\\' + file) for root, _, files in os.walk(model_name + '_full_size_pred') for file in files]
    labels = [root + '\\' + file for root, _, files in os.walk(mask_path) for file in files]
    imagess = [root + '\\' + file for root, _, files in os.walk(image_path) for file in files]
    print('TAHMİNLERİN SAYISI: ', len(preds))
    print('LABELLERİN SAYISI: ', len(labels))
    print('İMAGELERİN SAYISI: ', len(imagess))
    save_path = os.getcwd() + '\\' + model_name + '_preds'
    os.chdir(save_path)
    for i, m, p in zip(imagess, labels, preds):
        print('KAYIT YERİ: ', os.getcwd())  
        
        img = cv2.imread(i)
        lbl = cv2.imread(m)
        lbl = lbl * 255
        prd = cv2.imread(p)
   
        
        print('===========>>>>>>>>>>>>>>>>> GRAFİK İSMİ:', i.split('\\')[-1])
        visualize(
            title=i.split('\\')[-1],
            image = img,
            mask = lbl,
            prediction = prd
        )

def main():
    model_path = r'C:\Users\mustafa.cavusoglu2\Desktop\pred_main\20221024_165407842233_Linknet_resnet34_0.0007_150_sigmoid_model.hdf5'
    image_path = r'C:\Users\mustafa.cavusoglu2\Desktop\pred_main\mlp-test\\'
    mask_path  = r'C:\Users\mustafa.cavusoglu2\Desktop\pred_main\mlp-test-mask\\'
    save_path = r'C:\Users\mustafa.cavusoglu2\Desktop\pred_main\concat_pred'
    patch_size_w = 1024
    patch_size_h = 1024
    will_sliced = False
    
   
    backbone  = model_path.split('\\')[-1].split('_')[3]
    model_name = model_path.split('\\')[-1][22:-5]
    print(f'##############################\nModel İsmi: {model_name.upper()}\nBackbone İsmi: {backbone.upper()}\n##############################')
    predict(model_path, backbone, image_path, mask_path, save_path, patch_size_w, patch_size_h, will_sliced)
    


if __name__ == '__main__':
  main()
