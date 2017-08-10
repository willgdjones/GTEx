#numpy modules
import sys
sys.path = ['/nfs/gns/homes/willj/anaconda3/envs/GTEx/lib/python3.5/site-packages'] + sys.path
import numpy as np
import pickle
#Keras modules
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D, Input, Dense
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import load_model
#Sklearn modules
from sklearn.model_selection import train_test_split
#Scipy modules
from scipy.misc.pilutil import imresize
import argparse
import os
import pdb


def main():
    tissues_types = ['Lung',
     'Artery - Tibial',
     'Heart - Left Ventricle',
     'Breast - Mammary Tissue',
     'Brain - Cerebellum',
     'Pancreas',
     'Testis',
     'Liver',
     'Ovary',
     'Stomach']


    raw_X = []
    tissue_labels = []

    tile_number = args['tile_number']
    tile_level_index = args['tile_level_index']
    for t in tissues_types:
        print ('Loading {}:'.format(t))
        [rX, tl, rID] = pickle.load(open('data/processed/patches/data_{}_{}_{}_check.py'.format(t,tile_number,tile_level_index), 'rb'))
        raw_X.extend(rX)
        tissue_labels += tl

    print ('normalizing data')
    print (len(raw_X))
    X = (np.array(raw_X,dtype=np.float16) / 255)

    if args['grey_scale'] == '1':
        print ('making greyscale')
        new_X = np.zeros_like(X)
        mean_X = X.mean(3)
        new_X[:,:,:,0] = mean_X
        new_X[:,:,:,1] = mean_X 
        new_X[:,:,:,2] = mean_X 

        X = new_X
    


    classes, labels = np.unique(tissue_labels, return_inverse=True)
    pdb.set_trace()

    # X = np.array([imresize(x,(299,299)) for x in raw_X])
    y = np_utils.to_categorical(labels)

    print ('split train test set')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    print ('data loaded')

    print ('checking if model exists: models/inception_{}_{}_gs{}_check.h5'.format(tile_number,tile_level_index,args['grey_scale']))
    print (os.path.isfile('models/inception_{}_{}_gs{}_check.h5'.format(tile_number,tile_level_index,args['grey_scale'])))
    if os.path.isfile('models/inception_{}_{}_gs{}_check.h5'.format(tile_number,tile_level_index,args['grey_scale'])):
        model = load_model('models/inception_{}_{}_gs{}_check.h5'.format(tile_number,tile_level_index,args['grey_scale']))
        print ('loading pretrained model')
        print ('Evaluating')
        loss, acc = model.evaluate(X_test,y_test) 
        print ("loss: {}, acc: {}".format(loss,acc))
    else:
        print ('training model')
        inception_model = InceptionV3(weights='imagenet', include_top=False)

        x = inception_model.output

        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(10, activation='softmax')(x)

        model = Model(input=inception_model.input, output=predictions)

        for layer in inception_model.layers:
            layer.trainable = False
            

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'], verbose=2)


        datagen = ImageDataGenerator()

        datagen.fit(X_train)

        history1 = model.fit_generator(datagen.flow(X_train, y_train, batch_size=32), samples_per_epoch=len(X_train), nb_epoch=10, validation_data=(X_test, y_test))

        for layer in model.layers[:172]:
            layer.trainable = False
        for layer in model.layers[172:]:
            layer.trainable = True

        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'], verbose=2)
        model.evaluate(X_test,y_test)

        history2 = model.fit_generator(datagen.flow(X_train, y_train, batch_size=32), samples_per_epoch=len(X_train), nb_epoch=30, validation_data=(X_test, y_test))
        pickle.dump([history1.history,history2.history], open('models/histories_{}_{}_gs{}_check.py'.format(tile_number,tile_level_index,args['grey_scale']),'wb'))
        model.save('models/inception_{}_{}_gs{}_check.h5'.format(tile_number,tile_level_index, args['grey_scale']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-n','--tile_number', help='Description for foo argument', required=True)
    parser.add_argument('-l','--tile_level_index', help='Description for foo argument', required=True)
    parser.add_argument('-g','--grey_scale', help='Description for foo argument', required=True)
    args = vars(parser.parse_args())
    main()
    
