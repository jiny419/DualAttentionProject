from keras.losses import binary_crossentropy
import keras.backend as K
from keras.models import Model
from keras.layers import *
from keras import optimizers
from keras.applications import DenseNet121
from sklearn import metrics


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss



def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + (1 - dice_loss(y_true, y_pred))

def get_unet_128(input_shape=(448, 448, 3),
                 num_classes=1):
    input_size = input_shape[0]

    inputs = Input(shape=input_shape)


    dense_model = DenseNet121(input_tensor=inputs, include_top=False, weights=None, pooling='max') 
    dense_model.layers.pop()  #remove maxpolling layer

    # (28,28) 
    up4 = UpSampling2D((2, 2))(dense_model.layers[-1].output)
    up4 = concatenate([dense_model.get_layer('conv4_block24_concat').output, up4], axis=3)
    up4 = Conv2D(256, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)


    # (56,56) 
    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([dense_model.get_layer('conv3_block12_concat').output, up3], axis=3)
    up3 = Conv2D(128, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)


    # (112,112) 
    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([dense_model.get_layer('conv2_block6_concat').output, up2], axis=3)
    up2 = Conv2D(64, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)


    # (224,224) 
    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([dense_model.get_layer('conv1/conv').output, up1], axis=3)
    up1 = Conv2D(32, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)


    # (448,448)
    up0 = UpSampling2D((2, 2))(up1)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)


    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0)

    #u-net compile
    u_net_model = Model(inputs=inputs, outputs=classify)
    u_net_model.compile(optimizer=optimizers.adam(lr=0.001), loss=bce_dice_loss, metrics=[dice_loss])   
    print('u-net model : densenet encoder ')


    return input_size, u_net_model

def get_densenet_from_unet( unet_model):
  dense_input = unet_model.input
  dense_center_output = unet_model.get_layer('relu').output

  
  x = GlobalMaxPool2D()(dense_center_output)
  x = Dense(1024, name='fully', init='uniform')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Dense(512, init='uniform')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  densenet_output = Dense(7, activation='softmax', name='softmax')(x)
  
  densenet_model = Model(inputs=dense_input, outputs=densenet_output)
  densenet_model.compile(loss=weighted_categorical_crossentropy([4.375, 2.783, 1.301, 12.440, 1.285, 0.213, 10.075]),
                  optimizer=optimizers.adam(lr=0.003),
                  metrics=['acc'])

  return  densenet_model

from keras.applications import ResNet50, VGG16, DenseNet121
from keras.layers import *
from keras.models import Model
from keras.callbacks import EarlyStopping, LambdaCallback, CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
import numpy as np

_, u_net_model = get_unet_128()
model = get_densenet_from_unet(u_net_model)

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  

batch_size = 8
train_generator = train_datagen.flow_from_directory(
        './train',
        target_size=(448, 448),
        batch_size=batch_size,  
        class_mode='categorical',
        subset='training')



val_generator = train_datagen.flow_from_directory(
        './train',
        target_size=(448, 448),
        batch_size=batch_size, 
        shuffle = False,
        class_mode='categorical',
        subset='validation')

csv_logger = CSVLogger('training.log')

classification_checkpoint1 = ModelCheckpoint(monitor='val_loss',
                                    filepath='weights/classification_weights' + '.{epoch:02d}-{val_loss:.2f}.hdf5',
                                    verbose = 1,
                                    save_best_only=True,
                                    save_weights_only=True,
                                    mode='min')

classification_checkpoint2 = ModelCheckpoint(monitor='val_acc',
                                    filepath='weights/classification_weights' + '.{epoch:02d}-{val_acc:.2f}.hdf5',
                                    verbose = 1,
                                    save_best_only=True,
                                    save_weights_only=True,
                                    mode='max')

'''
history = model.fit_generator(train_generator,
                              steps_per_epoch=np.ceil(float(train_generator.samples) / float(batch_size)),
                              epochs=200,           
                              validation_data=val_generator,
                              validation_steps=np.ceil(float(val_generator.samples) / float(batch_size)),
                              callbacks=[csv_logger, classification_checkpoint1, classification_checkpoint2])
'''

model.load_weights('weights/classification_weights.138-0.76.hdf5')

#모델 평가1
print("-- Evaluate --")
scores = model.evaluate_generator(val_generator, steps=np.ceil(float(val_generator.samples) / float(batch_size)))
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

val_generator.reset()
predictions = model.predict_generator(val_generator, steps = np.ceil(float(val_generator.samples)/float(batch_size)))

val_preds = np.argmax(predictions, axis=-1)
val_trues = val_generator.classes
cm = metrics.confusion_matrix(val_trues, val_preds)

print('confusion matrix (predict 2000 images) : ')
print(cm)

print('accuracy' + str( metrics.accuracy_score(val_trues, val_preds))) 


precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(val_trues, val_preds) 

print('precisions : ' + str(precisions))
print('average precisions : ' + str(precisions.mean()))
print('recall : ' + str(recall) )
print('average recall : ' +str(recall.mean()))
print('f1_score : '+ str(f1_score))
print('average f1_score : ' + str(f1_score.mean()))



'''
#모델 평가1
print("-- Evaluate --")
scores = model1.evaluate_generator(val_generator, steps=np.ceil(float(val_generator.samples) / float(batch_size)))
print("%s: %.2f%%" %(model1.metrics_names[1], scores[1]*100))

val_generator.reset()
predictions = model1.predict_generator(val_generator, steps = np.ceil(float(val_generator.samples)/float(batch_size)))

val_preds = np.argmax(predictions, axis=-1)
val_trues = val_generator.classes
cm = metrics.confusion_matrix(val_trues, val_preds)

print('confusion matrix (predict 2000 images) : ')
print(cm)

print('accuracy' + str( metrics.accuracy_score(val_trues, val_preds))) 


precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(val_trues, val_preds) 

print('precisions : ' + str(precisions))
print('average precisions : ' + str(precisions.mean()))
print('recall : ' + str(recall) )
print('average recall : ' +str(recall.mean()))
print('f1_score : '+ str(f1_score))
print('average f1_score : ' + str(f1_score.mean()))



#모델 평가2
val_generator.reset()
print("-- Evaluate --")
scores = model2.evaluate_generator(val_generator, steps=np.ceil(float(val_generator.samples) / float(batch_size)))
print("%s: %.2f%%" %(model2.metrics_names[1], scores[1]*100))

val_generator.reset()
predictions = model2.predict_generator(val_generator, steps = np.ceil(float(val_generator.samples)/float(batch_size)))

val_preds = np.argmax(predictions, axis=-1)
val_trues = val_generator.classes
cm = metrics.confusion_matrix(val_trues, val_preds)

print('confusion matrix (predict 2000 images) : ')
print(cm)

print('accuracy' + str( metrics.accuracy_score(val_trues, val_preds))) 


precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(val_trues, val_preds) 

print('precisions : ' + str(precisions))
print('average precisions : ' + str(precisions.mean()))
print('recall : ' + str(recall) )
print('average recall : ' +str(recall.mean()))
print('f1_score : '+ str(f1_score))
print('average f1_score : ' + str(f1_score.mean()))
'''


