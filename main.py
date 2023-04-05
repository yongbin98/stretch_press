from cgi import test
import os, psutil
import random
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, train_test_split
import sklearn.preprocessing
import tensorflow as tf

resample_num = 150

def loadMatlabData(filePath):
    fileName = filePath + 'stretch_press_data_'+str(resample_num)+'.mat'
    
    ###============= Load Matlab files
    contentsMat = sio.loadmat(fileName)
    x_data = contentsMat['x_data']
    y_data = contentsMat['y_data']
    
    return x_data, y_data

def loadMatlabData2(filePath,i):
    fileName = filePath + 'data'+str(i)+'.mat'

    ###============= Load Matlab files
    contentsMat = sio.loadmat(fileName)
    aug_xtrain = contentsMat['aug_xtrain']
    aug_ytrain = contentsMat['aug_ytrain']
    data_valid = contentsMat['data_valid']
    label_valid = contentsMat['label_valid']
    x_test1 = contentsMat['x_test1']
    y_test1 = contentsMat['y_test1']
    train_time = contentsMat['train_time']
    valid_time = contentsMat['valid_time']
    test_time = contentsMat['test_time']
    train_amplitude = contentsMat['train_amplitude']
    valid_amplitude = contentsMat['valid_amplitude']
    test_amplitude = contentsMat['test_amplitude']
    
    return aug_xtrain, aug_ytrain, data_valid, label_valid, x_test1, y_test1, train_time, valid_time, test_time, train_amplitude, valid_amplitude, test_amplitude
def dnn_model():
    input1 = tf.keras.layers.Input(shape=(resample_num,1,1), name='stretch_press')
    input2 = tf.keras.layers.Input(shape=(1), name='amplitude')
    input3 = tf.keras.layers.Input(shape=(1), name='time')
    
    x_concat1 = same_model(input1)
    x_concat2 = input2
    x_concat3 = input3
    
    x = tf.keras.layers.Concatenate()([x_concat1,x_concat2,x_concat3])
    x = tf.keras.layers.Dense(256, activation='relu', name='FC0')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(64, activation='relu', name='FC1')(x)
    output = tf.keras.layers.Dense(8, activation='softmax', name='Output')(x)
    
    model= tf.keras.models.Model(inputs=[input1, input2, input3], outputs=output)
    # model = tf.keras.applications.resnet.ResNet50(weights=None, input_tensor=tf.keras.layers.Input(shape=(90, 1, 1)), classes=8)
    model.summary()
        
    return model
def same_model(input):
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(7,1), strides = (1), activation = 'relu', padding='valid', name='CV1')(input)
    x = tf.keras.layers.AveragePooling2D(pool_size = (3,1), strides = (2,1), name='AP1')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(7,1), strides = (1), activation = 'relu', padding='valid', name='CV2')(x)
    x = tf.keras.layers.AveragePooling2D(pool_size = (3,1), strides = (2,1), name='AP2')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7,1), strides = (1), activation = 'relu', padding='valid', name='CV3')(x)
    x = tf.keras.layers.AveragePooling2D(pool_size = (3,1), strides = (2,1), name='AP3')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(7,1), strides = (1), activation = 'relu', padding='valid', name='CV4')(x)
    x = tf.keras.layers.AveragePooling2D(pool_size = (3,1), strides = (2,1), name='AP4')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Flatten(name='Flatten')(x)
    return x

def one_hot(y_, n_classes=6):
    # Function to encode neural one-hot output labels from number indexes
    # e.g.:
    # one_hot(y_=[[5], [0], [3]], n_classes=6):
    #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS 

def decode_one_hot(y_, n_classes=6):
    new_y = np.zeros([int(y_.size/n_classes)])
    for i in range(0,int(y_.size/n_classes)):
        max = 0
        for j in range(0,n_classes):
            if(y_[i,max] < y_[i,j]):
                max = j
        new_y[i] = max;
        
    return new_y  # Returns FLOATS 

def resample_data_1D(x_data, num):  
    num-=1
    data_len = 1
    data_size = np.shape(x_data)[0]
    dif_range = num-1
    up_xdata = np.zeros(dif_range*(data_size-1)+data_size)
    new_xdata = np.zeros(num)
    for j in range(data_size-1):
        for z in range(dif_range+1):
            up_xdata[z+(j*(dif_range+1))] = ((x_data[j+1] - x_data[j])/(dif_range+1))*z + x_data[j]
    up_xdata[dif_range*(data_size-1)+data_size-1] = x_data[data_size-1]
    new_xdata = up_xdata[::data_size-1]
    
    return new_xdata

def reshape_input(x_train, y_train, x_valid, y_valid, x_test, y_test):
    
    x_train = x_train.reshape(int(x_train.size/(resample_num+12)),(resample_num+12),1)
    y_train = y_train.reshape(int(y_train.size),1,1)
    x_valid = x_valid.reshape(int(x_valid.size/(resample_num+12)),(resample_num+12),1)
    y_valid = y_valid.reshape(int(y_valid.size),1,1) 
    x_test = x_test.reshape(int(x_test.size/(resample_num+12)),(resample_num+12),1)
    y_test = y_test.reshape(int(y_test.size),1,1) 
                
    train_time = x_train[:, resample_num+10, :].flatten()
    valid_time = x_valid[:, resample_num+10, :].flatten()
    test_time = x_test[:, resample_num+10, :].flatten()
    
    train_time = train_time.reshape(train_time.size,1)
    valid_time = valid_time.reshape(valid_time.size,1)
    test_time = test_time.reshape(test_time.size,1)
    
    ss1 = sklearn.preprocessing.StandardScaler()
    ss1.fit(train_time)
    train_time = ss1.transform(train_time)
    valid_time = ss1.transform(valid_time)
    test_time = ss1.transform(test_time)
    
    train_amplitude = x_train[:, resample_num+11, :].flatten()
    valid_amplitude = x_valid[:, resample_num+11, :].flatten()
    test_amplitude = x_test[:, resample_num+11, :].flatten()
    
    train_amplitude = train_amplitude.reshape(train_amplitude.size,1)
    valid_amplitude = valid_amplitude.reshape(valid_amplitude.size,1)
    test_amplitude = test_amplitude.reshape(test_amplitude.size,1)
    
    ss2 = sklearn.preprocessing.StandardScaler()
    ss2.fit(train_amplitude)
    train_amplitude = ss2.transform(train_amplitude)
    valid_amplitude = ss2.transform(valid_amplitude)
    test_amplitude = ss2.transform(test_amplitude)
    
    return x_train, y_train, x_valid, y_valid, x_test, y_test, train_time, valid_time, test_time, train_amplitude, valid_amplitude, test_amplitude
    
def reshape_input2(x_train, y_train, x_test, y_test):
    
    x_train = x_train.reshape(int(x_train.size/(resample_num+12)),(resample_num+12),1)
    y_train = y_train.reshape(int(y_train.size),1,1)
    x_test = x_test.reshape(int(x_test.size/(resample_num+12)),(resample_num+12),1)
    y_test = y_test.reshape(int(y_test.size),1,1) 
                
    train_time = x_train[:, resample_num+10, :].flatten()
    test_time = x_test[:, resample_num+10, :].flatten()
    
    train_time = train_time.reshape(train_time.size,1)
    test_time = test_time.reshape(test_time.size,1)
    
    ss1 = sklearn.preprocessing.StandardScaler()
    ss1.fit(train_time)
    train_time = ss1.transform(train_time)
    test_time = ss1.transform(test_time)
    
    train_amplitude = x_train[:, resample_num+11, :].flatten()
    test_amplitude = x_test[:, resample_num+11, :].flatten()
    
    train_amplitude = train_amplitude.reshape(train_amplitude.size,1)
    test_amplitude = test_amplitude.reshape(test_amplitude.size,1)
    
    ss2 = sklearn.preprocessing.StandardScaler()
    ss2.fit(train_amplitude)
    train_amplitude = ss2.transform(train_amplitude)
    test_amplitude = ss2.transform(test_amplitude)
    
    return x_train, y_train, x_test, y_test, train_time, test_time, train_amplitude,test_amplitude

def K_fold(x_train, y_train, x_test, y_test):
    num_folds = 0
    str_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=44)
    y_train = y_train-1
    y_test = y_test-1
    accs = []
    recalls = []
    specs = []
    rocs = []
    f1s = []
    
    for train_idx, valid_idx in str_kf.split(x_train, y_train):
        num_folds += 1
        print(f'--------------------{num_folds}번째 KFold-------------------')
        print(f'train_idx_len : {len(train_idx)} / valid_idx_len : {len(valid_idx)}')

        data_train, data_valid = x_train[train_idx], x_train[valid_idx]
        label_train, label_valid = y_train[train_idx], y_train[valid_idx]
                
        # Data augmentation
        aug_xtrain, aug_ytrain = data_aug(data_train,label_train, [0, 1, 2, 3, 4, 5, 6, 7])
        
        # Data shuffle
        tmp = [[x,y] for x,y in zip(aug_xtrain,aug_ytrain)]
        random.shuffle(tmp)
        aug_xtrain = [n[0] for n in tmp]
        aug_ytrain = [n[1] for n in tmp]
        aug_xtrain = np.array(aug_xtrain)
        aug_ytrain = np.array(aug_ytrain)
        
        # Data reshape
        aug_xtrain, aug_ytrain, data_valid, label_valid, x_test1, y_test1, train_time, valid_time, test_time, train_amplitude, valid_amplitude, test_amplitude = reshape_input(aug_xtrain, aug_ytrain, data_valid, label_valid, x_test, y_test)
        
        # Data save
        # sio.savemat('./Stretch_press/Result/data'+ str(num_folds)+'.mat',{'x_train' : aug_xtrain, 'y_train' : aug_ytrain, 'x_valid' : data_valid, 'y_valid' : label_valid})
        sio.savemat('./Stretch_press/Result/data'+ str(num_folds)+'.mat',{'aug_xtrain' : aug_xtrain, 
                                                                          'aug_ytrain' : aug_ytrain, 
                                                                          'data_valid' : data_valid, 
                                                                          'label_valid' : label_valid, 
                                                                          'x_test1' : x_test1, 
                                                                          'y_test1' : y_test1, 
                                                                          'train_time' : train_time, 
                                                                          'valid_time' : valid_time, 
                                                                          'test_time' : test_time, 
                                                                          'train_amplitude' : train_amplitude, 
                                                                          'valid_amplitude' : valid_amplitude, 
                                                                          'test_amplitude' : test_amplitude
                                                                          })
                
        #model
        model = dnn_model()
        
        callback_list = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10),
            tf.keras.callbacks.ModelCheckpoint(filepath='./Stretch_press/model/model_best_fold' + str(num_folds) + '.h5', monitor='val_loss', mode='min', verbose=0, save_best_only=True, save_weights_only=True),
        ]

        Input1 = aug_xtrain[:, 5:(resample_num+5), :]
        Val_Input1 = data_valid[:, 5:(resample_num+5), :]
        
        Input2 = train_time
        Input3 = train_amplitude
        Val_Input2 = valid_time
        Val_Input3 = valid_amplitude
                       
        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        model.fit([Input1, Input2, Input3], aug_ytrain, batch_size=32, validation_data = ([Val_Input1, Val_Input2, Val_Input3],label_valid), epochs=300, callbacks=callback_list)
        model.load_weights('./Stretch_press/model/model_best_fold'+str(num_folds)+'.h5')
        vPred = model.predict([Val_Input1, Val_Input2, Val_Input3])        
        loss, acc = model.evaluate([Val_Input1, Val_Input2, Val_Input3], label_valid)
        # Pred = decode_one_hot(vPred,8)
        # result = pd.DataFrame({'Pred': Pred, 'label':y_test1.flatten()})
        # result.to_csv("./Stretch_press/Result/" + 'Result'+str(num_folds)+'.csv')
        
        Pred = decode_one_hot(vPred, 8)
        # Pred = one_hot(Pred)
        accuracy, precision, recall, spec, roc_auc, bal_acc, f1 = get_clf_eval(label_valid.flatten(), Pred.flatten(), vPred, [0, 1, 2, 3, 4, 5, 6, 7])
        result = pd.DataFrame({'accuracy': accuracy, 'precision':precision, 'recall':recall, 'spec':spec, 'roc_auc':roc_auc, 'bal_acc':bal_acc, 'f1':f1})
        result.to_csv("./Stretch_press/Result/" + 'score_fold'+str(num_folds)+'.csv')
        
        accs.append(acc)
        recalls.append(recall)
        specs.append(spec)
        rocs.append(roc_auc)
        f1s.append(f1)
    
    print(f'acc : {accs}')
    print(f'recalls : {recalls}')
    print(f'specs : {specs}')
    print(f'rocs : {rocs}')
    print(f'f1s : {f1s}')
    
    mean_acc = round(np.mean(accs),4)
    std_acc = round(np.std(accs),4)
    mean_recall = round(np.mean(recalls),4)
    std_recall = round(np.std(recalls),4)
    mean_spec = round(np.mean(specs),4)
    std_spec = round(np.std(specs),4)
    mean_roc = round(np.mean(rocs),4)
    std_roc = round(np.std(rocs),4)
    mean_f1 = round(np.mean(f1s),4)
    std_f1 = round(np.std(f1s),4)
    
    print(f'recalls : {mean_recall} std : {std_recall}')
    print(f'specs : {mean_spec} std : {std_spec}')
    print(f'accs : {mean_acc} std : {std_acc}')
    print(f'f1s : {mean_f1} std : {std_f1}')   
    print(f'rocs : {mean_roc} std : {std_roc}')
    
    return aug_xtrain, aug_ytrain, x_test1, y_test1, test_time, test_amplitude

def data_aug(x_data, y_data, labels):
    random.seed(1)
    
    # find maximum
    max_count = 0
    for i in labels:
        print(np.size(np.where(y_data == i)))
        if(max_count < np.size(np.where(y_data == i)[0])):
            max_count = np.size(np.where(y_data == i)[0])
        
    # augmentation
    new_data = np.zeros((1,resample_num+12))
    new_ydata = np.zeros((1,1))
    for i in labels:
        aug_num = max_count - np.size(np.where(y_data == i)[0])
        aug_data = np.where(y_data == i)[0]
        for j in range(aug_num):
            rand_aug = random.randrange(np.shape(aug_data)[0])
            rand_num = random.randrange(-5,6)
            new_data[0][0:5] = 0
            new_data[0][5:resample_num+5] = x_data[aug_data[rand_aug]][5+rand_num:resample_num+5+rand_num]
            new_data[0][resample_num+5:resample_num+10] = 0
            new_data[0][resample_num+10:resample_num+12] = x_data[aug_data[rand_aug]][resample_num+10:resample_num+12]
            
            x_data = np.concatenate((x_data,new_data))
            new_ydata[0][0] = i
            y_data = np.concatenate((y_data,new_ydata))
        
    print(np.size(y_data))
    return x_data, y_data

        
def get_clf_eval(y_test, pred=None, pred_proba=None, classes=[0, 1]):
    confusion = confusion_matrix(y_test, pred, labels=classes)
    # accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred,average=None)
    recall = recall_score(y_test, pred,average=None)
    f1 = f1_score(y_test, pred,average=None)
    roc_auc = roc_auc_score(y_test, pred_proba, average=None, multi_class='ovr')
    
    TP = np.zeros(len(classes))
    FP = np.zeros(len(classes))
    FN = np.zeros(len(classes))
    TN = np.zeros(len(classes))
    spec = np.zeros(len(classes))
    accuracy = np.zeros(len(classes))
    for i in classes:
        for j in classes:
            for z in classes:
                if i == j and j == z :
                    TP[i] += confusion[j,z]
                elif i == j and i != z:
                    FN[i] += confusion[j,z]
                elif i != j and i == z:
                    FP[i] += confusion[j,z]                    
                else : 
                    TN[i] += confusion[j,z]               
        spec[i] = TN[i]/(FP[i]+TN[i])
        
    bal_acc = ((recall + spec) / 2)
    
    for i in classes:
        accuracy[i] = (TP[i] + TN[i])/(TP[i]+FN[i]+FP[i]+TN[i])
    
    ### roc
    for i in range(len(classes)):
        ax_bottom = plt.subplot(2, 4, i+1)
        tpr_list = [0]
        fpr_list = [0]
        y_proba = pred_proba[:,i]
        c = classes[i]
        y_real = [1 if y == c else 0 for y in y_test]
        for j in range(len(y_proba)):
            threshold = y_proba[j]
            y_pred = y_proba >= threshold
            tpr, fpr = calculate_tpr_fpr(y_real, y_pred)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        plot_roc_curve(tpr_list,fpr_list,scatter=False,ax=ax_bottom)        
        if(i==0):
            ax_bottom.set_title("ROC Curve OvR Stretching 10%")
        if(i==1):
            ax_bottom.set_title("ROC Curve OvR Stretching 20%")
        if(i==2):
            ax_bottom.set_title("ROC Curve OvR Stretching 30%")
        if(i==3):
            ax_bottom.set_title("ROC Curve OvR Pressing 0.5N")
        if(i==4):
            ax_bottom.set_title("ROC Curve OvR Pressing 1N")
        if(i==5):
            ax_bottom.set_title("ROC Curve OvR Pressing 1.5N")
        if(i==6):
            ax_bottom.set_title("ROC Curve OvR Pressing 2N")
        if(i==7):
            ax_bottom.set_title("ROC Curve OvR Pressing 2.5N")
        
    plt.show()
    
    print('오차 행렬')
    print(confusion)
    print(f'정확도 : {accuracy}')
    print(f'정밀도 : {precision}')
    print(f'재현율 : {recall}')
    print(f'특이성 : {spec}')
    print(f'F1 : {f1}')
    print(f'AUC : {roc_auc}')
    print(f'bal_acc : {bal_acc}')
    # print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율(Sensitivity): {2:.4f}, 특이성(Specificity): {3:.4f}, F1: {4:.4f}, AUC: {5:.4f}, bal_acc: {6:.4f}'.format(accuracy, precision, recall, spec, f1, roc_auc, bal_acc))
    return accuracy, precision, recall, spec, roc_auc, bal_acc, f1

def calculate_tpr_fpr(y_real, y_pred):
    '''
    Calculates the True Positive Rate (tpr) and the True Negative Rate (fpr) based on real and predicted observations
    
    Args:
        y_real: The list or series with the real classes
        y_pred: The list or series with the predicted classes
        
    Returns:
        tpr: The True Positive Rate of the classifier
        fpr: The False Positive Rate of the classifier
    '''
    
    # Calculates the confusion matrix and recover each element
    cm = confusion_matrix(y_real, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    
    # Calculates tpr and fpr
    tpr =  TP/(TP + FN) # sensitivity - true positive rate
    fpr = 1 - TN/(TN+FP) # 1-specificity - false positive rate
    
    return tpr, fpr

def plot_roc_curve(tpr, fpr, scatter = True, ax = None):
    '''
    Plots the ROC Curve by using the list of coordinates (tpr and fpr).
    
    Args:
        tpr: The list of TPRs representing each coordinate.
        fpr: The list of FPRs representing each coordinate.
        scatter: When True, the points used on the calculation will be plotted with the line (default = True).
    '''
    if ax == None:
        plt.figure(figsize = (5, 5))
        ax = plt.axes()
    
    if scatter:
        sns.scatterplot(x = fpr, y = tpr, ax = ax)
    sns.lineplot(x = fpr, y = tpr, ax = ax)
    sns.lineplot(x = [0, 1], y = [0, 1], color = 'green', ax = ax)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    
x_data, y_data = loadMatlabData("./Stretch_press/data/")

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,test_size = 0.2, stratify=y_data , shuffle=True, random_state=99)

# ### Kfold
x_train, y_train, x_test, y_test, test_time, test_amplitude = K_fold(x_train, y_train, x_test, y_test)

# result = pd.DataFrame(x_train.squeeze(),y_train.squeeze())
# result.to_csv("./Stretch_press/Result/" + 'Result1.csv')

#### Ensemble load weight
# vPreds = []

# for i in range(5):
#     model = dnn_model()
#     model.load_weights('./Stretch_press/model/model_best_fold'+str(i+1)+'.h5')
#     aug_xtrain, aug_ytrain, data_valid, label_valid, x_test1, y_test1, train_time, valid_time, test_time, train_amplitude, valid_amplitude, test_amplitude = loadMatlabData2('./Stretch_press/model/',i+1)
    
#     vPred = model.predict([x_test1[:, 5:resample_num+5, :], test_time, test_amplitude]).squeeze()
#     vPreds.append(vPred)
#     process = psutil.Process(os.getpid())
#     print("Mem : " ,process.memory_info().rss/1024**2)
    
# vPred = (vPreds[0] + vPreds[1] + vPreds[2] + vPreds[3] + vPreds[4])/5
# Pred = decode_one_hot(vPred,8)
# accuracy, precision, recall, spec, roc_auc, bal_acc, f1 = get_clf_eval(y_test1.flatten(), Pred.flatten(), vPred, [0, 1, 2, 3, 4, 5, 6, 7])
# result = pd.DataFrame({'accuracy': accuracy, 'precision':precision, 'recall':recall, 'spec':spec, 'roc_auc':roc_auc, 'bal_acc':bal_acc, 'f1':f1})
# result.to_csv("./Stretch_press/Result/" + 'vPred_ensemble.csv')

# mean_acc = round(np.mean(accuracy),4)
# std_acc = round(np.std(accuracy),4)
# mean_recall = round(np.mean(recall),4)
# std_recall = round(np.std(recall),4)
# mean_spec = round(np.mean(spec),4)
# std_spec = round(np.std(spec),4)
# mean_roc = round(np.mean(roc_auc),4)
# std_roc = round(np.std(roc_auc),4)
# mean_f1 = round(np.mean(f1),4)
# std_f1 = round(np.std(f1),4)

# print(f'recalls : {mean_recall} std : {std_recall}')
# print(f'specs : {mean_spec} std : {std_spec}')
# print(f'accs : {mean_acc} std : {std_acc}')
# print(f'f1s : {mean_f1} std : {std_f1}')   
# print(f'rocs : {mean_roc} std : {std_roc}')

##### test data
# accs = []
# recalls = []
# specs = []
# rocs = []
# f1s = []
# for i in range(5):
#     model = dnn_model()
#     model.load_weights('./Stretch_press/model/model_best_fold'+str(i+1)+'.h5')
#     aug_xtrain, aug_ytrain, data_valid, label_valid, x_test1, y_test1, train_time, valid_time, test_time, train_amplitude, valid_amplitude, test_amplitude = loadMatlabData2('./Stretch_press/model/',i+1)    
    
#     # vPred = model.predict([data_valid[:, 5:resample_num+5, :], valid_time, valid_amplitude])
#     # Pred = decode_one_hot(vPred,8)        
#     # accuracy, precision, recall, spec, roc_auc, bal_acc, f1 = get_clf_eval(label_valid.flatten(), Pred.flatten(), vPred, [0, 1, 2, 3, 4, 5, 6, 7])
    
#     vPred = model.predict([x_test1[:, 5:resample_num+5, :], test_time, test_amplitude])
#     Pred = decode_one_hot(vPred,8)
#     accuracy, precision, recall, spec, roc_auc, bal_acc, f1 = get_clf_eval(y_test1.flatten(), Pred.flatten(), vPred, [0, 1, 2, 3, 4, 5, 6, 7])
#     result = pd.DataFrame({'accuracy': accuracy, 'precision':precision, 'recall':recall, 'spec':spec, 'roc_auc':roc_auc, 'bal_acc':bal_acc, 'f1':f1})
#     result.to_csv("./Stretch_press/Result/" + 'score_fold'+str(i+1)+'.csv')
    
#     accs.append(accuracy)
#     recalls.append(recall)
#     specs.append(spec)
#     rocs.append(roc_auc)
#     f1s.append(f1)
    
# print(f'acc : {accs}')
# print(f'recalls : {recalls}')
# print(f'specs : {specs}')
# print(f'rocs : {rocs}')
# print(f'f1s : {f1s}')

# mean_acc = round(np.mean(accs),4)
# std_acc = round(np.std(accs),4)
# mean_recall = round(np.mean(recalls),4)
# std_recall = round(np.std(recalls),4)
# mean_spec = round(np.mean(specs),4)
# std_spec = round(np.std(specs),4)
# mean_roc = round(np.mean(rocs),4)
# std_roc = round(np.std(rocs),4)
# mean_f1 = round(np.mean(f1s),4)
# std_f1 = round(np.std(f1s),4)

# print(f'recalls : {mean_recall} std : {std_recall}')
# print(f'specs : {mean_spec} std : {std_spec}')
# print(f'accs : {mean_acc} std : {std_acc}')
# print(f'f1s : {mean_f1} std : {std_f1}')   
# print(f'rocs : {mean_roc} std : {std_roc}')