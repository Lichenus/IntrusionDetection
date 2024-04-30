import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras import layers
from keras import optimizers
from keras.layers import Dense, LSTM, MaxPool1D, Flatten, Dropout,Conv1D,BatchNormalization
from keras.models import Sequential
from keras.layers import Input
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
from libs import ds_layer               # Dempster-Shafer layer
from libs import utility_layer_train    # Utility layer for training
from libs import utility_layer_test     # Utility layer for testing
from libs import AU_imprecision         # Metric average utility for set-valued classification
import model.Set_valued as SV
from scipy.optimize import minimize
import math


def pLSTM(hidden_units, num_class, model_filepath,is_train,
                     x_train, y_train, x_test, y_test, output_confusion_matrix, pic_filepath_loss=0, pic_filepath_acc=0):

    inputs = Input((x_train.shape[1], x_train.shape[2]))
    lstm1 = LSTM(units=hidden_units, return_sequences=True,dropout=0,recurrent_dropout=0)(inputs)
    lstm2 = LSTM(units=hidden_units, return_sequences=True,dropout=0,recurrent_dropout=0)(lstm1)
    flatten1 = Flatten()(lstm2)
    fc1 = Dense(64, activation='relu')(flatten1)

    outputs = Dense(units=num_class, activation='softmax')(fc1)
    model = keras.Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer=optimizers.Nadam(learning_rate=0.001,beta_1=0.9, beta_2=0.999, epsilon=None,
                                                   schedule_decay=0.004), loss='CategoricalCrossentropy',metrics=['accuracy'])

    # disusedModel.summary()

    checkpoint_callback = ModelCheckpoint(model_filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                          save_weights_only=True, save_frequency=1)
    if is_train:
        h = model.fit(x_train, y_train, batch_size=64, epochs=90,
                      verbose=1, callbacks=[checkpoint_callback], validation_data=(x_test, y_test), shuffle=True)
        history = h.history
        # epochs = range(len(history['loss']))
        # plt.plot(epochs, history['loss'], 'royalblue', label='Train loss')
        # plt.plot(epochs, history['val_loss'], 'crimson', label='Valid loss')  # Test loss
        # plt.xlabel("Epochs")
        # plt.ylabel("Loss")
        # plt.rcParams["figure.dpi"] = 300
        # plt.legend()
        # plt.savefig(pic_filepath_loss, dpi=300)
        # plt.close()
        #
        # plt.plot(epochs, history['accuracy'], 'royalblue', label='Train accuracy')
        # plt.plot(epochs, history['val_accuracy'], 'crimson', label='Valid accuracy')  # Test loss
        # plt.xlabel("Epochs")
        # plt.ylabel("Accuracy")
        # plt.rcParams["figure.dpi"] = 300
        # plt.legend()
        # plt.savefig(pic_filepath_acc, dpi=300)
        # plt.close()
    else:
        model.load_weights(model_filepath).expect_partial()
        # model_PR.load_weights(filepath1)
        # model_PR.evaluate(x_train, y_train, batch_size=25, verbose=1)
        # model_PR.evaluate(x_test, y_test, batch_size=32, verbose=1)
        if output_confusion_matrix == 1:
            y_pred = model.predict(x_test)
            y_pred = y_pred.argmax(axis=1)
            y_test = y_test.argmax(axis=1)
            confusion_mtx = confusion_matrix(y_test, y_pred)
            normalized_confusion_mtx = np.round(confusion_matrix(y_test, y_pred, normalize='true'), 3)
            f1 = f1_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            acc = accuracy_score(y_true=y_test, y_pred=y_pred)
            recall = recall_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            precision = precision_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            print(confusion_mtx)
            print(normalized_confusion_mtx)
            print('accuracy: ' + str(acc))
            print('recall: ' + str(recall))
            print('precision: ' + str(precision))
            print('F1-score: ' + str(f1))
            return acc, recall, precision, f1


def ELSTM(hidden_units, num_class, prototypes,nu, model_filepath,evi_filepath,mid_filepath,
          load_and_train, is_load, x_train, y_train, x_test, y_test, output_confusion_matrix):
    inputs = Input((x_train.shape[1], x_train.shape[2]))
    lstm1 = LSTM(units=hidden_units, return_sequences=True, dropout=0, recurrent_dropout=0)(inputs)
    lstm2 = LSTM(units=hidden_units, return_sequences=True, dropout=0, recurrent_dropout=0)(lstm1)
    flatten1 = Flatten()(lstm2)
    # DS layer
    ED = ds_layer.DS1(prototypes, hidden_units)(flatten1)
    ED_ac = ds_layer.DS1_activate(prototypes)(ED)
    mass_prototypes = ds_layer.DS2(prototypes, num_class)(ED_ac)
    mass_prototypes_omega = ds_layer.DS2_omega(prototypes, num_class)(mass_prototypes)
    mass_Dempster = ds_layer.DS3_Dempster(prototypes, num_class)(mass_prototypes_omega)
    mass_Dempster_normalize = ds_layer.DS3_normalize()(mass_Dempster)

    # Utility layer for training
    outputs = utility_layer_train.DM(nu, num_class)(mass_Dempster_normalize)
    model_evi = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model_evi.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
                                         schedule_decay=0.004),
        loss='CategoricalCrossentropy',
        metrics=['accuracy'])
    # model_evi.summary()

    if load_and_train == 1:
        # load the weights of probabilistic classifier
        # and get the feature for training DS layers
        model_evi.load_weights(model_filepath).expect_partial()
        feature = tf.keras.Model(inputs=[inputs], outputs=[flatten1])
        train_feature_for_DS = feature.predict(x_train)
        test_feature_for_DS = feature.predict(x_test)

        # training DS layers
        inputss = tf.keras.layers.Input(hidden_units)
        ED = ds_layer.DS1(prototypes, hidden_units)(inputss)
        ED_ac = ds_layer.DS1_activate(prototypes)(ED)
        mass_prototypes = ds_layer.DS2(prototypes, num_class)(ED_ac)
        mass_prototypes_omega = ds_layer.DS2_omega(prototypes, num_class)(mass_prototypes)
        mass_Dempster = ds_layer.DS3_Dempster(prototypes, num_class)(mass_prototypes_omega)
        mass_Dempster_normalize = ds_layer.DS3_normalize()(mass_Dempster)
        outputss = utility_layer_train.DM(nu, num_class)(mass_Dempster_normalize)
        model_mid = tf.keras.Model(inputs=[inputss], outputs=[outputss])
        model_mid.compile(
            optimizer=keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
                                             schedule_decay=0.004),
            # 0.001
            loss='CategoricalCrossentropy',
            metrics=['accuracy'])
        # model_mid.summary()
        mid_callback = ModelCheckpoint(mid_filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                       save_weights_only=True, save_frequency=1)
        h = model_mid.fit(train_feature_for_DS, y_train, batch_size=64, epochs=20, verbose=1,
                          validation_data=(test_feature_for_DS, y_test), callbacks=[mid_callback], shuffle=True)

        # feed the trained paramters to the evidential disusedModel
        model_evi.load_weights(model_filepath).expect_partial()
        DS1_W = tf.reshape(model_mid.layers[1].get_weights()[0], [1, prototypes, hidden_units])
        DS1_activate_W = model_mid.layers[2].get_weights()
        DS2_W = model_mid.layers[3].get_weights()
        model_evi.layers[4].set_weights(DS1_W)
        model_evi.layers[5].set_weights(DS1_activate_W)
        model_evi.layers[6].set_weights(DS2_W)

        checkpoint_callback = ModelCheckpoint(
            evi_filepath, monitor='val_accuracy', verbose=1,
            save_best_only=True, save_weights_only=True,
            save_frequency=1)
        model_evi.fit(x_train, y_train,batch_size=64, epochs=18, verbose=1,
                      callbacks=[checkpoint_callback], validation_data=(x_test, y_test), shuffle=True)

    if is_load == 1:
        model_evi.load_weights(evi_filepath).expect_partial()
        # model_evi.evaluate(x_train, y_train, batch_size=64, verbose=1)
        # model_evi.evaluate(x_test, y_test, batch_size=64, verbose=1)
        if output_confusion_matrix == 1:
            y_pred = model_evi.predict(x_test)
            y_pred = y_pred.argmax(axis=1)
            y_test = y_test.argmax(axis=1)
            confusion_mtx = confusion_matrix(y_test, y_pred)
            f1 = f1_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            acc = accuracy_score(y_true=y_test, y_pred=y_pred)
            recall = recall_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            preccision = precision_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            # print(confusion_mtx)
            # print(np.round(confusion_matrix(y_test, y_pred,normalize='true'),3))
            print('proto: ' + str(prototypes) + 'nu: ' + str(nu))
            print('accuracy: ' + str(acc))
            print('recall: ' + str(recall))
            print('precision: ' + str(preccision))
            print('F1-score: ' + str(f1))
            return y_pred


def ELSTM_direct_training(hidden_units, num_class, prototypes,nu, evi_filepath,
                          is_train, x_train, y_train, x_test, y_test, output_confusion_matrix):
    inputs = Input((x_train.shape[1], x_train.shape[2]))
    lstm1 = LSTM(units=hidden_units, return_sequences=True, dropout=0, recurrent_dropout=0)(inputs)
    lstm2 = LSTM(units=hidden_units, return_sequences=True, dropout=0, recurrent_dropout=0)(lstm1)
    flatten1 = Flatten()(lstm2)
    fc1 = Dense(64, activation='relu')(flatten1)
    # DS layer
    ED = ds_layer.DS1(prototypes, 64)(fc1)
    ED_ac = ds_layer.DS1_activate(prototypes)(ED)
    mass_prototypes = ds_layer.DS2(prototypes, num_class)(ED_ac)
    mass_prototypes_omega = ds_layer.DS2_omega(prototypes, num_class)(mass_prototypes)
    mass_Dempster = ds_layer.DS3_Dempster(prototypes, num_class)(mass_prototypes_omega)
    mass_Dempster_normalize = ds_layer.DS3_normalize()(mass_Dempster)

    # Utility layer for training
    outputs = utility_layer_train.DM(nu, num_class)(mass_Dempster_normalize)
    model_evi = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model_evi.compile(
        optimizer=optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7,
                                         schedule_decay=0.004),loss='CategoricalCrossentropy',metrics=['accuracy'])
    # model_evi.summary()
    checkpoint_callback = ModelCheckpoint(evi_filepath, monitor='val_accuracy', verbose=1,save_best_only=True, save_weights_only=True,save_frequency=1)
    if is_train:
        h = model_evi.fit(x_train, y_train,batch_size=64, epochs=90, verbose=1,
                      callbacks=[checkpoint_callback], validation_data=(x_test, y_test), shuffle=True)

    else:
        model_evi.load_weights(evi_filepath).expect_partial()
        # model_evi.evaluate(x_train, y_train, batch_size=64, verbose=1)
        # model_evi.evaluate(x_test, y_test, batch_size=64, verbose=1)
        if output_confusion_matrix == 1:
            y_pred = model_evi.predict(x_test)
            y_pred = y_pred.argmax(axis=1)
            y_test = y_test.argmax(axis=1)
            confusion_mtx = confusion_matrix(y_test, y_pred)
            f1 = f1_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            acc = accuracy_score(y_true=y_test, y_pred=y_pred)
            recall = recall_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            precision = precision_score(y_true=y_test, y_pred=y_pred, labels=None, average='macro')
            # print(confusion_mtx)
            # print(np.round(confusion_matrix(y_test, y_pred,normalize='true'),3))
            print('proto: ' + str(prototypes) + 'nu: ' + str(nu))
            print('accuracy: ' + str(acc))
            print('recall: ' + str(recall))
            print('precision: ' + str(precision))
            print('F1-score: ' + str(f1))
            return acc, recall, precision, f1


def ELSTM_SV(hidden_units,num_class, number_act_set, act_set, tol, nu, prototypes,
           utility_matrix, is_load, filepath, x_test, numerical_y_test):
    inputs = Input((x_test.shape[1], x_test.shape[2]))
    lstm1 = LSTM(units=hidden_units, return_sequences=True, dropout=0, recurrent_dropout=0)(inputs)
    lstm2 = LSTM(units=hidden_units, return_sequences=True, dropout=0, recurrent_dropout=0)(lstm1)
    flatten1 = Flatten()(lstm2)
    fc1 = Dense(64, activation='relu')(flatten1)
    # DS layer
    ED = ds_layer.DS1(prototypes, 64)(fc1)
    ED_ac = ds_layer.DS1_activate(prototypes)(ED)
    mass_prototypes = ds_layer.DS2(prototypes, num_class)(ED_ac)
    mass_prototypes_omega = ds_layer.DS2_omega(prototypes, num_class)(mass_prototypes)
    mass_Dempster = ds_layer.DS3_Dempster(prototypes, num_class)(mass_prototypes_omega)
    mass_Dempster_normalize = ds_layer.DS3_normalize()(mass_Dempster)

    # Utility layer for test
    outputs = utility_layer_test.DM_test(num_class, number_act_set, nu)(mass_Dempster_normalize)
    model_evi_SV = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model_evi_SV.compile(
        optimizer=optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7,
                                   schedule_decay=0.004), loss='CategoricalCrossentropy', metrics=['accuracy'])
    if is_load:
        model_evi_SV.layers[-1].set_weights(tf.reshape(utility_matrix, [1, number_act_set, num_class]))
        model_evi_SV.load_weights(filepath).expect_partial()
        results = tf.argmax(model_evi_SV.predict(x_test), -1)
        # print(model_evi_SV.predict(x_test).shape)
        # print(results.shape)
        imprecise_results = []
        sv_counter = 0
        for i in range(len(results)):
            act_local = results[i]
            set_valued_results = act_set[act_local]
            imprecise_results.append(set_valued_results)
            if len(set_valued_results) > 1:
                sv_counter = sv_counter + 1
        # print(imprecise_results)
        average_utility_imprecision = AU_imprecision.average_utility(utility_matrix, results, numerical_y_test, act_set)
        sv_rate = (sv_counter / len(results)) * 100
        print('prototypes=' + str(prototypes) + ' nu=' + str(nu) + ' tol=' + str(tol))
        print('**** **** **** **** **** **** AU = ' + str(
            average_utility_imprecision) + ' **** **** **** **** **** *****')
        print('**** **** **** **** **** **** SV Rates assignment = ' + str(sv_rate) + ' **** **** **** **** **** *****')
        return average_utility_imprecision, sv_rate


def LSTM_SV(hidden_units,num_class, number_act_set, act_set, tol, nu,
           utility_matrix, is_load, filepath, x_test, numerical_y_test):
    inputs = Input((x_test.shape[1], x_test.shape[2]))
    lstm1 = LSTM(units=hidden_units, return_sequences=True, dropout=0, recurrent_dropout=0)(inputs)
    lstm2 = LSTM(units=hidden_units, return_sequences=True, dropout=0, recurrent_dropout=0)(lstm1)
    flatten1 = Flatten()(lstm2)
    fc1 = Dense(64, activation='relu')(flatten1)
    softmax = Dense(units=num_class, activation='softmax')(fc1)

    outputs = utility_layer_test.DM_test(num_class, number_act_set, nu)(softmax)
    model_SV = keras.Model(inputs=[inputs], outputs=[outputs])
    model_SV.compile(optimizer=keras.optimizers.Nadam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004),
        loss='CategoricalCrossentropy', metrics=['accuracy'])
    # disusedModel.summary()
    checkpoint_callback = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                          save_weights_only=True, save_frequency=1)
    if is_load:
        model_SV.layers[-1].set_weights(tf.reshape(utility_matrix, [1, number_act_set, num_class]))
        model_SV.load_weights(filepath).expect_partial()

        results = tf.argmax(model_SV.predict(x_test), -1)
        # print(model_evi_SV.predict(x_test).shape)
        # print(results.shape)
        imprecise_results = []
        sv_counter = 0
        for i in range(len(results)):
            act_local = results[i]
            set_valued_results = act_set[act_local]
            imprecise_results.append(set_valued_results)

            if len(set_valued_results) > 1:
                sv_counter = sv_counter + 1

        average_utility_imprecision = AU_imprecision.average_utility(utility_matrix, results, numerical_y_test, act_set)
        sv_rate = (sv_counter / len(results)) * 100
        print('**** **** **** **** **** **** tol=' + str(tol) + ' **** **** **** **** **** *****')
        print('**** **** **** **** **** **** AU = ' + str(
            average_utility_imprecision) + ' **** **** **** **** **** *****')
        print('**** **** **** **** **** **** SV Rates assignment = ' + str(
            sv_rate) + ' **** **** **** **** **** *****')
        return average_utility_imprecision, sv_rate


def Test_LSTM_SV(hidden_units,data_WIDTH,num_class, number_act_set, nu,
                filepath, x_test, numerical_y_test):
    class_set = list(range(num_class))
    act_set = SV.PowerSets(class_set, no_empty=True, is_sorted=True)

    for i in [0, 1, 2, 3, 4]:
        m = 0
        UM = SV.utility_mtx(num_class, act_set=act_set, class_set=class_set, tol_i=i, m=m)
        LSTM_SV(hidden_units=hidden_units, data_WIDTH=data_WIDTH, num_class=num_class, number_act_set=number_act_set, tol=0.1 * i + 0.5 + m,
               utility_matrix=UM, is_load=1, nu=nu,filepath=filepath,
               act_set=act_set, x_test=x_test, numerical_y_test=numerical_y_test)

    for i in [4]:
        m = 0.05
        UM = SV.utility_mtx(num_class, act_set=act_set, class_set=class_set, tol_i=i, m=m)
        LSTM_SV(hidden_units=hidden_units, data_WIDTH=data_WIDTH, num_class=num_class, number_act_set=number_act_set, tol=0.1 * i + 0.5 + m,
               utility_matrix=UM, is_load=1, nu=nu, filepath=filepath,
               act_set=act_set, x_test=x_test, numerical_y_test=numerical_y_test)

