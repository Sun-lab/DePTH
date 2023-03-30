# -------------------
# train a model
# -------------------


import os
import sys
import time

import random

import numpy as np
import pandas as pd
import tensorflow as tf

from DePTH import _utils


def train(hla_class, data_dir, model_dir, enc_method,
          lr, n_dense, n_units_str, dropout_flag, p_dropout,
          rseed, np_seed, tf_seed):

    input_args = locals()
    print("input args are", input_args)

    random.seed(rseed)
    np.random.seed(np_seed)
    tf.random.set_seed(tf_seed)

    patience = 10

    dropout_flag = (dropout_flag == 'True')

    if n_dense > 2:
        sys.exit("Error: number of dense layers>2, which is a case not coded for yet.")

    len_n_units_str = len(n_units_str)
    n_units_str_input = n_units_str[1:(len_n_units_str - 1)].split(',')
    n_units = [int(i) for i in n_units_str_input]
    print("n_units = ", n_units)
    if len(n_units) != n_dense:
        sys.exit("Error: n_dense and n_units do not match.")

    print("arguments after format processing are: ")
    print("data_dir = ", data_dir)
    print("model_dir = ", model_dir)
    print("enc_method = ", enc_method)
    print("lr = ", lr)
    print("n_dense = ", n_dense)
    print("n_units = ", n_units)
    print("dropout_flag = ", dropout_flag)
    print("p_dropout = ", p_dropout)

    setting_name = \
        enc_method + '_' + str(lr)[2:] + \
        '_dense' + str(n_dense) + \
        '_n_units_' + '_'.join([str(n) for n in n_units]) + \
        ('_dropout_p_' + str(p_dropout)[2:]) * int(dropout_flag)

    print(setting_name)

    checkpoint_path = model_dir

    start = time.time()

    (((HLA_encoded_train, CDR3_encoded_train, CDR3_len_train,
      cdr1_encoded_train, cdr2_encoded_train, cdr25_encoded_train),
      y2_train, n_pos_train, n_neg_train),
     ((HLA_encoded_valid, CDR3_encoded_valid, CDR3_len_valid,
      cdr1_encoded_valid, cdr2_encoded_valid, cdr25_encoded_valid),
      y2_valid, n_pos_valid, n_neg_valid)) = \
        _utils.get_data(hla_class, data_dir, enc_method, False)

    print("shape of encoded HLA sequence from training data: ", HLA_encoded_train.shape)
    print("shape of encoded HLA sequence from validation data: ", HLA_encoded_valid.shape)
    print("shape of encoded CDR3 sequence from training data: ", CDR3_encoded_train.shape)
    print("shape of encoded CDR3 length part from training data: ", CDR3_len_train.shape)
    print("shape of encoded CDR1 sequence from training data: ", cdr1_encoded_train.shape)
    print("shape of encoded CDR2 sequence from training data: ", cdr2_encoded_train.shape)
    print("shape of encoded CDR2.5 sequencefrom training data: ", cdr25_encoded_train.shape)
    print("shape of label from training data: ", y2_train.shape)

    # get the model
    model = _utils.get_model(HLA_shape=HLA_encoded_train.shape[1:],
                             CDR3_shape=CDR3_encoded_train.shape[1:],
                             len_shape=CDR3_len_train.shape[1:],
                             cdr1_shape=cdr1_encoded_train.shape[1:],
                             cdr2_shape=cdr2_encoded_train.shape[1:],
                             cdr25_shape=cdr25_encoded_train.shape[1:],
                             n_dense=n_dense,
                             n_units=n_units,
                             dropout_flag=dropout_flag,
                             p_dropout=p_dropout)

    model.summary()

    # compile model

    METRICS = [
        #tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc_roc", curve='ROC')
        #tf.keras.metrics.AUC(name="auc_pr", curve='PR')
    ]

    adam_optim = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=adam_optim,
                  metrics=METRICS)

    # fit model

    weights = {0: n_pos_train, 1: n_neg_train}

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_auc_roc',
                                                patience=patience,
                                                mode='max')
    # this check_point saves the model with the best performance on validation data
    check_point = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_auc_roc',
        verbose=1,
        save_best_only=True,
        mode='max',
        save_weights_only=False
    )

    # validation sample weight seems to be needed for tensorflow version
    # on gpu on server, but not for the version on mac desktop
    # since we are using AUC as the validation metric
    # it doesn't matter whether we use weight or not
    # so it doesn't matter if the weight does not equal the proportion in valid
    y2_valid_list = [y[0] for y in y2_valid]
    w_valid_list = [n_pos_valid if y == 0 else n_neg_valid for y in y2_valid_list]

    model.fit(x=[HLA_encoded_train, CDR3_encoded_train,
                 CDR3_len_train, cdr1_encoded_train,
                 cdr2_encoded_train, cdr25_encoded_train], y=y2_train,
              validation_data=([HLA_encoded_valid,
                                CDR3_encoded_valid, CDR3_len_valid, cdr1_encoded_valid,
                                cdr2_encoded_valid, cdr25_encoded_valid], y2_valid, np.array(w_valid_list)),
              class_weight=weights, callbacks=[callback, check_point],
              epochs=300, batch_size=32)

    end = time.time()
    print("Training the model took around "+str(round((end - start)/60, 1)) + " minutes.")
    print("Finished training.")
