# -------------------
# take an input file of (TCR, HLA) pairs and make prediction
# -------------------

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf

import pkg_resources
#from importlib import resources

import argparse


from DePTH import _utils


def predict(test_file, hla_class, output_dir, default_model, model_dir=None, enc_method=None):

    input_args = locals()
    print("input args are", input_args)

    legacy_flag = (default_model == 'legacy')
    default_model = (default_model in ['True', 'legacy'])

    if default_model:
        enc_method = 'one_hot'
    # load pair list
    df_pair = pd.read_csv(test_file, header=0)

    pair_list = [(tcr, hla) for tcr, hla in \
                  zip(df_pair['tcr'].tolist(), df_pair['hla_allele'].tolist())]

    # get the elements for encoding
    (allele_dict, hla_len, HLA_enc, CDR3len_enc, CDR3_enc, cdr1_enc,
            cdr2_enc, cdr25_enc) = _utils.prepare_encoders(hla_class, enc_method)

    # get encoded pairs
    components_test = _utils.encode(pair_list, enc_method, allele_dict, hla_len, HLA_enc, CDR3len_enc, CDR3_enc,
                              cdr1_enc, cdr2_enc, cdr25_enc)

    HLA_encoded, CDR3_encoded, CDR3_len_encoded, cdr1_encoded, cdr2_encoded, cdr25_encoded = components_test

    print(HLA_encoded.shape)
    print(CDR3_encoded.shape)
    print(CDR3_len_encoded.shape)
    print(cdr1_encoded.shape)
    print(cdr2_encoded.shape)
    print(cdr25_encoded.shape)

    if default_model:

        print("Get average prediction scores from 20 models")

        seed_path = pkg_resources.resource_filename(__name__, 'data/ensemble_seeds_20.txt')
        print("seed file path is: ", seed_path)

        seeds_list = []

        with open(seed_path) as fp:

            Lines = fp.readlines()

            for line in Lines:

                line_split = line.split("\t")
                seed_1 = line_split[0]
                seed_2 = line_split[1]
                seed_3 = line_split[2].split("\n")[0]

                seeds_list += [[seed_1, seed_2, seed_3]]


        sum_yhat = np.zeros((len(pair_list), 1))


        for cur_seeds in seeds_list:

            tf.keras.backend.clear_session()
            if legacy_flag:
                cur_model_folder = hla_class+"_all_match/"+hla_class+"_all_match_model_"+"_".join(cur_seeds)
                cur_model_path = pkg_resources.resource_filename(__name__, 'data/trained_models_legacy/'+cur_model_folder)
            else:
                cur_model_folder = hla_class+"/"+"model_"+"_".join(cur_seeds)
                cur_model_path = pkg_resources.resource_filename(__name__, 'data/trained_models/'+cur_model_folder)                
            #with resources.path('DePTH.data', default_model_folder) as default_model_path:
            print("model path is: ", cur_model_path)
            cur_model = tf.keras.models.load_model(cur_model_path)
            cur_yhat = cur_model.predict(components_test)
            sum_yhat = np.add(sum_yhat, cur_yhat)

        yhat = np.divide(sum_yhat, len(seeds_list))

    else:

        print("Get prediction scores from one single model")
        print("model path is: ", model_dir)
        model = tf.keras.models.load_model(model_dir)
        yhat = model.predict(components_test)

    yhat_reshape = yhat.reshape(len(pair_list), )
    df_pair['score'] = yhat_reshape.tolist()

    df_pair.to_csv(output_dir + "/predicted_scores.csv", index=False)
