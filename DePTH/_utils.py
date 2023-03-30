
import sys

import numpy as np
import pandas as pd

import random

import pkg_resources
#from importlib import resources

from collections import defaultdict

from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.activations import relu
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Reshape, Dropout, concatenate

from tensorflow.keras.callbacks import Callback




def prepare_encoders(hla_class, enc_method):

    # load HLA pseudo sequence information for encoding
    if hla_class == "HLA_I":
        pseudo_sequence_file = "data/for_encoders/HLA_I_pseudo_40.csv"
    else:
        pseudo_sequence_file = "data/for_encoders/HLA_II_pseudo_45.csv"

    pseudo_seq_stream = pkg_resources.resource_stream(__name__, pseudo_sequence_file)
    HLA_pseudo = pd.read_csv(pseudo_seq_stream, sep=',', header=0)
    #with resources.path("DePTH.data", pseudo_sequence_file) as pseudo_seq_stream:
    #    HLA_pseudo = pd.read_csv(pseudo_seq_stream)

    # get the length of hla pseudo sequence
    hla_len = len(list(HLA_pseudo.seq.to_list()[0]))

    # build a dict for HLA allele aa info
    allele_dict = defaultdict(list)
    for cur, seq in zip(HLA_pseudo.hla.to_list(), HLA_pseudo.seq.to_list()):
        allele_dict[cur] = list(seq)

    # hard encode sequence length limit for Glazer 2022 mcpas data
    unique_TCR_aa_lens = list(range(1, 28))
    CDR3len_enc_template = \
        np.array(unique_TCR_aa_lens).reshape(len(unique_TCR_aa_lens), 1)
    CDR3len_enc = OneHotEncoder().fit(CDR3len_enc_template)

    # build encoder for amino acids in HLA and CDR3
    # one_hot_encoding first
    if enc_method == "one_hot":
        AA_SYMOLS = ['A', 'R', 'N', 'D', 'C',
                     'Q', 'E', 'G', 'H', 'I',
                     'L', 'K', 'M', 'F', 'P',
                     'S', 'T', 'W', 'Y', 'V']
        AA_SYMOLS.sort()
        # encoder for HLA
        # pseudo sequences in the file for HLA_I alleles
        # "X" for some HLA-I, so add 'X' to the encoding template
        AA_SYMOLS_X = AA_SYMOLS + ['X']
        encode_template = np.array([[aa] * hla_len for aa in AA_SYMOLS_X])
        HLA_enc = OneHotEncoder().fit(encode_template)
        # now move on to encode the v gene and CDR3 part
        # add '.' to the aa name list for the purpose of padding short CDR3 sequences
        # encoder for CDR3
        # need to use max length of CDR3 aa seq in current file
        AA_SYMOLS_pad = AA_SYMOLS + ['.']
        CDR3_enc_template = np.array([[aa] * 27 for aa in AA_SYMOLS_pad])
        CDR3_enc = OneHotEncoder().fit(CDR3_enc_template)
        # encoders for cdr1, cdr2, cdr2.5
        # due to the fact that the cdr1, cdr2 sequence contains '*', the CDR3_enc below is
        # adjusted to add a '*' option to make it work
        # for cdr1, cdr2, cdr25 as weill
        AA_SYMOLS_pad_star = AA_SYMOLS + ['.'] + ['*']
        cdr1_template = np.array([[aa] * 12 for aa in AA_SYMOLS_pad_star])
        cdr1_enc = OneHotEncoder().fit(cdr1_template)
        cdr2_template = np.array([[aa] * 10 for aa in AA_SYMOLS_pad_star])
        cdr2_enc = OneHotEncoder().fit(cdr2_template)
        cdr25_template = np.array([[aa] * 6 for aa in AA_SYMOLS_pad])
        cdr25_enc = OneHotEncoder().fit(cdr25_template)
    # or blosum62
    elif enc_method == "blosum62":
        # define encoder for HLA
        def HLA_enc(hla_list):
            # pseudo sequences in the file for HLA_I alleles
            # "X" for some HLA-I, use the blosum62 matrix with "X"
            blosum62_X_file = 'data/for_encoders/blosum62_X.csv'
            blosum62_X_stream = pkg_resources.resource_stream(__name__, blosum62_X_file)
            #with resources.path("DePTH.data", blosum62_X_file) as blosum62_X_stream:
            blosum62_X_matrix = pd.read_csv(blosum62_X_stream, sep=',', header=0)

            blosum62_X_array = np.array(blosum62_X_matrix)
            blosum62_X_dict = defaultdict(list)
            for i in range(21):
                blosum62_X_dict[blosum62_X_array[i][0]] = blosum62_X_array[i][1:].tolist()
            return [[blosum62_X_dict[aa] for aa in hla] for hla in hla_list]

        # define encoder for CDR3
        # this encoder also works for cdr25
        # it does not work for cdr1, cdr2 due to that some sequence in cdr1, cdr2 contains
        # a character "*"
        def cdr_enc(cdr_list):

            blosum62_file = 'data/for_encoders/blosum62.csv'
            blosum62_stream = pkg_resources.resource_stream(__name__, blosum62_file)
            #with resources.path("DePTH.data", blosum62_file) as blosum62_stream:
            blosum62_matrix = pd.read_csv(blosum62_stream, sep=',', header=0)

            blosum62_array = np.array(blosum62_matrix)
            blosum62_dict = defaultdict(list)
            for i in range(20):
                blosum62_dict[blosum62_array[i][0]] = \
                    blosum62_array[i][1:].tolist() + [-4]
            blosum62_dict['.'] = [-4 for _ in range(20)] + [1]
            return [[blosum62_dict[aa] for aa in cdr] for cdr in cdr_list]

        # define an encoder specifically for cdr1 and cdr2
        # since some sequence in cdr1, cdr2 contains a character "*"
        def cdr2_enc(cdr2_list):

            blosum62_file = 'data/for_encoders/blosum62.csv'
            blosum62_stream = pkg_resources.resource_stream(__name__, blosum62_file)
            #with resources.path("DePTH.data", blosum62_file) as blosum62_stream:
            blosum62_matrix = pd.read_csv(blosum62_stream, sep=',', header=0)

            blosum62_array = np.array(blosum62_matrix)
            blosum62_dict = defaultdict(list)
            for i in range(20):
                blosum62_dict[blosum62_array[i][0]] = \
                    blosum62_array[i][1:].tolist() + [-4]
            blosum62_dict['.'] = [-4 for _ in range(20)] + [1]
            blosum62_dict['*'] = [-4 for _ in range(20)] + [1]
            return [[blosum62_dict[aa] for aa in cdr2] for cdr2 in cdr2_list]
    # or atchley
    elif enc_method == "atchley":
        # define encoder for HLA
        def HLA_enc(hla_list):
            # pseudo sequences in the file for HLA_I alleles
            # "X" for some HLA-I, so add 'X' to the encoding template
            atchley_file = 'data/for_encoders/Atchley_factors.csv'
            atchley_stream = pkg_resources.resource_stream(__name__, atchley_file)
            #with resources.path("DePTH.data", atchley_file) as atchley_stream:
            atchley_matrix = pd.read_csv(atchley_stream, sep=',', header=0)

            atchley_array = np.array(atchley_matrix)
            atchley_dict = defaultdict(list)
            for i in range(20):
                atchley_dict[atchley_array[i][0]] = atchley_array[i][1:].tolist()
            atchley_dict['X'] = [0 for _ in range(5)]
            return [[atchley_dict[aa] for aa in hla] for hla in hla_list]

        # define encoder for CDR3
        # this encoder also works for cdr25
        # it does not work for cdr1, cdr2 due to that some sequence in cdr1, cdr2 contains
        # a charater "*"
        def cdr_enc(cdr_list):

            atchley_file = 'data/for_encoders/Atchley_factors.csv'
            atchley_stream = pkg_resources.resource_stream(__name__, atchley_file)
            #with resources.path("DePTH.data", atchley_file) as atchley_stream:
            atchley_matrix = pd.read_csv(atchley_stream, sep=',', header=0)

            atchley_array = np.array(atchley_matrix)
            atchley_dict = defaultdict(list)
            for i in range(20):
                atchley_dict[atchley_array[i][0]] = atchley_array[i][1:].tolist()
            atchley_dict['.'] = [0 for _ in range(5)]
            return [[atchley_dict[aa] for aa in cdr] for cdr in cdr_list]

        # define an encoder specifically for cdr1 and cdr2
        # since some sequence in cdr1, cdr2 contains a character "*"
        def cdr2_enc(cdr2_list):

            atchley_file = 'data/for_encoders/Atchley_factors.csv'
            atchley_stream = pkg_resources.resource_stream(__name__, atchley_file)
            #with resources.path("DePTH.data", atchley_file) as atchley_stream:
            atchley_matrix = pd.read_csv(atchley_stream, sep=',', header=0)

            atchley_array = np.array(atchley_matrix)
            atchley_dict = defaultdict(list)
            for i in range(20):
                atchley_dict[atchley_array[i][0]] = atchley_array[i][1:].tolist()
            atchley_dict['.'] = [0 for _ in range(5)]
            atchley_dict['*'] = [0 for _ in range(5)]
            return [[atchley_dict[aa] for aa in cdr2] for cdr2 in cdr2_list]
    # AAidx_PCA
    else:
        # define encoder for HLA
        def HLA_enc(hla_list):
            # pseudo sequences in the file for HLA_I alleles
            # "X" for some HLA-I, so add 'X' to the encoding template
            pca_file = 'data/for_encoders/AAidx_PCA.csv'
            pca_stream = pkg_resources.resource_stream(__name__, pca_file)
            #with resources.path("DePTH.data", pca_file) as pca_stream:
            pca_matrix = pd.read_csv(pca_stream, sep=',', header=0)

            pca_array = np.array(pca_matrix)
            pca_dict = defaultdict(list)
            for i in range(20):
                pca_dict[pca_array[i][0]] = pca_array[i][1:].tolist()
            pca_dict['X'] = [0 for _ in range(15)]
            return [[pca_dict[aa] for aa in hla] for hla in hla_list]

        # define encoder for CDR3
        # this encoder also works for cdr25
        # it does not work for cdr1, cdr2 due to that some sequence in cdr1, cdr2 contains
        # a character "*"
        def cdr_enc(cdr_list):

            pca_file = 'data/for_encoders/AAidx_PCA.csv'
            pca_stream = pkg_resources.resource_stream(__name__, pca_file)
            #with resources.path("DePTH.data", pca_file) as pca_stream:
            pca_matrix = pd.read_csv(pca_stream, sep=',', header=0)

            pca_array = np.array(pca_matrix)
            pca_dict = defaultdict(list)
            for i in range(20):
                pca_dict[pca_array[i][0]] = pca_array[i][1:].tolist()
            pca_dict['.'] = [0 for _ in range(15)]
            return [[pca_dict[aa] for aa in cdr] for cdr in cdr_list]

        # define an encoder specifically for cdr1, cdr2
        # since some sequence in cdr1, cdr2 contains a charater "*"
        def cdr2_enc(cdr2_list):

            pca_file = 'data/for_encoders/AAidx_PCA.csv'
            pca_stream = pkg_resources.resource_stream(__name__, pca_file)
            #with resources.path("DePTH.data", pca_file) as pca_stream:
            pca_matrix = pd.read_csv(pca_stream, sep=',', header=0)

            pca_array = np.array(pca_matrix)
            pca_dict = defaultdict(list)
            for i in range(20):
                pca_dict[pca_array[i][0]] = pca_array[i][1:].tolist()
            pca_dict['.'] = [0 for _ in range(15)]
            pca_dict['*'] = [0 for _ in range(15)]
            return [[pca_dict[aa] for aa in cdr2] for cdr2 in cdr2_list]


    if enc_method == "one_hot":
        return (allele_dict, hla_len, HLA_enc, CDR3len_enc, CDR3_enc, cdr1_enc,
                cdr2_enc, cdr25_enc)
    else:
        return (allele_dict, hla_len, HLA_enc, CDR3len_enc, cdr_enc, cdr2_enc,
                cdr2_enc, cdr_enc)


# define padding function for later use
def pad_cdr3(cdr3, max_length, pad_letter):
    if len(cdr3) == max_length:
        return cdr3
    else:
        gap_start = min(6, 3 + (len(cdr3) - 5) // 2)
        return cdr3[:gap_start] + [pad_letter for _ in range(max_length - len(cdr3))] + cdr3[gap_start:]


def encode(pairs_ori, enc_method, allele_dict, hla_len, HLA_enc, CDR3len_enc, CDR3_enc,
            cdr1_enc, cdr2_enc, cdr25_enc):

    # load info for cdr1, cdr2, cdr25 encoding
    V_info_file = 'data/for_encoders/combo_xcr.tsv'
    V_stream = pkg_resources.resource_stream(__name__, V_info_file)
    #with resources.path("DePTH.data", V_info_file) as V_stream:
    V_info = pd.read_csv(V_stream, sep='\t')

    V_sub_info = V_info.loc[(V_info.organism == 'human')
                            & (V_info.chain == 'B')
                            & (V_info.region == 'V')]
    cdr1_dict = defaultdict(str)
    cdr2_dict = defaultdict(str)
    cdr25_dict = defaultdict(str)

    for allele, cdrs in zip(V_sub_info.id.tolist(), V_sub_info.cdrs.tolist()):
        cdr_seqs = cdrs.split(";")
        cdr1_dict[allele] = cdr_seqs[0]
        cdr2_dict[allele] = cdr_seqs[1]
        cdr25_dict[allele] = cdr_seqs[2]

    # add the 'not_found' key
    cdr1_dict['not_found'] = ''.join(['.' for _ in range(12)])
    cdr2_dict['not_found'] = ''.join(['.' for _ in range(10)])
    cdr25_dict['not_found'] = ''.join(['.' for _ in range(6)])

    if len(set([name for _, name in pairs_ori]) - set(allele_dict.keys())) > 0:
        sys.exit("Some pairs have HLA not satisfying the format requirements. "\
                 "For hla_class='HLA_I', the HLAs should be in table HLA_I_pseudo_40.csv."\
                 "For hla_class='HLA_II', the HLAs should be in table HLA_II_pseudo_45.csv.")
    # encode
    # HLA
    HLA_aa_list = [allele_dict[name] for _, name in pairs_ori]
    # CDR3
    CDR3_list = [pad_cdr3(list(tcr.split(",")[1]), 27, '.')
                 for tcr, _ in pairs_ori]
    # cdr1, cdr2, cdr25
    V_allele_trans = [tcr.split(",")[0] for tcr, _ in pairs_ori]

    if len(set(V_allele_trans) - set(cdr1_dict.keys())) > 0:
        sys.exit("Some pairs have V allele not in the subset of combo_xcr.tsv "\
                 "(with organism=='human', chain=='B', region=='V')"\
                 " and not being 'not_found'. "\
                 "Please make sure that all V alleles follow the format requirements.")

    cdr1_list = [list(cdr1_dict[trans]) for trans in V_allele_trans]
    cdr2_list = [list(cdr2_dict[trans]) for trans in V_allele_trans]
    cdr25_list = [list(cdr25_dict[trans]) for trans in V_allele_trans]
    # length of cdr3
    CDR3_len_list = [[len(tcr.split(",")[1])] for tcr, _ in pairs_ori]
    CDR3_len = np.array(CDR3_len_list)
    CDR3len_encoded = CDR3len_enc.transform(CDR3_len).toarray()
    # encoding for amino acis in HLA and CDR3
    if enc_method == "one_hot":
        # HLA in negative
        HLA_aa_array = np.array(HLA_aa_list)
        HLA_encoded = HLA_enc.transform(HLA_aa_array).toarray()
        HLA_encoded = HLA_encoded.reshape(len(pairs_ori), hla_len, 21)
        # CDR3 in negative
        CDR3_array = np.array(CDR3_list)
        CDR3_encoded = CDR3_enc.transform(CDR3_array).toarray()
        CDR3_encoded = CDR3_encoded.reshape(len(pairs_ori), 27, 21)
        # cdr1, cdr2, cdr25 in negative
        cdr1_array = np.array(cdr1_list)
        cdr1_encoded = cdr1_enc.transform(cdr1_array).toarray()
        cdr1_encoded = cdr1_encoded.reshape(len(pairs_ori), 12, 22)
        cdr2_array = np.array(cdr2_list)
        cdr2_encoded = cdr2_enc.transform(cdr2_array).toarray()
        cdr2_encoded = cdr2_encoded.reshape(len(pairs_ori), 10, 22)
        cdr25_array = np.array(cdr25_list)
        cdr25_encoded = cdr25_enc.transform(cdr25_array).toarray()
        cdr25_encoded = cdr25_encoded.reshape(len(pairs_ori), 6, 21)
    # blosum62, atchley, pca
    else:
        HLA_encoded = np.array(HLA_enc(HLA_aa_list))
        CDR3_encoded = np.array(CDR3_enc(CDR3_list))
        cdr1_encoded = np.array(cdr1_enc(cdr1_list))
        cdr2_encoded = np.array(cdr2_enc(cdr2_list))
        cdr25_encoded = np.array(cdr25_enc(cdr25_list))

    # return encoded data
    return HLA_encoded, CDR3_encoded, CDR3len_encoded, \
           cdr1_encoded, cdr2_encoded, cdr25_encoded


def get_data(hla_class, data_dir, enc_method, cv_flag):

    # extract training/validation positive/negative pairs
    df_train_pos = pd.read_csv(data_dir + "/train_pos.csv", header=0)
    df_train_neg = pd.read_csv(data_dir + "/train_neg.csv", header=0)
    df_valid_pos = pd.read_csv(data_dir + "/valid_pos.csv", header=0)
    df_valid_neg = pd.read_csv(data_dir + "/valid_neg.csv", header=0)

    train_pos = [(tcr, hla) for tcr, hla in zip(df_train_pos.tcr.to_list(), df_train_pos.hla_allele.to_list())]
    train_neg = [(tcr, hla) for tcr, hla in zip(df_train_neg.tcr.to_list(), df_train_neg.hla_allele.to_list())]
    valid_pos = [(tcr, hla) for tcr, hla in zip(df_valid_pos.tcr.to_list(), df_valid_pos.hla_allele.to_list())]
    valid_neg = [(tcr, hla) for tcr, hla in zip(df_valid_neg.tcr.to_list(), df_valid_neg.hla_allele.to_list())]

    n_pos_train = len(train_pos)
    n_neg_train = len(train_neg)
    n_pos_valid = len(valid_pos)
    n_neg_valid = len(valid_neg)

    if cv_flag:
        # combine train and validation positive tcrs together, and negative tcrs together
        # for providing cross-validation pairs
        pos_combined = train_pos + valid_pos
        neg_combined = train_neg + valid_neg
        # shuffle
        random.shuffle(pos_combined)
        random.shuffle(neg_combined)
        # split into train and validation
        train_pos_cv = pos_combined[:n_pos_train]
        train_neg_cv = neg_combined[:n_neg_train]
        valid_pos_cv = pos_combined[n_pos_train:]
        valid_neg_cv = neg_combined[n_neg_train:]
        # combine into train and validation
        train_combined = train_pos_cv + train_neg_cv
        valid_combined = valid_pos_cv + valid_neg_cv
        # create labels corresponding to the pos and neg pairs
        train_y = np.array([1 for _ in range(len(train_pos_cv))] +
                           [0 for _ in range(len(train_neg_cv))]).reshape(len(train_pos_cv) + len(train_neg_cv), 1)
        valid_y = np.array([1 for _ in range(len(valid_pos_cv))] +
                           [0 for _ in range(len(valid_neg_cv))]).reshape(len(valid_pos_cv) + len(valid_neg_cv), 1)

    else:
        # combine pos and neg tcrs together, for training/validation separately
        train_combined = train_pos + train_neg
        valid_combined = valid_pos + valid_neg

        # create labels corresponding to the pos and neg pairs
        train_y = np.array([1 for _ in range(len(train_pos))] +
                           [0 for _ in range(len(train_neg))]).reshape(len(train_pos) + len(train_neg), 1)
        valid_y = np.array([1 for _ in range(len(valid_pos))] +
                           [0 for _ in range(len(valid_neg))]).reshape(len(valid_pos) + len(valid_neg), 1)

    # shuffle pairs for training/validation separately
    # use the indexes for shuffling to make the pair and label consistent
    train_ind_list = list(range(len(train_combined)))
    valid_ind_list = list(range(len(valid_combined)))
    random.shuffle(train_ind_list)
    random.shuffle(valid_ind_list)
    # get shuffled pairs
    train_ori = [train_combined[i] for i in train_ind_list]
    valid_ori = [valid_combined[i] for i in valid_ind_list]
    # get shuffled labels
    train_y = train_y[train_ind_list]
    valid_y = valid_y[valid_ind_list]

    # get the elements for encoding
    (allele_dict, hla_len, HLA_enc, CDR3len_enc, CDR3_enc, cdr1_enc,
            cdr2_enc, cdr25_enc) = prepare_encoders(hla_class, enc_method)

    # get encoded pairs
    components_train = encode(train_ori, enc_method, allele_dict, hla_len, HLA_enc, CDR3len_enc, CDR3_enc,
                               cdr1_enc, cdr2_enc, cdr25_enc)
    components_valid = encode(valid_ori, enc_method, allele_dict, hla_len, HLA_enc, CDR3len_enc, CDR3_enc,
                               cdr1_enc, cdr2_enc, cdr25_enc)

    return ((components_train, train_y, n_pos_train, n_neg_train),
            (components_valid, valid_y, n_pos_valid, n_neg_valid))



def get_model(HLA_shape, CDR3_shape, len_shape, cdr1_shape, cdr2_shape, cdr25_shape,
              n_dense, n_units, dropout_flag, p_dropout):

    # Define input layers
    HLA_input = Input(HLA_shape, name="HLA")
    HLA_reshape = Reshape((HLA_shape[0] * HLA_shape[1],),
                          input_shape=HLA_shape, name="HLA_reshape")(HLA_input)
    CDR3_input = Input(CDR3_shape, name="CDR3")
    len_input = Input(len_shape, name="CDR3_len")
    cdr1_input = Input(cdr1_shape, name="CDR1")
    cdr2_input = Input(cdr2_shape, name="CDR2")
    cdr25_input = Input(cdr25_shape, name="CDR2.5")
    cdr1_reshape = Reshape((cdr1_shape[0] * cdr1_shape[1],),
                           input_shape=cdr1_shape, name="CDR1_reshape")(cdr1_input)
    cdr2_reshape = Reshape((cdr2_shape[0] * cdr2_shape[1],),
                           input_shape=cdr2_shape, name="CDR2_reshape")(cdr2_input)
    cdr25_reshape = Reshape((cdr25_shape[0] * cdr25_shape[1],),
                            input_shape=cdr25_shape, name="CDR2.5_reshape")(cdr25_input)

    # construct CDR3_branches
    CDR3_branch = Conv1D(filters=8, kernel_size=2, activation=relu,
                         input_shape=CDR3_shape, name='CDR3_conv')(CDR3_input)
    CDR3_branch = MaxPooling1D(pool_size=2, strides=1, padding='valid',
                               name='CDR3_maxpooling')(CDR3_branch)
    CDR3_flatten = Flatten(name='CDR3_flatten')(CDR3_branch)
    CDR3_reshape = Reshape((CDR3_shape[0] * CDR3_shape[1],),
                           input_shape=CDR3_shape, name="CDR3_reshape")(CDR3_input)
    CDR3_inter_layer = concatenate([CDR3_flatten, CDR3_reshape], axis=-1)

    # concatenate parts together
    HLA_part = Dense(64, activation=relu, name="HLA_dense")(HLA_reshape)

    TCR_combined = concatenate([len_input, CDR3_inter_layer,
                                cdr1_reshape, cdr2_reshape, cdr25_reshape])
    TCR_part = Dense(64, activation=relu, name="TCR_dense")(TCR_combined)

    inter_layer = concatenate([HLA_part, TCR_part])

    # move on to see how many dense layers we want
    # and whether we want a dropout layer
    if n_dense == 1:
        if not dropout_flag:
            last_layer = Dense(n_units[0], activation=relu, name="pair_dense")(inter_layer)
        else:
            dense_layer = Dense(n_units[0], activation=relu, name="pair_dense")(inter_layer)
            last_layer = Dropout(p_dropout)(dense_layer)
    else:
        if not dropout_flag:
            first_dense = Dense(n_units[0], activation=relu, name="pair_dense1")(inter_layer)
            last_layer = Dense(n_units[1], activation=relu, name="pair_dense2")(first_dense)
        else:
            first_dense = Dense(n_units[0], activation=relu, name="pair_dense1")(inter_layer)
            dropout_layer = Dropout(p_dropout)(first_dense)
            last_layer = Dense(n_units[1], activation=relu, name="pair_dense2")(dropout_layer)
    # final output layer
    output = Dense(1, activation='sigmoid', name='output')(last_layer)
    # build the model
    model = Model(inputs=[HLA_input, CDR3_input, len_input,
                          cdr1_input, cdr2_input, cdr25_input], outputs=output)
    return model


class SaveBestWeights(Callback):
    def __init__(self, save_best_metric='val_auc_roc'):
        self.save_best_metric = save_best_metric
        self.best = float('-inf')

    def on_epoch_end(self, epoch, logs=None):
        metric_value = logs[self.save_best_metric]
        if metric_value > self.best:
            self.best = metric_value
            self.best_weights = self.model.get_weights()
