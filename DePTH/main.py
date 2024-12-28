import os, sys, argparse

def parse_args():
    parser = argparse.ArgumentParser(
    description=\
    "DePTH: a neural network model for sequence-based TCR and HLA association prediction",
        prog='DePTH')
    subparser = parser.add_subparsers(help='whether to train, predict or do cross-validation', dest='command')

    # --------------------------------
    # Parser for training a model
    # this one saves a model out
    # --------------------------------

    train_parser = subparser.add_parser('train', help='train a DePTH model')

    train_parser.add_argument('--hla_class', required=True, type=str,
                        help='the class of HLA, either HLA_I or HLA_II',
                        choices=['HLA_I', 'HLA_II'])
    train_parser.add_argument('--data_dir', required=True, type=str,
                        help='directory where training and validation data files are located')
    train_parser.add_argument('--model_dir', required=True, type=str,
                        help='the path to the folder to hold the trained model, '\
                        'should not include ".." as part of the path')
    train_parser.add_argument('--enc_method', type=str, default='one_hot',
                        help='encoding method for amino acid, can be one of one_hot, blosum62, atchley, and pca',
                        choices=['one_hot', 'blosum62', 'atchley', 'pca'])
    train_parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate, for example, 0.0001')
    train_parser.add_argument('--n_dense', type=int, default=2,
                        help='number of dense layers, can be either 1 or 2',
                        choices=[1, 2])
    train_parser.add_argument('--n_units_str', type=str, default='[64,16]',
                        help='a list of sizes of the dense layers, must be in the format of a list '\
                             'with length matching n_dense (with only ",", no space between numbers in the list, '\
                             'if the list has length 2), for example, [64,16]')
    train_parser.add_argument('--dropout_flag', type=str, default='True',
                        help='whether to use dropout or not, can be either True or False',
                        choices=['True', 'False'])
    train_parser.add_argument('--p_dropout', type=float, default=0.2,
                        help='dropout probability, for example, 0.2')
    train_parser.add_argument('--rseed', type=int, default=1000,
                        help='random seed for random')
    train_parser.add_argument('--np_seed', type=int, default=1216,
                        help='random seed for numpy')
    train_parser.add_argument('--tf_seed', type=int, default=2207,
                        help='random seed for tensorflow')


    # --------------------------------
    # Parser for loading a trained model to make prediction
    # this one saves out a file of input pairs and predicted scores
    # --------------------------------

    predict_parser = subparser.add_parser('predict', help='load a trained DePTH model to make prediction')

    predict_parser.add_argument('--test_file', required=True, type=str,
                        help='path to the file of (TCR, HLA) pairs to make prediction for')
    predict_parser.add_argument('--hla_class', required=True, type=str,
                        help='the class of HLA, either HLA_I or HLA_II',
                        choices=['HLA_I', 'HLA_II'])
    predict_parser.add_argument('--output_dir', required=False, type=str, default=".",
                        help='path to the folder to put the file with predicted scores,'\
                             ' should not include ".." as part of the path')
    predict_parser.add_argument('--default_model', required=False, default='True',
                        help='What model to use. Three options: the default trained model v2.0 ("True"),' \
                             'the legacy default trained model v1.0 ("legacy"), or model trained by the user ("False").' \
                             'A model directory '\
                             'and the enc_method used for training the model must be provided if default_model is "False"',
                        choices=['True', 'False', 'legacy'])
    predict_parser.add_argument('--model_dir', required=False, type=str,
                        help='the path to the folder containing the trained model, '\
                             'should not be provided if default_model is "True"')
    predict_parser.add_argument('--enc_method', required=False, type=str,
                        help='encoding method for amino acid, can be one of one_hot, blosum62, atchley, and pca.'\
                        ' Must be consistent with the one used for training the model',
                        choices=['one_hot', 'blosum62', 'atchley', 'pca'])




    # --------------------------------
    # Parser for cross-validation for a given hyper parameter setting
    # this one does not save any model, but output a file of average validation auc roc
    # --------------------------------

    cv_parser = subparser.add_parser('cv', help='cross-validation for a specific hyperparameter setting')

    cv_parser.add_argument('--hla_class', required=True, type=str,
                        help='the class of HLA, either HLA_I or HLA_II',
                        choices=['HLA_I', 'HLA_II'])
    cv_parser.add_argument('--data_dir', required=True, type=str,
                        help='directory where training and validation data files are located')
    cv_parser.add_argument('--average_valid_dir', default=".", type=str,
                        help='path to the folder to put the file of average validation AUC, '\
                             'should not include parent ".." as part of the path')
    cv_parser.add_argument('--enc_method', type=str, default='one_hot',
                        help='encoding method for amino acid, can be one of one_hot, blosum62, atchley, and pca',
                        choices=['one_hot', 'blosum62', 'atchley', 'pca'])
    cv_parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    cv_parser.add_argument('--n_dense', type=int, default=2,
                        help='number of dense layers, can be either 1 or 2',
                        choices=[1, 2])
    cv_parser.add_argument('--n_units_str', type=str, default='[64,16]',
                        help='a list of sizes of the dense layers, must be in the format of a list '\
                             'with length matching n_dense (with only ",", no space between numbers '\
                             'in the list, if the list has length 2)')
    cv_parser.add_argument('--dropout_flag', type=str, default='True',
                        help='whether to use dropout or not, can be either True or False',
                        choices=['True', 'False'])
    cv_parser.add_argument('--p_dropout', type=float, default=0.2,
                        help='dropout probability')
    cv_parser.add_argument('--rseed', type=int, default=1000,
                        help='random seed for random')
    cv_parser.add_argument('--np_seed', type=int, default=1216,
                        help='random seed for numpy')
    cv_parser.add_argument('--tf_seed', type=int, default=2207,
                        help='random seed for tensorflow')


    args = parser.parse_args()
    print("User input arguments: ", args)
    return args


def main():
    args = parse_args()

    # training
    if "train" == args.command:

        if not os.path.exists(args.data_dir):
            sys.exit("The directory supposed to contain training and validation data files does not exist.")
        data_file_list = os.listdir(args.data_dir)

        for needed_file in ["train_pos.csv", "train_neg.csv", "valid_pos.csv", "valid_neg.csv"]:
            if needed_file not in data_file_list:
                sys.exit("File "+needed_file+" does not exist under data directory.")

        if not os.path.exists(args.model_dir):
            print("Creating model directory: %s" % args.model_dir)
            os.makedirs(args.model_dir, exist_ok=False)

        from DePTH import train

        del args.command
        print(args)
        train.train(**vars(args))

    ## prediction
    elif "predict" == args.command:

        if not os.path.exists(args.output_dir):
            print("Creating output directory: %s" % args.output_dir)
            os.makedirs(args.output_dir, exist_ok=False)

        if args.default_model in ["True", "legacy"]:
            if args.model_dir is not None:
                sys.exit("When the default model (or legacy default model) in the package is used, no model_dir should be passed.")
            if args.enc_method != None:
                if args.enc_method != "one_hot":
                    sys.exit("When the default model (or legacy default model) in the package is used, enc_method should be one_hot.")
        else:
            if not os.path.exists(args.model_dir):
                sys.exit("The folder containing trained model does not exist. Please provide one. This can be done by running 'DePTH train' first to produce it.")
            if args.enc_method is None:
                sys.exit("The enc_method is missing. Please provide the enc_method used for training the model.")


        from DePTH import predict

        del args.command
        print(args)
        predict.predict(**vars(args))

    # cross-validation
    elif "cv" == args.command:

        if not os.path.exists(args.data_dir):
            sys.exit("The directory supposed to contain training and validation data files does not exist.")
        data_file_list = os.listdir(args.data_dir)

        for needed_file in ["train_pos.csv", "train_neg.csv", "valid_pos.csv", "valid_neg.csv"]:
            if needed_file not in data_file_list:
                sys.exit("File "+needed_file+" does not exist under data directory.")

        if not os.path.exists(args.average_valid_dir):
            print("Creating average validation AUC directory: %s" % args.average_valid_dir)
            os.makedirs(args.average_valid_dir, exist_ok=False)

        from DePTH import cv

        del args.command
        print(args)
        cv.cv(**vars(args))

    else:
        sys.exit("The command input after 'DePTH' must be one of train, predict and cv.")
