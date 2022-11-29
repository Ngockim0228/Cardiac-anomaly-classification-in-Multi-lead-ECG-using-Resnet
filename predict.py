import numpy as np
import warnings
import argparse
warnings.filterwarnings("ignore")
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from datasets import ECGSequence_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get performance on test set from hdf5')

    parser.add_argument('--path_to_model', default='./model/other_seeds/model_1.hdf5', # or model_date_order.hdf5,
                        help='file containing training model.')
    parser.add_argument('--dataset_name', type=str, default='tracings',
                        help='name of the hdf5 dataset containing tracings')
    parser.add_argument('--output_file', default="./Resnet_predicts/other_seeds/model_1.npy",  # or predictions_date_order.csv
                        help='output csv file.')
    parser.add_argument('-bs', type=int, default=32,
                        help='Batch size.')

    args, unk = parser.parse_known_args()
    if unk:
        warnings.warn("Unknown arguments:" + str(unk) + ".")
    # Import data
    seq = ECGSequence_test( args.dataset_name, batch_size=args.bs)
    # Import model
    model = load_model(args.path_to_model, compile=False)
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    y_score = model.predict(seq,  verbose=1)

    # Generate dataframe
    np.save(args.output_file, y_score)

    print("Output predictions saved")
