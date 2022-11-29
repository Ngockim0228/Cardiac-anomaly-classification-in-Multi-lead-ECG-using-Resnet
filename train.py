from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
                                        CSVLogger, EarlyStopping)
from model import get_model
import argparse
from datasets import ECGSequence
import tensorflow as tf

# This project uses a 5% dataset only from doi: 10.5281/zenodo.4916206

k = 1
while k < 10:

    if __name__ == "__main__":
        # Get data and train
        parser = argparse.ArgumentParser(description='Train neural network.')


        parser.add_argument('--path_to_csv', type=str, default='data_train/exams.csv',
                            help='path to csv file containing annotations')
        parser.add_argument('--val_split', type=float, default=0.02,
                            help='number between 0 and 1 determining how much of'
                                 ' the data is to be used for validation. The remaining '
                                 'is used for validation. Default: 0.02')
        parser.add_argument('--dataset_name', type=str, default='tracings',
                            help='name of the hdf5 dataset containing tracings')
        args = parser.parse_args()
        # Optimization settings
        loss = 'binary_crossentropy'
        lr = 0.0001
        batch_size = 16
        opt = Adam(lr)
        callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       patience=7,
                                       min_lr=lr / 100),
                     EarlyStopping(patience=9,  # Patience should be larger than the one in ReduceLROnPlateau
                                   min_delta=0.00001)]

        train_seq, valid_seq = ECGSequence.get_train_and_val(
             args.dataset_name, args.path_to_csv, batch_size, args.val_split)

        model = get_model(train_seq.n_classes)

        model.compile(loss=loss, optimizer=opt)

        # Create log
        callbacks += [TensorBoard(log_dir='./logs', write_graph=False),
                      CSVLogger('training.log', append=False)]  # Change append to true if continuing training
        # Save the BEST and LAST model
        callbacks += [ModelCheckpoint('./backup_model_last.hdf5'),
                      ModelCheckpoint('./backup_model_best.hdf5', save_best_only=True)]
        # Train neural network

        history = model.fit(train_seq,
                            epochs=50, 
                            initial_epoch=0,  # If you are continuing a interrupted section change here
                            callbacks=callbacks,
                            validation_data=valid_seq,
                            verbose=1)

        k = k + 1
        # Save final result
        model.save("./model/other_seeds/model_"+ str(k) + ".hdf5")
        if k == 10:
            break

