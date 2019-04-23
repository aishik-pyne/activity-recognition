"""
Train our RNN on extracted features or images.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import LSTMModel
from dataset import VGGFeatureDataSet
import time
import os.path


def train_lstm(seq_length, saved_model=None, features_length=7680,
               batch_size=4, nb_epoch=100, run_number=1):
    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath=os.path.join('logs', 'checkpoints', 'lstm' + '-' + str(run_number) +
                              '.{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)

    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('logs', 'tb', 'lstm'))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=5)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('logs', 'csv', 'lstm' + '-' + 'training-' +
                                        str(timestamp) + '.log'))

    # Get the data and process it.
    train_data = VGGFeatureDataSet(
        seq_length=seq_length,
        mode='train'
    )
    test_data = VGGFeatureDataSet(
        seq_length=seq_length,
        mode='train'
    )

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(train_data.data) * 0.7) // batch_size

    generator = train_data.frame_generator(batch_size)
    val_generator = test_data.frame_generator(batch_size)

    # Get the model.
    rm = LSTMModel(nb_classes=len(train_data.classes), seq_length=seq_length,
                   saved_model=saved_model,
                   features_length=features_length)

    # Fit!
    rm.model.fit_generator(
        generator=generator,
        steps_per_epoch=steps_per_epoch,
        epochs=nb_epoch,
        verbose=1,
        callbacks=[tb, early_stopper, csv_logger, checkpointer],
        validation_data=val_generator,
        validation_steps=40,
        workers=4)


def main():
    """These are the main training settings. Set each before running
    this file."""
    # model can be one of lstm, lrcn, mlp, conv_3d, c3d
    # saved_model = 'logs/checkpoints/lstm-.010-0.884.hdf5'  # None or weights file
    saved_model = None  # None or weights file
    seq_length = 40
    batch_size = 4
    nb_epoch = 50
    run_number = 3
    train_lstm(seq_length, batch_size=batch_size,
               nb_epoch=nb_epoch, saved_model=saved_model, run_number=run_number)

if __name__ == '__main__':
    main()
