import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import argparse
import time
import json
from datetime import datetime

# print("TensorFlow version: ", tf.__version__)
import mlflow
from keras.models import load_model

from src.utils.parsing import add_arguments
import src.utils.augmentation as preprocess
import src.experimental_stuff.custom_losses as my_custom_losses
import src.experimental_stuff.train_test as train_test
from src.utils.helpers import check_gpu
import src.experimental_stuff.autoencoder as autoencoder

parser = argparse.ArgumentParser()
args = add_arguments(parser)
print(args)
if args.run_name != "local":
    mlflow.start_run(run_name=args.run_name)
mlflow.log_params(vars(args))


def run():
    print("-----------------Start run-----------------")
    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    print("-----------------Check devices-----------------")
    gpu = check_gpu()

    print("-----------------Locate data-----------------")

    X_train_dir = "../data/cubic/non_defective/folds/fold{}/train/classes/nondefective/images".format(args.fold)
    X_val_dir = "../data/cubic/non_defective/folds/fold{}/test/classes/nondefective/images".format(args.fold)
    X_test_def_dir = "../data/cubic/defective/folds/fold{}/train/classes/defective/images".format(args.fold)
    X_test_nondef_dir = "../data/cubic/defective/folds/fold{}/train/classes/nondefective/images".format(args.fold)

    assert os.path.exists(X_train_dir)
    assert os.path.exists(X_val_dir)
    assert os.path.exists(X_test_def_dir)
    assert os.path.exists(X_test_nondef_dir)

    print("X_train_dir=", X_train_dir)
    print("X_val_dir=", X_val_dir)
    print("X_test_def_dir=", X_test_def_dir)
    print("X_test_def_dir=", X_test_nondef_dir)

    print("-----------------Load settings-----------------")
    print(vars(args))

    Losses = my_custom_losses.CustomLosses(beta=args.beta, ksize=args.ksize)
    losses = {
        "fourier": Losses.fourier_loss,
        "mse_fourier": Losses.mse_and_fourier,
        "mse": Losses.mean_squared_error,
        "spectral": Losses.spectral_loss,
        "mse_spectral": Losses.mse_and_spectral,
        "texture": Losses.texture_loss,
    }
    try:
        loss = losses[args.loss]
    except KeyError:
        print("Loss '{}' not implemented yet.".format(args.loss))
        exit()

    architectures = {
        # "fully": {autoencoder.fullyconnected},
        "contracting": autoencoder.ae_contracting(depth=args.depth, filters_init=args.filters, dense=args.dense),
        "expanding": autoencoder.ae_expanding(depth=args.depth, filters_init=args.filters, dense=args.dense)
    }
    try:
        architecture = architectures[args.architecture]

    except KeyError:
        print(
            "Architecture '{}' not implemented yet.".format(
                args.architecture,
            )
        )
        exit()

    # architecture = args.architecture
    epochs = args.epochs
    steps = args.steps
    batch_size = args.batch_size
    # learning_rate = args.learning_rate
    beta = args.beta
    ksize = args.ksize

    if args.train:
        print("-----------------Training-----------------")

        model_name = "{}ae_{}_d{}_e{}_spe{}_b{}_{}loss_{}beta_{}ksize".format(
            "TEM",
            args.architecture,
            str(args.depth),
            str(epochs),
            str(steps),
            str(batch_size),
            args.loss,
            str(beta),
            str(ksize),
        )

        output_dir = os.path.join("../output/", model_name, datetime_now)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(output_dir)

        # Save parameters
        model_parameter_path = os.path.join(
            output_dir, "parameterDict_{}.json".format(datetime_now)
        )
        with open(model_parameter_path, "w") as jsonFile:
            json.dump(vars(args), jsonFile)

        train_generator = preprocess.preprocess_generator(X_train_dir, batch_size=batch_size) # TODO: Why not replace this with keras generator?
        validation_generator = preprocess.preprocess_generator(X_val_dir, batch_size=batch_size)
        test_defective_generator = preprocess.preprocess_generator(X_test_def_dir, batch_size=batch_size)
        test_nondefective_generator = preprocess.preprocess_generator(X_test_nondef_dir, batch_size=batch_size)

        starttime = time.time()
        if args.variational_ae:
            if args.architecture == 'expanding':
                expanding = True
            elif args.architecture == 'contracting':
                expanding = False
            else:
                print("Not a valid architecture. Try again.")
                exit()

            model, history = train_test.train_vae(train_gen=train_generator, validation_gen=validation_generator,
                                                  test_def_gen=test_defective_generator,
                                                  test_nondef_gen=test_nondefective_generator, output_dir=output_dir,
                                                  loss=loss, depth=args.depth, epochs=epochs, steps_per_epoch=steps,
                                                  gpu=gpu, expanding=expanding, filters_init=args.filters)
        else:
            model, history = train_test.train_model(train_gen=train_generator, validation_gen=validation_generator,
                                                    test_def_gen=test_defective_generator,
                                                    test_nondef_gen=test_nondefective_generator, output_dir=output_dir,
                                                    architecture=architecture, epochs=epochs, steps_per_epoch=steps,
                                                    loss=loss, es_patience=args.patience, gpu=gpu)


        # Save model
        model_path = os.path.join(output_dir, "model_{}.h5".format(datetime_now))
        model.save(model_path)
        mlflow.log_artifact(model_path)

        # Save model history
        model_history_path = os.path.join(
            output_dir, "trainHistoryDict_{}.json".format(datetime_now)
        )
        with open(model_history_path, "w") as f:
            json.dump(str(history.history), f)
        mlflow.log_artifact(model_history_path)

        print("Model trained, saved to disk")
        print("Training finished in {}s".format(time.time() - starttime))
        print("t = {} sec".format(time.time() - starttime))

    else:
        print("-----------------No training-----------------")
        # args.model = "TEMae_expanding_e809_spe64_b32_lr0.001_mse_spectralloss_1000.0beta_9ksize"
        assert args.model is not None
        model_name = args.model
        model = load_model(
            "../models/" + model_name + ".h5",
            custom_objects={
                "mse_and_spectral": Losses.mse_and_spectral,
                "texture_loss": Losses.texture_loss,
            },
        )

    # if args.test:
    #     print("-----------------Testing-----------------")
    #     output_dir = "../output/" + model_name
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #
    #     test_defective_generator = preprocess.crop_generator(
    #         test_dir, crop_length=256, batch_size=40
    #     )
    #     train_test.test_model(
    #         model,
    #         test_defective_generator,
    #         output_dir,
    #         score_function="ssim",
    #         show_plot=show_images,
    #     )


run()
