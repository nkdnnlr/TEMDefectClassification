import os
import sys
import json
from time import time
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse

# import mlflow
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import History

from src.utils.helpers import count_files, LossHistory

parser = argparse.ArgumentParser()
# Data structure arguments
parser.add_argument("-train_dir", "--train_dir", type=str, default="../data/all_data/6folds_128/fold0/train/",
                    help="Train directory. Contains subdirs separating images in classes.")
parser.add_argument("-val_dir", "--val_dir", type=str, default="../data/all_data/6folds_128/fold0/test/",
                    help="Validation directory. Contains subdirs separating images in classes.")
parser.add_argument("-output_dir", "--output_dir", type=str, default="output/test/",
                    help="Output directory")
parser.add_argument("-base", "--base_model", type=str, default="VGG16",
                    help="Keras Base model for transfer learning. So far tried 'VGG16', 'ResNet50'")
parser.add_argument("-weights", "--weights", type=str, default=None,
                    help="Weights for model. Either 'imagenet' or None.")
parser.add_argument("-name", "--name", type=str, default="CNN_VGG16", help="Name for mlflow logging.")
# Architecture arguments
parser.add_argument("-train_all", "--train_all", type=bool, default=False,
                    help="If true, train all parameters of all layers of network again.")
parser.add_argument("-fc", "--fc_layers", nargs='+', type=int, help="FC layers", )
parser.add_argument("-d", "--dropout", type=float, default=0.5, help="Dropout fraction", )
# Training arguments
parser.add_argument("-e", "--epochs", type=int, default=100, help="Epochs")
parser.add_argument("-b", "--batch_size", type=int, default=8, help="Batch size. Scales steps accordingly.")
parser.add_argument("-l", "--learning_rate", type=float, default=0.00001)

args = parser.parse_args()
print("Arguments:", args)
# mlflow.start_run(run_name=args.name)
# mlflow.log_params(vars(args))

datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
TRAIN_DIR = args.train_dir
VAL_DIR = args.val_dir
TRAIN_TOTAL = count_files(TRAIN_DIR)
VAL_TOTAL = count_files(VAL_DIR)
class_list = ["defective", "non_defective"]
HEIGHT = 128  # Height of image. Optimally, for ResNet50, this should be equal to 224
WIDTH = 128  # Width of image. Optimally, for ResNet50, this should be equal to 224
FC_LAYERS = args.fc_layers
FC_LAYERS = [1024, 1024, 512]
DROPOUT = args.dropout
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs
OUTPUT_DIR = os.path.join(args.output_dir, datetime_now)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Total files in TRAIN_DIR: {}".format(TRAIN_TOTAL))
print("Total files in VAL_DIR: {}".format(VAL_TOTAL))

# Load Base model with weights. Skip top layer

base_models = {"VGG16": VGG16,
               "ResNet50": ResNet50}

base_model = base_models[args.base_model](
    weights=args.weights, include_top=False, input_shape=(HEIGHT, WIDTH, 3)
)

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE
)



def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    """
    Build upon last layer of base_model. This allows finetuning to our data set.
    :param base_model:
    :param dropout:
    :param fc_layers:
    :param num_classes:
    :return:
    """
    for layer in base_model.layers:
        layer.trainable = args.train_all

    x = base_model.output
    x = Flatten()(x)
    if fc_layers is not None:
        for fc in fc_layers:
            # New FC layer, random init
            x = Dense(fc, activation="relu")(x)
            if dropout is not None:
                x = Dropout(dropout)(x)

    # New softmax layer
    predictions = Dense(num_classes, activation="softmax")(x)

    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model


finetune_model = build_finetune_model(
    base_model, dropout=DROPOUT, fc_layers=FC_LAYERS, num_classes=len(class_list)
)

# finetune_model = simple_cnn(num_classes=len(class_list))

# In[4]
adam = Adam(lr=args.learning_rate)
finetune_model.compile(adam, loss="categorical_crossentropy", metrics=["accuracy"])
print(finetune_model.summary())

# exit()

# filepath = "./checkpoints/" + "ResNet50" + "_model_weights.h5"
# checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode="max")
# callbacks_list = [checkpoint]

loss_history = LossHistory()
# additional = AdditionalValidationGenerators({"test_defective": test_def_gen,
#                                              "test_nondefective": test_nondef_gen,
#                                              "train_m": train_gen}, verbose=2, steps=1, gpu=gpu)
# early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=50, min_delta=0.1)

callbacks = []

history_ = History()
callbacks.append(history_)
# callbacks.append(loss_history)
# callbacks.append(additional)

# callbacks_list = []
starttime = time()
history = finetune_model.fit_generator(
    train_generator,
    epochs=NUM_EPOCHS,
    workers=8,
    steps_per_epoch=TRAIN_TOTAL // BATCH_SIZE,
    shuffle=True,
    validation_data=val_generator,
    validation_steps=VAL_TOTAL // (BATCH_SIZE),
    callbacks=callbacks,
    verbose=2
)

print(history_)
print(history)
# Save model
model_path = os.path.join(OUTPUT_DIR, "model.h5")
finetune_model.save(model_path)
print("Saved model under {}".format(model_path))

# Save model history
model_history_path = os.path.join(
    OUTPUT_DIR, "trainHistoryDict.json")

with open(model_history_path, "w") as f:
    json.dump(str(history.history), f)
print("Saved history under {}".format(model_history_path))
# mlflow.log_artifact(model_history_path)
print("Logged history.")

print("Training time: {}s".format(time()-starttime))

print("Done.")

