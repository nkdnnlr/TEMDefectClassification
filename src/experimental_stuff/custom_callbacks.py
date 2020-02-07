from keras.callbacks import Callback, BaseLogger
import mlflow


class LossHistory(Callback):
    """
    Keras Callback that allows logging on MLFlow after each epoch end
    """

    def on_epoch_end(self, epoch, logs=None):
        metrics = logs.keys()
        for metric in metrics:
            # print("metric: ", metric)
            mlflow.log_metric(metric, logs.get(metric), step=epoch)


class AdditionalValidationGenerators(Callback):
    def __init__(self, dict_validation_generator, verbose=0, steps=None, gpu=False):
        """
        Keras Callback that allows additional validation generators to be evaluated and logged.
        :param dict_validation_generator: {name1: generator1, name2: generator2, ...}
        :param verbose:  verbosity mode, 1 or 0
        :param steps: Total number of steps (batches of samples) to yield from `generator` and to be used when evaluating on the additional datasets
        """
        super(AdditionalValidationGenerators, self).__init__()
        self.gpu = gpu
        self.validation_generators = dict_validation_generator
        self.epoch = []
        self.history = {}
        self.verbose = verbose
        self.steps = steps


    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # evaluate on the additional validation sets
        for key, value in self.validation_generators.items():
            validation_generator_name = key
            validation_generator = value

            results = self.model.evaluate_generator(
                generator=validation_generator, steps=self.steps
            )

            for i, result in enumerate(results):
                if i == 0:
                    valuename = validation_generator_name + "_loss"
                else:
                    if self.gpu:
                        valuename = (
                            validation_generator_name + "_" + self.model.metrics[i - 1]
                        )
                    else:
                        valuename = (
                                validation_generator_name + "_" + self.model.metrics[i - 1].name
                        )
                print(valuename, result)
                self.history.setdefault(valuename, []).append(result)
                mlflow.log_metric(valuename, result, step=epoch)
                # print("Logged.")
