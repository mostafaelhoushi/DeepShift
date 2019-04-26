from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.keras.callbacks import Callback
import numpy as np

'''
    Fixing some keras callbacks to work with tf.train optimizers
'''


@tf_export('keras.callbacks.ReduceLROnPlateau')
class ReduceLROnPlateauMODIFIED(Callback):
    """Reduce learning rate when a metric has stopped improving.

    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Example:

    Arguments:
            monitor: quantity to be monitored.
            factor: factor by which the learning rate will
                    be reduced. new_lr = lr * factor
            patience: number of epochs with no improvement
                    after which learning rate will be reduced.
            verbose: int. 0: quiet, 1: update messages.
            mode: one of {auto, min, max}. In `min` mode,
                    lr will be reduced when the quantity
                    monitored has stopped decreasing; in `max`
                    mode it will be reduced when the quantity
                    monitored has stopped increasing; in `auto`
                    mode, the direction is automatically inferred
                    from the name of the monitored quantity.
            min_delta: threshold for measuring the new optimum,
                    to only focus on significant changes.
            cooldown: number of epochs to wait before resuming
                    normal operation after lr has been reduced.
            min_lr: lower bound on the learning rate.
    """

    def __init__(self,
                             monitor='val_loss',
                             factor=0.1,
                             patience=10,
                             verbose=0,
                             mode='auto',
                             min_delta=1e-4,
                             cooldown=0,
                             min_lr=0,
                             **kwargs):
        super(ReduceLROnPlateauMODIFIED, self).__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau ' 'does not support a factor >= 1.0.')
        if 'epsilon' in kwargs:
            min_delta = kwargs.pop('epsilon')
            logging.warning('`epsilon` argument is deprecated and '
                                            'will be removed, use `min_delta` instead.')
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0    # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            logging.warning('Learning Rate Plateau Reducing mode %s is unknown, '
                                            'fallback to auto mode.', self.mode)
            self.mode = 'auto'
        if (self.mode == 'min' or
                (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
#         print("DEBUG")
#         print(self.model.optimizer.optimizer._lr)
#         print(dir(self.model.optimizer.optimizer._lr))
#         print(K.get_value(self.model.optimizer.optimizer._lr))
        logs['lr'] = self.model.optimizer.optimizer._lr
        current = logs.get(self.monitor)
        if current is None:
            logging.warning('Reduce LR on plateau conditioned on metric `%s` '
                                            'which is not available. Available metrics are: %s',
                                            self.monitor, ','.join(list(logs.keys())))

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = float(self.model.optimizer.optimizer._lr)
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        self.model.optimizer.optimizer._lr = new_lr
                        if self.verbose > 0:
                            print('\nEpoch %05d: ReduceLROnPlateau reducing learning '
                                        'rate to %s.' % (epoch + 1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0


@tf_export('keras.callbacks.LearningRateScheduler')
class LearningRateSchedulerMODIFIED(Callback):
    """Learning rate scheduler.

    Arguments:
            schedule: a function that takes an epoch index as input
                    (integer, indexed from 0) and returns a new
                    learning rate as output (float).
            verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, schedule, verbose=0):
        super(LearningRateSchedulerMODIFIED, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer.optimizer, '_lr'):
            raise ValueError('Optimizer must have a "_lr" attribute.')
        try:    # new API
            lr = float(self.model.optimizer.optimizer._lr)
            lr = self.schedule(epoch, lr)
        except TypeError:    # Support for old API for backward compatibility
            lr = self.schedule(epoch)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                                             'should be float.')
        self.model.optimizer.optimizer._lr = lr
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler reducing learning '
                        'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = self.model.optimizer.optimizer._lr
