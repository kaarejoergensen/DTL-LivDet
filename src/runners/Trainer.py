import logging
import time

import tensorflow as tf
import tensorflow.keras.backend as K

from Dataset import Dataset
from Loss import leaf_l1_loss
from runners.RunnerBase import RunnerBase


class Trainer(RunnerBase):
    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger("main")

    def train(self):
        config = self.config
        epochs = config.args.epochs
        types = config.args.training_types
        if not config.args.dont_validate:
            self.logger.info("Training with validation")
            while True:
                for val_type in types:
                    types_to_load = [t for t in types if t != val_type]
                    self.logger.info("Training with types {} and validation type {}".format(types_to_load, val_type))
                    dataset = Dataset(config, types_to_load, val_type)
                    epochs_train = int((epochs + epochs % len(types)) / len(types))
                    self._train(dataset, epochs_train)
                    self.last_epoch += epochs_train
                if not config.args.keep_running:
                    break
        else:
            self.logger.info("Training without validation, using types {}".format(types))
            dataset = Dataset(config, types)
            self._train(dataset, epochs)

    def _train(self, dataset, epochs):
        config = self.config
        step_per_epoch = config.args.steps
        step_per_epoch_val = config.args.steps_val
        if dataset.feed_val:
            self.logger.info("Training for {} epochs, with {} steps per epoch and {} steps per epoch for validation"
                             .format(epochs, step_per_epoch, step_per_epoch_val))
        else:
            self.logger.info("Training for {} epochs, with {} steps per epoch".format(epochs, step_per_epoch))

        # data stream
        global_step = self.last_epoch * step_per_epoch
        for epoch in range(0, epochs):
            start = time.time()
            # define the
            self.dtn_op = tf.compat.v1.train.AdamOptimizer(config.args.lr, beta1=0.5)
            ''' train phase'''
            for step in range(step_per_epoch):
                # for data_batch in it:
                class_loss, route_loss, uniq_loss, spoof_counts, eigenvalue, trace, _to_plot = \
                    self._train_one_step(next(dataset.feed), global_step, True)

                global_step += 1
                if not config.args.log_less or (step + 1) % (int(step_per_epoch / 10)) == 0:
                    self.logger.info(
                        'Epoch {:d}-{:d}/{:d}: Cls:{:.3g}, Route:{:.3g}({:3.3f}, {:3.3f}), Uniq:{:.3g}, '
                        'Counts:[{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d}]     '.
                            format(self.last_epoch + epoch + 1, step + 1, step_per_epoch,
                                   self.class_loss(class_loss),
                                   self.route_loss(route_loss), eigenvalue, trace,
                                   self.uniq_loss(uniq_loss),
                                   spoof_counts[0], spoof_counts[1], spoof_counts[2], spoof_counts[3],
                                   spoof_counts[4], spoof_counts[5], spoof_counts[6], spoof_counts[7]))
                # plot the figure
                if config.args.plot:
                    if (step + 1) % 400 == 0:
                        fname = config.args.logging_path + '/epoch-' + str(epoch + 1) + '-train-' + str(
                            step + 1) + '.png'
                        super().plot_results(fname, _to_plot)

            # save the model
            self.checkpoint_manager.save(checkpoint_number=self.last_epoch + epoch + 1)

            ''' eval phase'''
            if dataset.feed_val:
                for step in range(step_per_epoch_val):
                    class_loss, route_loss, uniq_loss, spoof_counts, eigenvalue, trace, _to_plot = \
                        self._train_one_step(next(dataset.feed_val), global_step, False)
                    if not config.args.log_less or (step + 1) % (int(step_per_epoch_val / 5)) == 0:
                        self.logger.info('Val-{:d}/{:d}: Cls:{:.3g}, Route:{:.3g}({:3.3f}, {:3.3f}), Uniq:{:.3g}, '
                                         'Counts:[{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d}]     '.
                                         format(step + 1, step_per_epoch_val,
                                                self.class_loss(class_loss, val=1),
                                                self.route_loss(route_loss, val=1), eigenvalue, trace,
                                                self.uniq_loss(uniq_loss),
                                                spoof_counts[0], spoof_counts[1], spoof_counts[2], spoof_counts[3],
                                                spoof_counts[4], spoof_counts[5], spoof_counts[6], spoof_counts[7]))
                    # plot the figure
                    if config.args.plot:
                        if (step + 1) % 100 == 0:
                            fname = config.args.logging_path + '/epoch-' + str(epoch + 1) + '-val-' + str(
                                step + 1) + '.png'
                            super().plot_results(fname, _to_plot)
                self.class_loss.reset()
                self.route_loss.reset()
                self.uniq_loss.reset()

            # time of one epoch
            self.logger.info('Time taken for epoch {} is {:3g} sec'.format(epoch + 1, time.time() - start))

    # @tf.function
    def _train_one_step(self, data_batch, step, training):
        dtn = self.dtn
        dtn_op = self.dtn_op
        image, labels, spoof_type, sensor_type, dataset_name = data_batch
        with tf.GradientTape() as tape:
            cls_pred, route_value, leaf_node_mask, tru_loss, mu_update, eigenvalue, trace = \
                dtn(image, labels, True)

            # supervised feature loss
            supervised_loss = leaf_l1_loss(cls_pred, labels, leaf_node_mask)

            # unsupervised tree loss
            route_loss = tf.reduce_mean(tf.stack(tru_loss[0], axis=0) * [1., 0.5, 0.5, 0.25, 0.25, 0.25, 0.25])
            uniq_loss = tf.reduce_mean(tf.stack(tru_loss[1], axis=0) * [1., 0.5, 0.5, 0.25, 0.25, 0.25, 0.25])
            eigenvalue = tf.reduce_mean(tf.stack(eigenvalue, axis=0) * [1., 0.5, 0.5, 0.25, 0.25, 0.25, 0.25])
            trace = tf.reduce_mean(tf.stack(trace, axis=0) * [1., 0.5, 0.5, 0.25, 0.25, 0.25, 0.25])
            unsupervised_loss = 2 * route_loss + 0.001 * uniq_loss

            # total loss
            if step > 10000:
                loss = supervised_loss + unsupervised_loss
            else:
                loss = supervised_loss

        if training:
            # back-propagate
            gradients = tape.gradient(loss, dtn.variables)
            dtn_op.apply_gradients(zip(gradients, dtn.variables))
            # Update mean values for each tree node
            mu_update_rate = self.config.TRU_PARAMETERS["mu_update_rate"]
            mu = [dtn.tru_0.project.mu, dtn.tru_1.project.mu, dtn.tru_2.project.mu, dtn.tru_3.project.mu,
                  dtn.tru_4.project.mu, dtn.tru_5.project.mu, dtn.tru_6.project.mu]
            for mu, mu_of_visit in zip(mu, mu_update):
                if step == 0:
                    update_mu = mu_of_visit
                else:
                    update_mu = mu_of_visit * mu_update_rate + mu * (1 - mu_update_rate)
                K.set_value(mu, update_mu)

        # leaf counts
        spoof_counts = []
        for leaf in leaf_node_mask:
            spoof_count = tf.reduce_sum(leaf[:, 0]).numpy()
            spoof_counts.append(int(spoof_count))

        _to_plot = [image, cls_pred]
        return supervised_loss, route_loss, uniq_loss, spoof_counts, eigenvalue, trace, _to_plot
