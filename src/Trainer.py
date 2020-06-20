import logging
import time

import cv2
import numpy as np
import tensorflow as tf

from Dataset import Dataset
from Loss import leaf_l1_loss
from models.DTN import DTN


def plotResults(fname, result_list):
    logging.info("Plotting results with name {}".format(fname))
    columm = []
    for fig in result_list:
        shape = fig.shape
        fig = fig.numpy()
        row = []
        for idx in range(shape[0]):
            item = fig[idx, :, :, :]
            if item.shape[2] == 1:
                item = np.concatenate([item, item, item], axis=2)
            item = cv2.cvtColor(cv2.resize(item, (128, 128)), cv2.COLOR_RGB2BGR)
            row.append(item)
        row = np.concatenate(row, axis=1)
        columm.append(row)
    columm = np.concatenate(columm, axis=0)
    img = np.uint8(columm * 255)
    cv2.imwrite(fname, img)


class Error:
    def __init__(self):
        self.value = 0
        self.value_val = 0
        self.step = 0
        self.step_val = 0

    def __call__(self, update, val=0):
        if val == 1:
            self.value_val += update
            self.step_val += 1
            return self.value_val / self.step_val
        else:
            self.value += update
            self.step += 1
            return self.value / self.step

    def reset(self):
        self.value = 0
        self.value_val = 0
        self.step = 0
        self.step_val = 0


class Trainer:
    def __init__(self, config):
        self.config = config
        # model
        self.dtn = DTN(32, config)
        # model optimizer
        self.dtn_op = tf.compat.v1.train.AdamOptimizer(config.args.lr, beta1=0.5)
        # model losses
        self.class_loss = Error()
        self.route_loss = Error()
        self.uniq_loss = Error()
        # model saving setting
        self.last_epoch = 0
        self.compile()

    def compile(self):
        checkpoint_dir = self.config.args.checkpoint_path
        checkpoint = tf.train.Checkpoint(dtn=self.dtn,
                                         dtn_optimizer=self.dtn_op)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=30)
        if not self.config.args.ignore_checkpoint:
            last_checkpoint = self.checkpoint_manager.latest_checkpoint
            checkpoint.restore(last_checkpoint)
            if last_checkpoint:
                self.last_epoch = int(last_checkpoint.split('-')[-1])
                logging.info("Restored from {}".format(last_checkpoint))
            else:
                logging.info("Initializing from scratch.")
        else:
            logging.info("Ignoring checkpoint and initializing from scratch.")

    def train(self):
        config = self.config
        epochs = config.args.epochs
        types = config.args.training_types
        if config.args.validate:
            logging.info("Training with validation")
            for val_type in types:
                types_to_load = [t for t in types if t != val_type]
                logging.info("Training with types {} and validation type {}".format(types_to_load, val_type))
                dataset = Dataset(config, types_to_load, val_type)
                self._train(dataset, int(epochs / len(types)))
        else:
            logging.info("Training without validation, using types {}".format(types))
            dataset = Dataset(config, types)
            self._train(dataset, epochs)

    def _train(self, dataset, epochs):
        config = self.config
        step_per_epoch = config.args.steps
        step_per_epoch_val = config.args.steps_val
        if dataset.feed_val:
            logging.info("Training for {} epochs, with {} steps per epoch and {} steps per epoch for validation"
                         .format(epochs, step_per_epoch, step_per_epoch_val))
        else:
            logging.info("Training for {} epochs, with {} steps per epoch".format(epochs, step_per_epoch))

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

                # display loss
                global_step += 1
                logging.info(
                    'Epoch {:d}-{:d}/{:d}: Cls:{:.3g}, Route:{:.3g}({:3.3f}, {:3.3f}), Uniq:{:.3g}, '
                    'Counts:[{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d}]     '.
                        format(epoch + 1, step + 1, step_per_epoch,
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
                        plotResults(fname, _to_plot)

            # save the model
            if (epoch + 1) % 1 == 0:
                self.checkpoint_manager.save(checkpoint_number=epoch + 1)

            ''' eval phase'''
            if dataset.feed_val:
                for step in range(step_per_epoch_val):
                    class_loss, route_loss, uniq_loss, spoof_counts, eigenvalue, trace, _to_plot = \
                        self._train_one_step(next(dataset.feed_val), global_step, False)

                    # display something
                    logging.info('Val-{:d}/{:d}: Cls:{:.3g}, Route:{:.3g}({:3.3f}, {:3.3f}), Uniq:{:.3g}, '
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
                            plotResults(fname, _to_plot)
                self.class_loss.reset()
                self.route_loss.reset()
                self.uniq_loss.reset()

            # time of one epoch
            logging.info('Time taken for epoch {} is {:3g} sec'.format(epoch + 1, time.time() - start))

    # @tf.function
    def _train_one_step(self, data_batch, step, training):
        dtn = self.dtn
        dtn_op = self.dtn_op
        image, labels = data_batch
        with tf.GradientTape() as tape:
            cls_pred, route_value, leaf_node_mask, tru_loss, mu_update, mu, eigenvalue, trace = \
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
            for mu, mu_of_visit in zip(mu, mu_update):
                if step == 0:
                    update_mu = mu_of_visit
                else:
                    update_mu = mu_of_visit * mu_update_rate + mu * (1 - mu_update_rate)
                mu = update_mu

        # leaf counts
        spoof_counts = []
        for leaf in leaf_node_mask:
            spoof_count = tf.reduce_sum(leaf[:, 0]).numpy()
            spoof_counts.append(int(spoof_count))

        _to_plot = [image, cls_pred]
        return supervised_loss, route_loss, uniq_loss, spoof_counts, eigenvalue, trace, _to_plot
