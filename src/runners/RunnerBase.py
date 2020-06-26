import logging

import cv2
import numpy as np
import tensorflow as tf

from models.DTN import DTN


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


class RunnerBase:
    def __init__(self, config):
        self.logger = logging.getLogger("main")
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
                self.logger.info("Restored from {}".format(last_checkpoint))
            else:
                self.logger.info("Initializing from scratch.")
        else:
            self.logger.info("Ignoring checkpoint and initializing from scratch.")

    def plot_results(self, fname, result_list):
        self.logger.info("Plotting results with name {}".format(fname))
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
