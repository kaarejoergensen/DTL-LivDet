import logging

import tensorflow as tf

from models.CRU import CRU
from models.Common import Conv
from models.SFL import SFL
from models.TRU import TRU


class DTN(tf.keras.models.Model):
    def __init__(self, filters, config):
        self.logger = logging.getLogger("main")
        super(DTN, self).__init__()

        self.config = config
        self.conv1 = Conv(filters, 5, apply_batchnorm=False)

        alpha = config.TRU_PARAMETERS['alpha']
        beta = config.TRU_PARAMETERS['beta']
        height = config.args.height

        self.leafs = int(pow(2, config.args.height) / 2)
        self.logger.info("Initializing DTN with height {} and {} leaf nodes".format(height, self.leafs))

        self.level_counts = [pow(2, x) for x in range(height)]

        for count in range(self.leafs):
            if count != self.leafs - 1:
                setattr(self, "cru_{}".format(count), CRU(filters))
                setattr(self, "tru_{}".format(count), TRU(filters, str(count + 1), alpha, beta))
            setattr(self, "sfl_{}".format(count), SFL(filters))

    @tf.function
    def call(self, x, label, training):
        if training:
            mask_spoof = label
            mask_live = 1 - label
        else:
            mask_spoof = tf.ones_like(label)
            mask_live = tf.zeros_like(label)
        x = self.conv1(x, training)

        route_values = []
        tru_losses = []
        clss = []
        leaf_node_mask = []

        previous_cru = [x]
        previous_tru = [[mask_spoof]]
        index = 0
        for level_count in self.level_counts:
            new_previous_cru = []
            new_previous_tru = []
            for node in range(level_count):
                if level_count != self.level_counts[-1]:
                    cru = getattr(self, "cru_{}".format(index))
                    x_cru = cru(previous_cru[int(node / 2)])
                    tru = getattr(self, "tru_{}".format(index))
                    x_tru, route_value, tru_loss = tru(x_cru, previous_tru[int(node / 2)][node % 2], training)
                    new_previous_cru.append(x_cru)
                    new_previous_tru.append(x_tru)

                    route_values.append(route_value)
                    tru_losses.append(tru_loss)
                else:
                    sfl = getattr(self, "sfl_{}".format(node))
                    cls = sfl(previous_cru[int(node / 2)], training)
                    x_tru = tf.concat([previous_tru[int(node / 2)][node % 2], mask_live], axis=1)

                    clss.append(cls)
                    leaf_node_mask.append(x_tru)
                index = index + 1
            previous_cru = new_previous_cru
            previous_tru = new_previous_tru

        if training:
            # for the training
            route_loss, recon_loss = map(list, zip(*tru_losses))
            mu_update = []
            eigenvalue = []
            trace = []
            for index in range(self.leafs - 1):
                tru = getattr(self, "tru_{}".format(index))
                mu_update.append(tru.project.mu_of_visit + 0)
                eigenvalue.append(tru.project.eigenvalue)
                trace.append(tru.project.trace)

            return clss, route_values, leaf_node_mask, [route_loss, recon_loss], mu_update, eigenvalue, trace
        else:
            return clss, route_values, leaf_node_mask
