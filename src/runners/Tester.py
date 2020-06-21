import logging

import tensorflow as tf

from Dataset import Dataset
from runners.RunnerBase import RunnerBase


class Tester(RunnerBase):
    def __init__(self, config):
        super().__init__(config)

    def test(self):
        config = self.config
        types = config.args.training_types
        logging.info("Testing using types {}".format(types))
        dataset = Dataset(config, types)
        self._test(dataset)

    def _test(self, dataset):
        for batch in dataset.feed:
            image, labels = batch
            cls_pred, route_value, leaf_node_mask = self.dtn(image, labels, False)
            # leaf counts
            spoof_counts = []
            for leaf in leaf_node_mask:
                spoof_count = tf.reduce_sum(leaf[:, 0]).numpy()
                spoof_counts.append(int(spoof_count))
            logging.info("spoof_counts:{}"
                         .format(spoof_counts))
