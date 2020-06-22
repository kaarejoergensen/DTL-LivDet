import logging

import tensorflow as tf

from Dataset import Dataset
from runners.RunnerBase import RunnerBase


class Tester(RunnerBase):
    def __init__(self, config):
        super().__init__(config)

    def test(self):
        config = self.config
        types = config.args.testing_types
        logging.info("Testing using types {}".format(types))
        dataset = Dataset(config, types)
        self._test(dataset)

    def _test(self, dataset):
        total_count = 0
        correct_count = 0
        for batch in dataset.feed:
            image, labels = batch
            cls_pred, route_value, leaf_node_mask = self.dtn(image, labels, False)
            # leaf counts
            spoof_counts = []
            for leaf in leaf_node_mask:
                spoof_count = tf.reduce_sum(leaf[:, 0]).numpy()
                spoof_counts.append(int(spoof_count))
            cls_total = tf.math.add_n(cls_pred) / len(cls_pred)
            index = 0
            for label in tf.unstack(labels):
                cls = cls_total[index].numpy()
                cls_result = cls - label.numpy()
                if cls_result > 0.49 or cls_result < -0.49:
                    logging.info("WRONG: label: {}, cls: {}, cls_result: {}"
                                 .format(label.numpy(), cls, cls_result))
                else:
                    correct_count += 1
                index += 1
            total_count += len(image)
        logging.info("Correct: {}, incorrect: {}, total: {}".
                     format(correct_count, total_count - correct_count, total_count))
