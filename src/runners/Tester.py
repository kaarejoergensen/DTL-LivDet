import logging

import tensorflow as tf

from Dataset import Dataset
from runners.RunnerBase import RunnerBase


class Tester(RunnerBase):
    def __init__(self, config):
        super().__init__(config)

    def test(self):
        config = self.config
        logging.info("Testing using all types")
        dataset = Dataset(config)
        self._test(dataset)

    def _test(self, dataset):
        total_count = 0
        correct_count = 0
        spoof_type_incorrect_counts = {}
        spoof_type_correct_counts = {}
        for batch in dataset.feed:
            image, labels, spoof_type = batch
            cls_pred, route_value, leaf_node_mask = self.dtn(image, labels, False)
            # leaf counts
            # spoof_counts = []
            # for leaf in leaf_node_mask:
            #     spoof_count = tf.reduce_sum(leaf[:, 0]).numpy()
            #     spoof_counts.append(int(spoof_count))
            cls_total = tf.math.add_n(cls_pred) / len(cls_pred)
            index = 0
            for label in tf.unstack(labels):
                cls = cls_total[index].numpy()
                cls_result = cls - label.numpy()
                sp = spoof_type[index].numpy()
                if cls_result > 0.49 or cls_result < -0.49:
                    if sp not in spoof_type_incorrect_counts:
                        spoof_type_incorrect_counts[sp] = 0
                    spoof_type_incorrect_counts[sp] = spoof_type_incorrect_counts[sp] + 1
                    # logging.info("WRONG: label: {}, cls: {}, cls_result: {}, type: {}"
                    #              .format(label.numpy(), cls, cls_result, spoof_type[index].numpy()))
                else:
                    if sp not in spoof_type_correct_counts:
                        spoof_type_correct_counts[sp] = 0
                    spoof_type_correct_counts[sp] = spoof_type_correct_counts[sp] + 1
                    correct_count += 1
                index += 1
            total_count += len(image)
        logging.info("Correct: {}, incorrect: {}, total: {}"
                     .format(correct_count, total_count - correct_count, total_count))
        logging.info("Incorrect count: {}, correct count: {}"
                     .format(spoof_type_incorrect_counts, spoof_type_correct_counts))
