import logging

import tensorflow as tf

from Dataset import Dataset
from runners.RunnerBase import RunnerBase


class Tester(RunnerBase):
    def __init__(self, config):
        super().__init__(config)
        self.main_logger = logging.getLogger("main")
        self.spoof_logger = logging.getLogger("test_spoof")
        self.live_logger = logging.getLogger("test_live")

    def test(self):
        config = self.config
        self.main_logger.info("Testing using all types")
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
                if sp not in spoof_type_incorrect_counts:
                    spoof_type_incorrect_counts[sp] = 0
                if sp not in spoof_type_correct_counts:
                    spoof_type_correct_counts[sp] = 0
                if sp == b'live':
                    self.live_logger.info(cls[0])
                else:
                    self.spoof_logger.info(cls[0])
                if cls_result > 0.49 or cls_result < -0.49:
                    spoof_type_incorrect_counts[sp] = spoof_type_incorrect_counts[sp] + 1
                else:
                    spoof_type_correct_counts[sp] = spoof_type_correct_counts[sp] + 1
                    correct_count += 1
                index += 1
            total_count += len(image)
        self.main_logger.info("Correct: {}, incorrect: {}, total: {}"
                              .format(correct_count, total_count - correct_count, total_count))
        for key, correct in spoof_type_correct_counts.items():
            incorrect = spoof_type_incorrect_counts[key]
            total = correct + incorrect
            percent_correct = 100 / total * correct
            self.main_logger.info("Type: {}, incorrect: {}, correct: {}, total: {}, percentage correct: {}"
                                  .format(key, incorrect, correct, total, percent_correct))
        self.main_logger.info("Incorrect count: {}, correct count: {}"
                              .format(spoof_type_incorrect_counts, spoof_type_correct_counts))
