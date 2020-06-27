import logging

import tensorflow as tf

from Dataset import Dataset
from runners.RunnerBase import RunnerBase


class Tester(RunnerBase):
    def __init__(self, config):
        super().__init__(config)
        self.main_logger = logging.getLogger("main")

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
        result_map = {}
        total_spoof_counts = []
        for batch in dataset.feed:
            image, labels, spoof_type, sensor_type, dataset_name = batch
            cls_pred, route_value, leaf_node_mask = self.dtn(image, labels, False)
            spoof_counts = []
            for leaf in leaf_node_mask:
                spoof_count = tf.reduce_sum(leaf[:, 0]).numpy()
                spoof_counts.append(int(spoof_count))
            total_spoof_counts.append(spoof_counts)
            cls_total = tf.math.add_n(cls_pred) / len(cls_pred)
            index = 0
            for label in tf.unstack(labels):
                cls = cls_total[index].numpy()
                cls_result = cls - label.numpy()
                ds = dataset_name[index].numpy().decode("utf-8")
                st = sensor_type[index].numpy().decode("utf-8")
                sp = spoof_type[index].numpy().decode("utf-8")
                if ds not in result_map:
                    self.main_logger.info("Adding dataset {}".format(ds))
                    result_map[ds] = {}
                if st not in result_map[ds]:
                    self.main_logger.info("Adding sensor type {} to dataset {}".format(ds, st))
                    result_map[ds][st] = {'live': [], 'spoof': []}
                if sp == 'live':
                    result_map[ds][st]['live'].append(cls[0])
                else:
                    result_map[ds][st]['spoof'].append(cls[0])
                if sp not in spoof_type_incorrect_counts:
                    spoof_type_incorrect_counts[sp] = 0
                if sp not in spoof_type_correct_counts:
                    spoof_type_correct_counts[sp] = 0
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

        self.main_logger.info("Total spoof count: {}".format([sum(x) for x in zip(*total_spoof_counts)]))

        for dataset_name, sensor_types in result_map.items():
            for sensor_type, result_types in sensor_types.items():
                for result_type, result_list in result_types.items():
                    name = "testing.{}.{}.{}".format(dataset_name, sensor_type, result_type)
                    with open("{}/{}".format(self.config.args.logging_path, name), mode='w') as f:
                        for item in result_list:
                            f.write("{}\n".format(item))
        with open("{}/spoof_count".format(self.config.args.logging_path), mode="w") as f:
            for spoof_list in total_spoof_counts:
                f.write("{}\n".format(" ".join([str(x) for x in spoof_list])))