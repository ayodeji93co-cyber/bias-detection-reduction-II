from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

def evaluate_fairness(df, protected_attr='sex_Female'):
    dataset = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=df,
        label_names=['class_>50K'],
        protected_attribute_names=[protected_attr]
    )
    metric = BinaryLabelDatasetMetric(dataset, privileged_groups=[{protected_attr: 0}], unprivileged_groups=[{protected_attr: 1}])
    return metric.disparate_impact()