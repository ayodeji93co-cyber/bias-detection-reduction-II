from aif360.algorithms.preprocessing import Reweighing

def mitigate_bias(dataset, protected_attr='sex_Female'):
    RW = Reweighing(unprivileged_groups=[{protected_attr: 1}], privileged_groups=[{protected_attr: 0}])
    RW.fit(dataset)
    return RW.transform(dataset)