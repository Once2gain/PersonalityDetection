import numpy
import pandas as pd


value_name = ['introspection_value', 'temper_value', 'attitude_value', 'sensitivity_value', 'primary_emotion',
              'secondary_emotion', 'polarity_label', 'polarity_value', 'semantics1', 'semantics2', 'semantics3',
              'semantics4', 'semantics5']


def inject_to_tree(tree, concept_name, value):
    sub_tree = {
        'concept-value': [],
        'sub-concepts': {}
    }
    sub_names = concept_name.split('_', 1)

    if sub_names[0] in tree.keys():
        sub_tree = tree[sub_names[0]]
    else:
        tree[sub_names[0]] = sub_tree

    if len(sub_names) == 1:
        sub_tree['concept-value'] = value
    else:
        inject_to_tree(sub_tree['sub-concepts'], sub_names[1], value)


def build_sentic_tree_from_xls(xls_file):
    """
    load senticNet to senticTree
    """
    sentic_tree = {}

    df = pd.read_table(xls_file)
    for i in df.index:
        if i/100 == 0:
            print('process {}-line/{}'.format(i, df.shape[0]))
        concept_name = str(df.loc[i].values[0]).strip()
        value = []
        for j in range(1, df.shape[1]):
            val = df.loc[i].values[j]
            if type(val) == numpy.float64:
                val = numpy.around(val, 3)
            value.append(val)
        inject_to_tree(sentic_tree, concept_name, value)

    return sentic_tree


if __name__ == '__main__':
    sentic_tree = build_sentic_tree_from_xls('senticnet.xls')
    numpy.save('sentic_tree.npy', sentic_tree)
