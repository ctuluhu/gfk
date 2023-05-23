import pandas as pd

from main import remove_uncommon_features, drop_selected_columns


def test_remove_uncommon_features():
    df1 = pd.DataFrame(columns=['a', 'b', 'c', 'd'])
    df2 = pd.DataFrame(columns=['b', 'd', 'e', 'f'])

    df1, df2 = remove_uncommon_features(df1, df2)
    assert all(df1.columns == ['b', 'd'])
    assert all(df1.columns == df2.columns)


def test_drop_selected_columns():
    df = pd.DataFrame(columns=['a', 'b', 'c', 'd'])
    df = drop_selected_columns(df, ['a', 'd'])
    assert all(df.columns == ['b', 'c'])
