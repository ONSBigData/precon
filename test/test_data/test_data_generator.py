# -*- coding: utf-8 -*-
import glob

import pandas as pd

def generate_test_data(df, filename):
    """Generate hard-coded test data from a DataFrame."""
    headers = (df.index.name,) + tuple(df.columns)
    df_as_list = list(df.itertuples(index=True, name=None))
    
    with open(filename, 'w') as f:
        f.write(str(headers)+',\n')
        f.write('\n'.join(str(t)+',' for t in df_as_list))
    

def load(filename):
    return pd.read_csv(filename, index_col=0, parse_dates=True, dayfirst=True)


def main():
    files = glob.glob('*.csv')
    txt_files = [s.split('.')[0] + '.txt' for s in files]
    
    for i, file in enumerate(files):
        df = load(file)
        generate_test_data(df, txt_files[i])


if __name__ == "__main__":
    main()
