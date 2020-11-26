"""Helper script to generate tuple txt data from CSVs."""
import os
import glob
from pathlib import Path
from typing import Sequence, Callable

import pandas as pd
from pandas._typing import TimestampConvertibleTypes

# Define the directory to find CSVs, and optional output TXT directory.
test_dir = Path(__file__).parent.parent

CSV_DIR = os.path.join(test_dir, "test_data", "rounding")
TXT_DIR = CSV_DIR


def main() -> None:
    """Convert the files."""
    # If you only want specific files in the CSV directory list them
    # here.
    # files = [
    #     'myfile1.csv',
    #     'myfil2.csv',
    # ]
    # convert_all_csvs(files)

    convert_all_csvs()


def _generate_test_data(df: pd.DataFrame, filename: str) -> None:
    """Generate hard-coded test data from a DataFrame."""
    headers = (df.index.name,) + tuple(df.columns)
    df_as_list = list(df.itertuples(index=True, name=None))

    with open(filename, 'w') as f:
        f.write(str(headers)+',\n')
        f.write('\n'.join(str(t)+',' for t in df_as_list))


def _load(
    filename: str,
    date_parser: Callable[[TimestampConvertibleTypes], pd.Series] = None,
) -> pd.DataFrame:
    """Quick load file with optional date parser."""
    return pd.read_csv(
        filename,
        index_col=0,
        parse_dates=True,
        dayfirst=True,
        date_parser=date_parser,
    )


def convert_all_csvs(
    files: Sequence[str] = None,
    date_parser: Callable[[TimestampConvertibleTypes], pd.Series] = None,
) -> None:
    """Convert all CSVs to tuple text data.

    Default behaviour is to convert all files in directory given by
    CSV_TXT, but files can be specified using files argument.

    Parameters
    ----------
    files: list of str, optional
        The filenames to convert.
    date_parser: callable, optional
        To convert date types in formats not picked up by pandas i.e.
        "201701".

    """
    if not files:
        files = glob.glob(CSV_DIR + r'\*.csv')
    else:
        files = [os.path.join(CSV_DIR, file) for file in files]

    txt_files = [
        os.path.join(TXT_DIR, Path(s).name.split('.')[0] + '.txt')
        for s in files
    ]

    for i, file in enumerate(files):
        df = _load(file, date_parser)
        _generate_test_data(df, txt_files[i])


if __name__ == "__main__":
    main()
