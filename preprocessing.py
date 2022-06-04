import json
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def read_df(folder_path, file_path, date_col='date', id_col='user_id'):

    if isinstance(folder_path, str):
        folder_path = Path(folder_path)
    if isinstance(file_path, str):
        file_path = Path(file_path)
    df = pd.read_json(folder_path / file_path)
    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values([id_col, date_col], inplace=True)

    return df


def split_col_by_delimiter(df, col_to_split, delimiter, replace_sym="_", left_postfix="0", right_postfix="1"):

    if replace_sym:
        df[col_to_split] = df[col_to_split].str.replace(replace_sym, delimiter)

    df[[f'{col_to_split}_{left_postfix}', f'{col_to_split}_{right_postfix}']
       ] = df[col_to_split].str.split(delimiter, expand=True)
    df.drop([col_to_split], axis=1, inplace=True)

    return df


def expand_json_col_to_rows(df, col_to_expand):

    df_expand = pd.DataFrame(df[col_to_expand].tolist()).T.melt().dropna()
    df_tmp = pd.DataFrame(df_expand.value.tolist(),
                          columns=df_expand['value'][0].keys(),
                          index=df_expand.variable)
    df = df.join(df_tmp)
    df.drop(columns=[col_to_expand], inplace=True)

    return df


def convert_catergorical_cols(df, categorical_cols):

    label_encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = df[col].astype(str)
        df[col] = label_encoder.fit_transform(df[col])

    return df


def exctract_date_features(df, date_features, date_col='date'):

    for col in date_features:
        if col == 'dayofweek':
            df[col] = df[date_col].dt.dayofweek
        if col == 'dayofyear':
            df[col] = df[date_col].dt.dayofyear
        if col == 'weekofyear':
            df[col] = df[date_col].dt.weekofyear
        if col == 'monthofyear':
            df[col] = df[date_col].dt.month
        if col == 'year':
            df[col] = df[date_col].dt.year
        if col == 'dayofmonth':
            df[col] = df[date_col].dt.day
    df.drop(date_col, axis=1, inplace=True)
    
    return df


def exctract_time_features(df, time_features, time_col='time', format_time='%H:%M:%S'):

    for col in time_features:
        if col == 'hour':
            df[col] = pd.to_datetime(df[time_col], format=format_time).dt.hour
        if col == 'minute':
            df[col] = pd.to_datetime(df[time_col], format=format_time).dt.minute
            
    df.drop(time_col, axis=1, inplace=True)
    
    return df

def exclude_periods(df, exclude_before, exclude_after):
    
    if exclude_before:
        df = df[(df['date'] > exclude_before)]
    if exclude_after:
        df = df[(df['date'] < exclude_after)]
    
    
    return df


def main():
    path_config = "config.json"

    with(open(path_config, 'r')) as file:
        config = json.load(file)

    df = read_df(Path(config['source_data_path']), Path(config['train']))
    
    df = exclude_periods(df, config['exclude_dates']['exclude_before'], config['exclude_dates']['exclude_after'])
    
    for col in config['col_to_split']:
        if col == 'locale':
            df = split_col_by_delimiter(df=df,
                                        col_to_split=col,
                                        left_postfix='lang',
                                        right_postfix='country',
                                        delimiter='-',
                                        replace_sym='_'
                                        )
        if col == 'location':
            df = split_col_by_delimiter(df=df,
                                        col_to_split=col,
                                        left_postfix='country',
                                        right_postfix='city',
                                        delimiter='/',
                                        replace_sym=None
                                        )

    for col in config['col_to_expand']:
        df = expand_json_col_to_rows(df=df, col_to_expand=col)

    df = convert_catergorical_cols(df, config['categorical_columns'])

    df = exctract_date_features(df, config['date_features'])
    
    df = exctract_time_features(df, config['time_features'])

    df.to_csv(Path(config['source_data_path']) / Path('preprared_df.csv'), index=False)


if __name__ == '__main__':
    main()
