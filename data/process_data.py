import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load dataframe from filepaths

    INPUT
    messages_filepath: Path to the messages csv file
    categories_filepath: Path to the categories csv file

    OUTPUT
    df - pandas DataFrame joing the messages and the categories
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # Since the all the categories used in messages are present in categories we can use the inner join
    df = messages.merge(categories, how='inner', on=['id'])
    return df


def clean_data(df):
    """Clean data included in the DataFrame and transform categories part

    INPUT
    df:raw pandas DataFrame

    OUTPUT
    df: cleaned pandas DataFrame
    """
    categories_split = df['categories'].str.split(";", expand=True)
    row = categories_split.iloc[0].apply(lambda x: x.strip()[:-2])
    category_colnames = row.tolist()
    categories_split.columns = category_colnames

    for column in categories_split:
        # set each value to be the last character of the string
        categories_split[column] = categories_split[column].apply(lambda x: x.strip()[-1])
        # convert column from string to numeric
        categories_split[column] = categories_split[column].astype(int)

    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df,categories_split], axis=1)
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """Saves DataFrame (df) to database path"""
    name = 'sqlite:///' + database_filename
    engine = create_engine(name)
    df.to_sql('disaster_responses', engine, index=False)


def main():
    """Runs main functions: Loads the data, cleans it and saves it in a database"""
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()