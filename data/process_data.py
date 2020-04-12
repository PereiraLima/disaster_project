import sys
import pandas as pd
import re
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ Load data, merge them together and return the merged dataframe: df"""

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, on='id', how='left')

    return df

def clean_data(df):
    """ - Clean categories col: split it into different categorical columns, one for each category
        - replace all the values in new categories columns by 1 or 0
        - Drop duplicates
        - return clean dataframe: df """

    # Extract categories column from df: categories
    categories = df.categories.str.split(';', expand=True)

    # select the first row of the categories dataframe to create the col names
    row = categories.iloc[1,:].tolist()
    row_cleaned = [re.findall(r'[^0-9-]+', s) for s in row]

    # Create a list with the column names: category_colnames
    category_colnames = []
    for i in row_cleaned:
        category_colnames.append(i[0])

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Replace the values in categories df by 1 or 0 and make sure the dtype is integer
    for c in category_colnames:
        categories[c] = [int(x[-1]) for x in categories[c]]

    # drop the original categories column from `df` and replace them by the new columns
    df.drop(columns=['categories'], axis=1, inplace=True)
    df = df.join(categories)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
    """ Save the clean dataframe (df) in the table 'message_cat' in the database in the parameter"""
    engine = create_engine(database_filename)
    df.to_sql('message_cat', engine, index = False)


def main():
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