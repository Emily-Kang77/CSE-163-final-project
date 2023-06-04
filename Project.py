"""
Emily Kang, Sophie Tacher, Ryan Nguyen
CSE 163

This project focuses on analyzing housing data from
Northern California from the years 2000 to 2018,
specifying visualizations for data about housing prices,
factors in a house, demographic housing owners, and housing locations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import os
import plotly.express as px
sns.set(font_scale=2)


def main():
    # reads in primary data file: displaying apartment data
    # from craigslist 2000-2018
    home_df = pd.read_csv('clean_2000_2018.csv', low_memory=False)

    # reads in new data files representing income information from 2010-2018
    path = 'income'

    # reads in geographical data representing counties in California 
    counties_shapes = gpd.read_file('./CA_Counties/CA_Counties_TIGER2016.shp')

    # for question 3
    populations_df = pd.read_csv('population-decennial-2020.csv')

    # QUESTION 1
    clean_home_df = make_clean_housing_data(home_df)
    plot_price_trends(clean_home_df)
    plot_bed_to_price(clean_home_df)
    
    # QUESTION 2
    all_income = make_income_datasets(path)
    all_income = filter_all_income(all_income)
    av_incomes, av_prices = make_average_prices_incomes(all_income,
                                                        clean_home_df)
    plot_average_incomes_prices(av_incomes, av_prices)

    # QUESTION 3
    pops_shapes, price_by_county = prep_q3_populations(populations_df,
                                                       clean_home_df,
                                                       counties_shapes)
    plot_populations_prices(price_by_county, pops_shapes)


def make_clean_housing_data(home_df: pd.DataFrame) -> pd.DataFrame:
    '''
    This function cleans the home dataset from Craigslist so it has
    less scams, hotels, and motels.
    '''
    # FILTER OUT irrelevant beds and baths, with less likely scam prices
    is_not_missing = (home_df['beds'].notna()) & (home_df['baths'].notna())
    # apply it early on to avoid errors
    home_df = home_df[is_not_missing]
    is_not_too_cheap = home_df['price'] > 700
    # filter out hotel/motel data
    not_motel_hotel = ~(home_df['title'].str.lower().str.contains('.*night.*', regex=True) | 
                        home_df['title'].str.lower().str.contains('.*hotel*', regex=True) | 
                        home_df['title'].str.lower().str.contains('.*motel*', regex=True))

    counties = ['san francisco', 'santa clara', 'san mateo', 'alameda',
                'contra costa', 'sonoma']

    is_counties = home_df['county'].isin(counties)
    not_expensive = home_df['price'] < 12000
    home_df = home_df.astype({'beds': 'int64'})
    beds_within_range = (home_df['beds'] <= 4) & (home_df['beds'] > 0)
    baths_within_range = (home_df['baths'] <= 2)

    # APPLY MASKS and turn into integer
    clean_home_df = home_df[is_not_too_cheap 
                            & not_motel_hotel & beds_within_range
                            & baths_within_range & is_counties
                            & not_expensive]

    return clean_home_df


# NOTE: Question 1: How have prices of apartments changed over time based on their types?
# THIS SHOWS DATA FOR ALL COUNTIES, grouped by the bed/bath information.
# Create Line Plots for relationship between Apartment Pricing and Apartment Types
def plot_price_trends(clean_home_df: pd.DataFrame) -> None:
    """
    This function takes in a dataframe representing housing data
    for northern California off craigslist from approximately 2000 to
    2020 and plots a series of graphs representing how prices have
    changed based on bed and bath amounts
    """
    plots = sns.relplot(data=clean_home_df, x='year', y='price', col='beds',
                        row='baths', kind='line')
    plots.fig.suptitle('Prices over time for 1-4 beds for 1, 1.5, 2 baths.',
                       size='xx-large')
    plots.fig.subplots_adjust(top=0.85)
    plots.set_axis_labels('Year (2000-2018)', 'Prices (USD)')
    plt.savefig('price_trends_apartment_type.png')


# NOTE: Question 1 pt.2, simplifying to how have overall price per beds changed over time
def plot_bed_to_price(clean_home_df: pd.DataFrame) -> None:
    """
    his function takes in a dataframe representing housing data
    for northern California off craigslist from approximately 2000 to
    2020 and plots a graph and plots price change over time based on a ratio
    that averages price to bed amounts
    """
    # Focuses the year range of our dataset
    data_y2k = (clean_home_df['year'] >= 2000)
    # Creates a dataFrame that will represent our additional ratio for this question
    price_per_bed_data = clean_home_df[data_y2k].copy()

    # Creates a new column representing a ratio of price changes and bed changes
    price_per_bed_data['price/beds'] = price_per_bed_data['price'] / price_per_bed_data['beds']

    # plot it all
    plots = sns.relplot(data=price_per_bed_data, x='year', y='price/beds', kind='line', height=6, aspect=3)
    plots.set_axis_labels('Year (2000-2018)', 'Prices of Apartments (USD)', fontsize=16)
    plots.set_xticklabels(fontsize=16)
    plots.set_yticklabels(fontsize=16)
    plt.xlim(2000)
    plt.title("Changes in Price Over Time Based on Amount of Beds", y=1.2, fontsize=20)
    plt.savefig('bed_to_price.png')


# NOTE: QUESTION 2: how does average income affect housing prices by COUNTY
def make_income_datasets(path: str) -> pd.DataFrame:
    """
    This function takes in a string representing a path to
    a directory of files representing income information from
    2010-2018 for different races and locations in California.
    It returns a dataFrame that merges all these data into
    one big dataframe.
    """
    income_datasets = [] # build this up

    for filename in os.listdir(path):
        # get file name without csv extension
        file_tokens = filename[0:filename.find('.csv')]
        year = file_tokens[-4:] # gets year
        dataset = pd.read_csv(path + '/' + filename)
        dataset['Year'] = year # adds year column

        # make column names shorter and more readable
        dataset.columns = dataset.columns.str.replace('!!', ' ', regex = False)
        dataset.columns = dataset.columns.str.replace('California ', '', regex = False)
        dataset.columns = dataset.columns.str.replace(' (dollars) Estimate', '', regex = False)
        dataset = dataset.dropna()
        income_datasets.append(dataset)

    all_income = income_datasets[0]
    # all_income = None
    print(all_income.columns)
    # merge the list of datasets
    for dataset in income_datasets[1:]:
        all_income = pd.concat([all_income, dataset], join='inner', axis=0)
        print(all_income.columns)

    # all_income.head(10)

    all_income = all_income.loc[:, all_income.columns.str.contains('Median') | 
                                    all_income.columns.str.contains('Group') | 
                                    all_income.columns.str.contains('Year')]

    all_income.rename(columns={'Label (Grouping)':'Race'}, inplace=True)
    # contains is used because these values are right-justified with unknown amounts of spaces
    return all_income


def make_race_filters(all_income: pd.DataFrame) -> list[pd.Series]:
    """
    This makes filters for white, black, native american / alaskan native,
    and asian people.
    """
    white_filter = all_income['Race'].str.contains("White alone")
    black_filter = all_income['Race'].str.contains("Black or African American")
    native_filter = all_income['Race'].str.contains("American Indian and Alaska Native")
    asian_filter = all_income['Race'].str.contains("Asian")
    race_filters = [white_filter, black_filter, native_filter, asian_filter]
    return race_filters


def filter_all_income(all_income: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out all incomes
    """
    # delete commas in all rows since only the median columns have commas
    all_income = all_income.replace(',', '', regex=True)
    # could honestly combine all these into  one regex line
    all_income = all_income.replace('N', '0')
    all_income = all_income.replace('-', '0')
    all_income = all_income.replace('(X)', '0')

    # turn into integer
    cols_medians = ['Alameda County, Median income',
                    'Contra Costa County, Median income',
                    'San Francisco County, Median income',
                    'San Mateo County, Median income', 'Santa Clara County, Median income',
                    'Sonoma County, Median income']

    # convert medians from strings to integers
    for col in cols_medians:
        all_income[col] = all_income[col].astype('int32')

    return all_income

# NOTE: Still Q2. MAKE AVERAGES NOW. USE PLOTLY
def make_average_prices_incomes(all_income: pd.DataFrame,
                               clean_home_df: pd.DataFrame) -> \
                               tuple[pd.DataFrame, pd.DataFrame]:
    """
    This ...
    """
    races_list = ['White', 'Black', 'Native', 'Asian']
    list_df_incomes = []
    race_filters = make_race_filters(all_income)

    # put average income dataframes in a list
    
    for i in range(0, len(races_list)):
        race_income = all_income[race_filters[i]].sort_values('Year', 
                                                              ascending=True)
        print(race_income)
        race = races_list[i]
        # add 'Average' column, then filter down to just Year and Average.
        race_income[race + ' Average'] = race_income.iloc[:, 1:7].mean(axis=1)
        race_income = race_income.loc[:, ['Year', race + ' Average']]
        list_df_incomes.append(race_income)  # add this df to list
    
    # make ONE dataset of average incomes by merging all of them.
    average_incomes = list_df_incomes[0]
    for i in range(1, len(list_df_incomes)):
        average_incomes = average_incomes.merge(list_df_incomes[i],
                                                left_on='Year',
                                                right_on='Year')

    # MAKE NEW DF FOR MONTHLY AVERAGE HOUSING PRICES X12
    average_prices = clean_home_df.groupby('year')['price'].mean()
    average_prices = average_prices * 12

    # make dataset of average incomes by merging onto one of them. Then rename.
    average_incomes = list_df_incomes[0]
    for i in range(1, len(list_df_incomes)):
        average_incomes = average_incomes.merge(list_df_incomes[i], left_on='Year', right_on='Year')

    # Remove years that don't match with income
    average_prices = average_prices[average_prices.index >= 2010]
    return average_incomes, average_prices


# PLOT
def plot_average_incomes_prices(average_incomes: pd.DataFrame,
                                price_avg_2010_2018: pd.DataFrame):
    """
    This uses the external Plotly library, so it is not saved as a fig. The
    plot is interactive.
    """
    # average_incomes is already AT 2010 - 2018.
    fig = px.line(average_incomes, x="Year", y=average_incomes.columns[1:],
                labels={
                    "Year":"Year (2010-2018)",
                    "value": "Amount of USD"
                }, title="2010-2018: Average Incomes by Demographics and California 1-4 bed housing prices")

    fig.add_scatter(x=price_avg_2010_2018.index, y=price_avg_2010_2018, name='Average Price of Housing')
    fig.show()


def prep_q3_populations(populations_df: pd.DataFrame,
                        clean_home_df: pd.DataFrame,
                        counties_shapes: pd.DataFrame) -> tuple[pd.DataFrame]:
    """
    This prepares population data to shade in the geographical plot,
    and the average housing prices by county.
    """
    # Clarifies data to only relevant counties (those with most information)
    counties = ['San Francisco', 'Santa Clara', 'San Mateo', 'Alameda', 'Contra Costa', 'Sonoma']
    # this selects the 4 races
    populations_df = populations_df.iloc[4:8, :]
    populations_df = populations_df.transpose()

    populations_df.columns = populations_df.iloc[0]
    populations_df = populations_df.tail(-1)
    populations_df.reset_index(inplace=True, drop=False)

    # normalize county names in index column
    populations_df.columns = populations_df.columns.str.lstrip()
    populations_df['index'] = populations_df['index'].str.replace('County, California', '', regex=False)

    populations_df['index']= populations_df['index'].str.lower().str.strip()

    # Filters price dataframe to only use 2018 prices
    average_2018_prices = clean_home_df[clean_home_df['year'] == 2018]

    # Groups the county prices into a mean and creates a dataframe with
    # their county and average price
    price_by_county = average_2018_prices.groupby('county')['price'].mean()
    price_by_county = price_by_county.to_frame()
    price_by_county.reset_index(inplace=True, drop=False)

    # select only the 6 counties we're looking at in the geodataframe
    mask_counties_shapes = counties_shapes['NAME'].isin(counties)
    counties_shapes = counties_shapes[mask_counties_shapes].copy()

    # merge counties_shapes with populations_df
    counties_shapes['NAME'] = counties_shapes['NAME'].str.lower().str.strip()
    pops_shapes = counties_shapes.merge(populations_df, left_on='NAME',
                                        right_on='index', how='inner')

    return pops_shapes, price_by_county


def plot_populations_prices(price_by_county: pd.DataFrame,
                            pops_shapes: pd.DataFrame) -> None:
    """
    Question 3. Plot 4 geographical plots showing population in
    each county, and 1 bar chart with the average housing prices
    of each county.
    """
    fig = px.bar(price_by_county, x='county', y='price', labels={
        'county': 'County',
        'price': 'Average Housing Price for 1-4 Bedrooms'
    }, title='Average Housing Prices for 1-4 bedrooms by county, 2018')
    fig.show()

    # plot them and make them monotone to compare shades easily
    # hard to put in for loop without nesting because there's 2 lists of axes
    # in one list
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(30, 30))
    pops_shapes.plot(ax=ax1, column='White alone', legend=True, cmap='Greys',
                     edgecolor='green', linewidth=3)
    ax1.set_title('White Population')
    ax1.set_facecolor('lightgray')

    pops_shapes.plot(ax=ax2, column='Black or African American alone',
                     legend=True, cmap='Purples', edgecolor='green', linewidth=3)
    ax2.set_title('Black/African American Population')
    ax2.set_facecolor('lightgray')

    pops_shapes.plot(ax=ax3, column='American Indian and Alaska Native alone',
                     legend=True, cmap='Reds', edgecolor='green', linewidth=3)
    ax3.set_title('American Indian/Alaska Native Population')
    ax3.set_facecolor('lightgray')

    pops_shapes.plot(ax=ax4, column='Asian alone', legend=True, cmap='Blues',
                     edgecolor='green', linewidth=3)
    ax4.set_title('Asian Population')
    ax4.set_facecolor('lightgray')

    plt.savefig('Q3_populations_prices.png')


if __name__ == '__main__':
    main()
