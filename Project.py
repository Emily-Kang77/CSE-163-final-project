"""
Emily Kang, Sophie Tacher, Ryan Nguyen
CSE 163

This project focuses on analyzing housing data from six counties of
Northern California from the years 2000 to 2018. The counties are
San Francisco, Santa Clara, San Mateo, Alameda, Contra Costa, and Sonoma.

This plots housing prices for combinations of beds and bathrooms,
as well as the average price change over time for 1-4 bed and 0-2 bath
homes.

Then for 2018, it plots the average incomes of all the counties for 4 races
alongside the average housing prices of the counties.

Finally, for 2018, it plots the populations of the counties onto
4 geographical maps by race, and compares it to a bar chart of
average housing prices by county.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import os
import plotly.express as px
sns.set(font_scale=2)
plt.rcParams['legend.fontsize'] = 30


def main():
    # reads in primary data file: displaying apartment data
    # from craigslist 1990ish-2018
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
    price_per_bed_df = plot_bed_to_price(clean_home_df)  # save for test

    # QUESTION 2
    all_income = make_income_datasets(path)
    all_income = clean_all_income(all_income)
    av_incomes, av_prices = make_average_prices_incomes(all_income,
                                                        clean_home_df)
    plot_average_incomes_prices(av_incomes, av_prices)

    # QUESTION 3
    pops_shapes, price_by_county = prep_q3_populations(populations_df,
                                                       clean_home_df,
                                                       counties_shapes)
    plot_populations_prices(price_by_county, pops_shapes)

    # TEST FUNCTIONS HERE
    test_Q1(clean_home_df, price_per_bed_df)
    test_Q2(clean_home_df, av_prices, all_income, av_incomes)


def test_Q1(clean_home_df: pd.DataFrame, price_per_bed_df: pd.DataFrame) \
            -> None:
    """
    Parameters:
    - clean_home_df, a DataFrame of 2000-2018 apartments.
    - price_per_bed_df, a . . .
    Behavior: This tests if the averaging isdone properly...
    """
    # the indices are unordered, so loc with the first 5 rows doesn't work.
    # this just gets the price and bed. ppb = price per bed
    df_test_ppb = clean_home_df[clean_home_df['year'] >= 2000].iloc[0:5, 6:8]
    print('print the test ppb just in case: ')
    print(df_test_ppb)
    print()
    test_ratio_1 = df_test_ppb.iloc[0]['price'] / df_test_ppb.iloc[0]['beds']
    orig_ratio_1 = price_per_bed_df.iloc[0]['price/beds']
    print('Comparing price/bed ratio of test vs original: ' +
          str(test_ratio_1) + ' vs ' + str(orig_ratio_1))

    print("Comparing small test divison of series with original. \n" +
          "Empty means they're the same")
    test_ratio_series = df_test_ppb['price'] / df_test_ppb['beds']
    print(test_ratio_series.compare(price_per_bed_df.iloc[0:5]['price/beds'],
                                    result_names=('test', 'original')))


def test_Q2(clean_home_df: pd.DataFrame, average_prices: pd.DataFrame,
            all_income: pd.DataFrame, average_incomes: pd.DataFrame) \
            -> None:
    """
    Parameters:
    - clean_home_df, a DataFrame of 2000-2018 apartments.
    -
    -
    -
    Behavior: This tests if the mean of the prices and incomes are
    calculated as expected.
    """
    # Check the average price.
    print('Assert if my test 2018 average price == original. No output and' +
          ' program continuation means the assertion is true.')
    df_test_av_p = clean_home_df[clean_home_df['year'] == 2018]
    test_price_2018 = df_test_av_p['price'].sum() / len(df_test_av_p) * 12
    # year is index. the only column is the price
    orig_price_2018 = average_prices.iloc[-1]
    print('test: ' + str(test_price_2018))
    print('orig: ' + str(orig_price_2018))
    # print('orig: ' + type(orig_price_2018))
    assert test_price_2018 == orig_price_2018

    """
    # Check average income of all counties for one race.
    print('Check if test average income == original average income')
    is_native_american = all_income['Race'].str.contains('American Indian')
    is_2018 = all_income['Year'] == 2018
    df_test_av_i = all_income[is_native_american & is_2018]
    assert df_test_av_i.iloc[1:7].mean(axis=1) == average_incomes[average_incomes['']]
    """

def test_Q3():
    """

    """


def make_clean_housing_data(home_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Parameter: home_df, a 40 mb DataFrame with Craigslist apartment data
    from 2000-2018.
    Behavior: This function cleans the home dataset from Craigslist
    so it has less scams, hotels, and motels.
    '''
    # FILTER OUT irrelevant beds and baths, with less likely scam prices
    is_not_missing = (home_df['beds'].notna()) & (home_df['baths'].notna())
    # apply it early on to avoid errors
    home_df = home_df[is_not_missing]
    is_not_too_cheap = home_df['price'] > 700

    # filter out hotel/motel data
    no_night = home_df['title'].str.lower().str.contains('.*night.*',
                                                         regex=True)
    no_hotel = home_df['title'].str.lower().str.contains('.*hotel*',
                                                         regex=True)
    no_motel = home_df['title'].str.lower().str.contains('.*motel*',
                                                         regex=True)
    not_motel_hotel = ~(no_night | no_hotel | no_motel)

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


def plot_price_trends(clean_home_df: pd.DataFrame) -> None:
    """
    For QUESTION 1: How have prices of apartments changed over time
    based on their types?

    Parameter: clean_home_df. DataFrame of Craigslist apt data
    but with less entries that are suspiciously cheap or for
    hotels and motels. Goes from 2000-2018.
    Behavior: Makes 12 sub-plots representing how prices have
    changed based on bed and bath amounts.
    """
    plots = sns.relplot(data=clean_home_df, x='year', y='price', col='beds',
                        row='baths', kind='line')
    plots.fig.suptitle('Prices over time for 1-4 beds for 1, 1.5, 2 baths.',
                       size='xx-large')
    plots.fig.subplots_adjust(top=0.85)
    plots.set_axis_labels('Year (2000-2018)', 'Prices (USD)')
    plt.savefig('price_trends_apartment_type.png')


def plot_bed_to_price(clean_home_df: pd.DataFrame) -> pd.Series:
    """
    For QUESTION 1 p2: how have overall price per beds changed over time?

    This function takes in a dataframe representing housing data
    for northern California off craigslist from approximately 2000 to
    2020 and plots a graph and plots price change over time based on a ratio
    that averages price to bed amounts
    """
    # Focuses the year range of our dataset
    data_y2k = clean_home_df['year'] >= 2000
    # Creates a dataFrame that will represent our additional
    # ratio for this question
    price_per_bed_df = clean_home_df[data_y2k].copy()

    # Creates a new column representing a ratio of price changes
    # and bed changes
    price_per_bed_df['price/beds'] = (price_per_bed_df['price'] /
                                      price_per_bed_df['beds'])

    # plot it all
    plots = sns.relplot(data=price_per_bed_df, x='year', y='price/beds',
                        kind='line', height=6, aspect=3)
    plots.set_axis_labels('Year (2000-2018)', 'Prices of Apartments (USD)',
                          fontsize=20)
    plots.set_xticklabels(fontsize=16)
    plots.set_yticklabels(fontsize=16)
    plt.xlim(2000)
    plt.title("Changes in Price Over Time Based on Amount of Beds", y=1.2,
              fontsize=30)
    plt.savefig('bed_to_price.png')
    return price_per_bed_df


def make_income_datasets(path: str) -> pd.DataFrame:
    """
    For QUESTION 2: How does race change the amount of income
    taken up by housing over time?

    Parameter: path, a string representing a path to
    a directory of files representing income information from
    2010-2018 for different races and locations in California.

    Behavior / return: It returns a DataFrame merges all these
    data into one big dataframe.
    """
    income_datasets = []  # build this up

    for filename in os.listdir(path):
        # get file name without csv extension
        file_tokens = filename[0:filename.find('.csv')]
        year = file_tokens[-4:]  # gets year
        dataset = pd.read_csv(path + '/' + filename)
        dataset['Year'] = year  # adds year column

        # make column names shorter and more readable
        dataset.columns = dataset.columns.str.replace('!!', ' ',
                                                      regex=False)
        dataset.columns = dataset.columns.str.replace('California ', '',
                                                      regex=False)
        dataset.columns = dataset.columns.str.replace(' (dollars) Estimate',
                                                      '', regex=False)
        dataset = dataset.dropna()
        income_datasets.append(dataset)

    all_income = income_datasets[0]

    # merge the list of datasets
    for dataset in income_datasets[1:]:
        all_income = pd.concat([all_income, dataset], join='inner', axis=0)

    # contains is used because these values are right-justified with unknown
    # amounts of spaces
    all_income = all_income.loc[:, all_income.columns.str.contains('Median') |
                                all_income.columns.str.contains('Group') |
                                all_income.columns.str.contains('Year')]

    all_income.rename(columns={'Label (Grouping)': 'Race'}, inplace=True)
    return all_income


def make_race_filters(all_income: pd.DataFrame) -> list[pd.Series]:
    """
    For QUESTION 2

    Parameter: all_income, a DataFrame of incomes from 2010-2018 for
    four races.
    Behavior / return: This makes filters for White, Black,
    Native American / Alaskan Native, and Asian people.
    """
    white_filter = all_income['Race'].str.contains("White alone")
    # excluded one letter to fit flake8 and still filters properly
    black_filter = all_income['Race'].str.contains("Black or African America")
    # the original data used "American Indian" instead of "Native American"
    native_filter = all_income['Race'].str.contains("American Indian and " +
                                                    "Alaska Native")
    asian_filter = all_income['Race'].str.contains("Asian")
    race_filters = [white_filter, black_filter, native_filter, asian_filter]
    return race_filters


def clean_all_income(all_income: pd.DataFrame) -> pd.DataFrame:
    """
    For QUESTION 2

    Parameter: all_income, a DataFrame of incomes from 2010-2018 for
    four races.
    Behavior / return: Handle rows with missing data and turn the
    income entries from strings to integers. Return the new all_income.
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
                    'San Mateo County, Median income',
                    'Santa Clara County, Median income',
                    'Sonoma County, Median income']

    # convert medians from strings to integers
    for col in cols_medians:
        all_income[col] = all_income[col].astype('int32')

    return all_income


def make_average_prices_incomes(all_income: pd.DataFrame,
                                clean_home_df: pd.DataFrame) -> \
                                tuple[pd.DataFrame, pd.DataFrame]:
    """
    For QUESTION 2: How does race change the amount of income
    taken up by housing over time?

    Parameters:
    - all_income, a DataFrame of 2010-2018 incomes.
    - clean_home_df, a DataFrame of 2000-2018 apartments.

    Behavior: Calculate average prices and average incomes
    from 2010-2018. Return them as a tuple of DataFrames.
    """
    races_list = ['White', 'Black', 'Native', 'Asian']
    list_df_incomes = []
    race_filters = make_race_filters(all_income)

    # put average income dataframes in a list
    for i in range(0, len(races_list)):
        race_income = all_income[race_filters[i]].sort_values('Year',
                                                              ascending=True)
        race = races_list[i]
        # add '<race> + Average' column, then filter to just that new col.
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

    # Remove years that don't match with income
    average_prices = average_prices[average_prices.index >= 2010]
    return average_incomes, average_prices


def plot_average_incomes_prices(average_incomes: pd.DataFrame,
                                price_avg_2010_2018: pd.DataFrame):
    """
    For QUESTION 2

    Parameters:
    - average_incomes, a DataFrame of average incomes by year and race.
    - price_avg_2010_2018, a DataFrame of average housing prices from
      2010-2018.

    Behavior: Plot the average incomes and prices onto one graph with Plotly.
    This is not saved as a fig. The plot is interactive and opens up in the
    browser.
    """
    my_title = "2010-2018: Average Incomes by Demographics and California" + \
               " 1-4 bed housing prices"

    # average_incomes is already AT 2010 - 2018.
    fig = px.line(average_incomes, x="Year", y=average_incomes.columns[1:],
                  labels={
                     "Year": "Year (2010-2018)",
                     "value": "Amount of USD"
                  }, title=my_title)

    fig = px.line(average_incomes, x="Year", y=average_incomes.columns[1:],
                  width=1000, height=800)

    # add price averages onto plot
    fig.add_scatter(x=price_avg_2010_2018.index, y=price_avg_2010_2018,
                    name='Average Price of Housing')
    plot_title = "2010-2018: Average Incomes by Demographics and " + \
                 "1-4 bed housing prices in California"
    fig.update_layout(
        title=plot_title,
        xaxis_title='Year (2010-2018)',
        yaxis_title='Amount of USD',
        legend_title='Incomes and Pricing',
        margin=dict(l=10, r=10, t=80, b=10),
        font=dict(size=16)
    )
    fig.show()


def prep_q3_populations(populations_df: pd.DataFrame,
                        clean_home_df: pd.DataFrame,
                        counties_shapes: pd.DataFrame) -> \
                        tuple[pd.DataFrame, pd.DataFrame]:
    """
    For QUESTION 3: How does race change the impact of housing prices on
    income based on location?

    Parameters:
    - populations_df, a DataFrame of 2018 populations with many demographics.
    - clean_home_df, a DataFrame of 2000-2018 apartments.
    - counties_shapes, a DataFrame with the shapes of counties boundaries.

    Behavior: This prepares population data to shade in the geographical plot,
    and the average housing prices by county. It returns a tuple of
    2 DataFrames. One is the merged form of the populations and
    counties_shapes, as well as average prices by county in 2018.
    """
    # Clarifies data to only relevant counties (those with most information)
    counties = ['San Francisco', 'Santa Clara', 'San Mateo', 'Alameda',
                'Contra Costa', 'Sonoma']
    # this selects the 4 races
    populations_df = populations_df.iloc[4:8, :]
    populations_df = populations_df.transpose()

    populations_df.columns = populations_df.iloc[0]
    populations_df = populations_df.tail(-1)
    populations_df.reset_index(inplace=True, drop=False)

    # normalize county names in index column
    populations_df.columns = populations_df.columns.str.lstrip()
    suffix = 'County, California'
    populations_df['index'] = populations_df['index'].str.replace(suffix, '',
                                                                  regex=False)

    populations_df['index'] = populations_df['index'].str.lower().str.strip()

    # Filters price dataframe to only use 2018 prices
    average_2018_prices = clean_home_df[clean_home_df['year'] == 2018]

    # Groups the 2018 county prices into a mean and creates a dataframe with
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
    For QUESTION 3

    Parameters:
    - price_by_county, a DataFrame of
    - pops_shapes, a DataFrame of populations by race and with county shapes.
    Behavior: Plot 4 geographical plots showing population in
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
                     legend=True, cmap='Purples', edgecolor='green',
                     linewidth=3)
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

    fig.suptitle('2018 Populations of  San Francisco, Santa Clara,' +
                 'San Mateo, Alameda, Contra Costa, and Sonoma by Race',
                 fontsize=35)
    plt.savefig('Q3_populations_prices.png')


if __name__ == '__main__':
    main()
