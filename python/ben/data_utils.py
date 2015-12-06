import pandas as pd
import numpy as np
import csv
import datetime
import copy


def convert_to_date(year, month):
    try:
        return datetime.date(int(year), int(month), day=15)
    except ValueError:
        return datetime.date(1900, 1, 1)


def date_from_year_week(year, week):
    try:
        week_str = "%s-%s" % (int(year), int(week))
        return datetime.datetime.strptime(week_str + '-1', "%Y-%W-%w")
    except ValueError:
        return None


def transform_data(sales, stores):
    stores['CompetitionDistance'] = stores.CompetitionDistance.fillna(100000)
    # stores['NearCompetition'] = np.where(stores['CompetitionDistance'] <= 250, 1, 0)
    # stores['MidCompetition'] = np.where(stores['CompetitionDistance'] <= 500, 1, 0)
    # stores['FarCompetition'] = np.where(stores['CompetitionDistance'] <= 750, 1, 0)
    # stores['VeryFarCompetition'] = np.where(stores['CompetitionDistance'] > 750, 1, 0)
    stores['CompetitionOpenSinceYear'] = stores.CompetitionOpenSinceYear.fillna(2050)
    stores['CompetitionOpenSinceMonth'] = stores.CompetitionOpenSinceMonth.fillna(1)
    stores['CompetitionOpenSince'] = pd.to_datetime(stores[['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth']].apply(lambda s: convert_to_date(s[0], s[1]),axis = 1))
    # stores['Promo2Since'] = pd.to_datetime(stores[['Promo2SinceYear', 'Promo2SinceWeek']].apply(lambda s: date_from_year_week(s[0], s[1]),axis = 1))
    sales['Date'] = pd.to_datetime(sales.Date)
    # if 'Sales' in sales.columns:
    #     sales = sales[sales.Sales>0]
        # sales['Sales'] = sales.Sales
    sales = sales.sort_values('Date')
    # sales['Year'] = sales.Date.dt.year - 2013
    # sales['Month'] = sales.Date.dt.month
    sales['Week'] = sales['Date'].dt.week
    # sales['DayOfYear'] = sales['Date'].dt.dayofyear
    # sales['DaySinceStart'] = sales['Year'] * 365 + sales['DayOfYear']
    # sales['Seasonal_4_sin'] = np.sin(sales.DayOfYear/365*4*2*np.pi)
    # sales['Seasonal_4_cos'] = np.cos(sales.DayOfYear/365*4*2*np.pi)
    # sales['Seasonal_3_sin'] = np.sin(sales.DayOfYear/365*3*2*np.pi)
    # sales['Seasonal_3_cos'] = np.cos(sales.DayOfYear/365*3*2*np.pi)
    if 'Customers' in sales.columns:
        sales = sales.drop(['Customers'], axis=1)
    sales['StateHoliday'].loc[sales['StateHoliday'] == 0] = '0'
    sales = pd.get_dummies(sales, columns=['StateHoliday', 'DayOfWeek'])
    stores = stores.drop(['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'CompetitionDistance'], axis=1)
    stores = pd.get_dummies(stores, columns=['StoreType', 'Assortment'])
    all_data = pd.merge(sales, stores, on='Store')
    all_data['PostComp'] = (all_data['Date'] > all_data['CompetitionOpenSince'])
    # all_data['PromoInterval'] = all_data.PromoInterval.fillna(0)
    all_data['Open'] = all_data.Open.fillna(1)
#    all_data = all_data.fillna(0)
    #construct if it's promo2 start month
    # month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', \
    #          7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    # all_data['monthStr'] = all_data.Month.map(month2str)
    # all_data.loc[all_data.PromoInterval == 0, 'PromoInterval'] = ''
    # all_data['IsPromo2Month'] = 0
    # for interval in all_data.PromoInterval.unique():
    #     if interval != '':
    #         for month in interval.split(','):
    #             all_data.loc[(all_data.monthStr == month) & (all_data.PromoInterval == interval), 'IsPromo2Month'] = 1
    # all_data.loc[(all_data.Promo2Since>all_data.Date), 'IsPromo2Month'] = 0

    #all_data = pd.get_dummies(all_data, columns=['Month'])
    all_data = all_data.drop([
            'Promo2',
            # 'Promo2Since',
            'Promo2SinceWeek',
            'Promo2SinceYear',
            'PromoInterval',
    #         'monthStr',
            'CompetitionOpenSince',
            'Date',
    #         'Year',
    #         'Week',
    #         'DayOfYear'
        ], axis=1)
    return all_data


def calc_store_sales_distributions(all_data):
    stores_mean_post = all_data[all_data.PostComp==True][['Store', 'Sales']].groupby('Store').mean()
    stores_mean_pre = all_data[all_data.PostComp==False][['Store', 'Sales']].groupby('Store').mean()
    stores_mean_post['Store'] = stores_mean_post.index
    stores_mean_pre['Store'] = stores_mean_pre.index
    stores_mean_post = stores_mean_post.rename(columns={'Sales': 'Sales_mean_post'})
    stores_mean_pre = stores_mean_pre.rename(columns={'Sales': 'Sales_mean_pre'})
    stores_mean_post['PostComp'] = True
    stores_mean_pre['PostComp'] = False

    stores_std_post = all_data[all_data.PostComp==True][['Store', 'Sales']].groupby('Store').std()
    stores_std_pre = all_data[all_data.PostComp==False][['Store', 'Sales']].groupby('Store').std()
    stores_std_post['Store'] = stores_std_post.index
    stores_std_pre['Store'] = stores_std_pre.index
    stores_std_post = stores_std_post.rename(columns={'Sales': 'Sales_std_post'})
    stores_std_pre = stores_std_pre.rename(columns={'Sales': 'Sales_std_pre'})
    stores_std_post['PostComp'] = True
    stores_std_pre['PostComp'] = False

    results_mean = pd.concat([stores_mean_post, stores_mean_pre], axis=0)
    results_std = pd.concat([stores_std_post, stores_std_pre], axis=0)

    results = pd.merge(results_mean, results_std, on=['Store', 'PostComp'])

    #fill missing pre/post competition values with distribution values from the other
    fillers = []
    for row in results.iterrows():
        store = row[1]['Store']
        if len(results[results.Store==store]) == 1:
            new_series = pd.Series(copy.deepcopy(row[1]))
            new_series['PostComp'] = not new_series['PostComp']
            new_series['Sales_mean_post'] = row[1]['Sales_mean_pre']
            new_series['Sales_mean_pre'] = row[1]['Sales_mean_post']
            new_series['Sales_std_post'] = row[1]['Sales_std_pre']
            new_series['Sales_std_pre'] = row[1]['Sales_std_post']
            fillers.append(new_series)
    results = pd.concat([results, pd.DataFrame(fillers)], axis=0)

    return results


def merge_sales_with_distributions(all_data, dist):
    all_data = pd.merge(all_data, dist, how='left', on=['Store', 'PostComp'])
    all_data['Sales_mean'] = all_data[['Sales_mean_post', 'Sales_mean_pre']].sum(axis=1)
    all_data['Sales_std'] = all_data[['Sales_std_post', 'Sales_std_pre']].sum(axis=1)
    all_data['PostComp'] = all_data['PostComp'].astype(int)
    # all_data['HugeStore'] = np.where(all_data['Sales_mean'] > 9000, 1, 0)
    # all_data['SmallStore'] = np.where(all_data['Sales_mean'] < 5000, 1, 0)
    all_data = all_data.drop(['Sales_mean_post', 'Sales_mean_pre', 'Sales_std_post', 'Sales_std_pre'], axis=1)
    if 'Sales' in all_data.columns:
        all_data['Sales'] = (all_data.Sales - all_data.Sales_mean) / all_data.Sales_std
    return all_data


def load_transformed_data():
    # load training and test set
    train = pd.read_csv('../../data/train.csv')
    test = pd.read_csv('../../data/test.csv')
    store = pd.read_csv('../../data/store.csv')

    # transform training data
    all_data = transform_data(train, store)
    #all_data = all_data[all_data.Open==1]  #get rid of all closed days
    store_sales_distributions = calc_store_sales_distributions(all_data)
    all_data = merge_sales_with_distributions(all_data, store_sales_distributions)

    # transform test set
    transformed_test = transform_data(test, store)
    transformed_test = merge_sales_with_distributions(transformed_test, store_sales_distributions)
    test_ids = transformed_test.Id
    transformed_test = transformed_test.reindex_axis(all_data.columns, axis='columns', fill_value=0)
    transformed_test = pd.concat([test_ids, transformed_test], axis=1)
    transformed_test = transformed_test.sort_values('Id')

    return all_data, transformed_test


def get_raw_values(dataframe):
    X = dataframe.drop(['Sales', 'Store', 'Sales_mean', 'Sales_std'], axis=1).values
    y = dataframe['Sales'].values
    return X, y

