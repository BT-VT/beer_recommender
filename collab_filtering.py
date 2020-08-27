# Ben Trono
# CIS 3715
# Final Project
# Spring 2020
# this file contains functions related to a KNN beer recommender using collaborative filtering.

import os
import time

import math
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd

from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise import dump
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split

from fuzzywuzzy import fuzz
import json
import seaborn as sns
import matplotlib.pyplot as plt


"""
import_data()               returns: (df,beer_ids)
get_maps()                  returns: (id2beer, beer2id)
get_beer_matrix()           returns: sparse matrix
get_beer2idx()              returns: dict

build_knn_predictor()       returns: untraineed knn predictor
get_svd_recommender()       returns: trained Surprise! svd recommender

make_knn_recommendation()   returns: list of recommended beers
make_svd_recommendation()   returns: list of tuples: (beer name, estimated rating)
make_single_prediction()    returns: string name of beer predicted, float prediction

get_user_top_beers()        returns: user dataframe sorted by beer ratings
get_beers_not_tried()       returns: numpy list of beer ID's
fuzzy_matching()            returns: index loc. of closest matching beer

"""


# returns a df of beer data ratings and a df of beer ID's to beer names
def import_data():
    # totally numeric, no beer names
    df = pd.read_csv('./Beer_Data/reduced_data_X2.csv')
    # beer names to ID's
    beer_ids = pd.read_csv('./Beer_Data/beer_ids.csv')

    return df, beer_ids


# returns maps of ID's to beer names and beer names to ID's
def get_maps(beer_ids):
    # drop index in beer_ids and build maps
    beer_ids = beer_ids.set_index('beer_id')
    id2beer = beer_ids.to_dict()['beer_full']
    beer2id = {name: beer_id for beer_id, name in id2beer.items()}

    return id2beer, beer2id


# returns a sparse matrix with beer_id rows, user_id columns, user_score values
def get_beer_matrix(df):
    # pivot ratings into movie features
    df_beer_features = df.pivot(
        index='beer_id',
        columns='user_id',
        values='user_score'
    ).fillna(0)
    # convert df of beer features to sparse matrix
    mat_beer_features = csr_matrix(df_beer_features.values)

    return mat_beer_features


# used by make_knn_recommendation
def get_beer2idx():
    """
    :return: a beer to matrix index dictionary
    """
    with open('./Beer_Data/beer2idx.json', 'r') as fp:
        return json.load(fp)


def build_knn_predictor():
    """
    creates a knn recommender
    :param num_rec: number of recommendations that it will return
    :return: a KNN predictor that is not fit
    """
    # build predictor
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)

    return model_knn


def get_svd_recommender(df, test_size=0.25, path="", exists=False):
    """
    builds and trains an SVD recommender
    :param df: a dataframe containing user ID's, beer ID's and ratings
    :param test_size: the fraction of samples that should be reserved for testing
    :param path: the path to an existing svd recommender that was saved to a file
    :param exists: whether or not to upload the algo from a saved file
    :return: trained recommender, list of predictions, and the root mean square error of the recommender
    """
    if exists:
        return dump.load(path)[1]

    # allows surprise to read df
    reader = Reader(rating_scale=(1, 5))
    # must load in particular column order
    data = Dataset.load_from_df(df[['user_id', 'beer_id', 'user_score']], reader)

    trainset, testset = train_test_split(data, test_size=test_size)
    algo = SVD()
    # Train the algorithm on the trainset
    algo.fit(trainset)
    # and predict ratings for the testset. test() returns a list of prediction objects
    # which have several attributes such as est (the prediction) and r_ui (the true rating)
    predictions = algo.test(testset)

    # rmse below 1 is considered low
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)

    return algo, predictions, rmse


def make_knn_recommendation(model_knn, data, fav_beer, n_recommendations=10, mapper=False, verbose=False):
    """
    determines the top recommended beers based on the beer input
    Parameters
    ----------
    :param model_knn: sklearn model, knn model (untrained)
    :param data: [beer,user] matrix
    :param mapper: dict, map beer name to beer index loc in data
    :param fav_beer: str, name of user input beer
    :param n_recommendations: int, top n recommendations
    :param verbose: boolean, display fuzzywuzzy string matching results
    :return: list of top recommended beers
    ------
    list of top n similar beer recommendations
    """

    if not mapper:
        mapper = get_beer2idx()
    if n_recommendations > 1000:
        n_recommendations = 1000

    # fit
    model_knn.fit(data)
    # get input movie index
    print('You have input beer:', fav_beer)
    idx = fuzzy_matching(mapper, fav_beer, verbose=verbose)
    if idx < 0: return False
    # inference
    print('Recommendation system: starting to make inference')
    print('......\n')
    distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations + 1)
    # get list of (raw idx of recommendations, dist from fav_beer) in beer matrix (which is 'data')
    raw_recommends = \
        sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])
    # get reverse mapper, idx to beer name
    reverse_mapper = {v: k for k, v in mapper.items()}
    # print recommendations
    print('Recommendations for {}:'.format(reverse_mapper[raw_recommends[0][0]]))
    for i, (idx, dist) in enumerate(raw_recommends[1:]):
        print('{0}: {1}, with distance of {2}'.format(i + 1, reverse_mapper[idx], round(dist, 3)))
    # return recommendations
    return [reverse_mapper[idx] for idx, dist in raw_recommends[1:]]


def make_svd_recommendation(user, algo, df, num_beers=10):
    """
    determine the best beers to recommend to a user given a trained SVD predictor.
    Requires path './Beer_Data/beer_ids.csv' to exist

    :param user: int, user ID
    :param algo: trained surprise SVD algorithm
    :param df: dataframe of ratings
    :param num_beers: number of beers to recommend
    :return: a list of beer names in sorted order from highest to lowest recommendation
    """

    beer_ids = pd.read_csv('./Beer_Data/beer_ids.csv')
    id2beer, beer2id = get_maps(beer_ids)

    # build a list of beers from the dataset that a user has not tried
    user_df = df.groupby('user_id')
    beers_tried = user_df.get_group(1)['beer_id'].unique()
    not_tried = df[~df['beer_id'].isin(beers_tried)]['beer_id'].unique()
    # get all predictions for the beers a user has not tried
    user_preds = sorted([algo.predict(user, beer) for beer in not_tried], key=lambda pred: pred.est, reverse=True)
    top_beers = [(id2beer[pred.iid], pred.est) for pred in user_preds]
    # print top recommendations
    for rec in enumerate(top_beers[:num_beers]):
        print('%d: %s, with estimation of %2f' % (rec[0], rec[1][0], rec[1][1]))

    return top_beers[:num_beers]


def make_single_prediction(user_id, new_beer, beer_ids, svd_pred):
    """
    prints a predicted rating for a single beer using SVD
    :param user_id: ID of user in dataset
    :param new_beer: string name of beer and/or brewery to predict
    :param beer_ids: beer to ID dataframe imported with import_data() function
    :param svd_pred: a trained predictor returned by get_svd_recommender()
    :return: the name of the predicted beer, and its estimated prediction
    """
    id2beer, beer2id = get_maps(beer_ids)

    beer_id = fuzzy_matching(mapper=beer2id, fav_beer=new_beer, verbose=False)
    if beer_id < 0: return "beer not found", False
    prediction = svd_pred.predict(user_id, beer_id)
    predicted_beer = id2beer[beer_id]
    pred = prediction.est
    print("%s: \t%2f\n" %(predicted_beer, pred))
    return predicted_beer, pred


def get_user_top_beers(user_id, df, num_beers=10):
    """
    gets a users top rated beers by both name and beer id
    :param user_id: int ID representing a user
    :param df: the dataframe containing all beer reviews
    :param num_beers: number of beers to return
    :return: a dataframe of the users top rated beers
    """
    if df[df['user_id'] == user_id]['user_id'].count() < 1:
        print('user not found')
        return

    # grab all users ratings and sort them from highest to lowest
    user = df[df['user_id'] == user_id].sort_values(by='user_score', ascending=False)
    # save only the first num_beers and add their names to the df
    user_top_beers = user.head(num_beers)
    user_top_beers['beer_name'] = [id2beer[beer_id] for beer_id in user_top_beers['beer_id'].values]

    return user_top_beers


def get_beers_not_tried(user_id, df):
    """
    returns a list of beers in the dataset that a user has not tried
    :param user_id: int id of a user
    :param df: dataframe containing user ratings
    :return: a numpy array of beer ID's
    """
    user_df = df.groupby('user_id')
    beers_tried = user_df.get_group(1)['beer_id'].unique()
    not_tried = df[~df['beer_id'].isin(beers_tried)]['beer_id'].unique()
    return not_tried


# used by make_knn_recommendation
def fuzzy_matching(mapper, fav_beer, verbose=True):
    """
    return index loc (int) of the closest matching beer name in dataset compared to fav_beer.
    If no match found, return None.

    Parameters
    ----------
    :param mapper: dict, map beer name to beer index loc in data
    :param fav_beer: str, name of user input beer
    :param verbose: bool, print log if True
    :return: int
    ------
    beer ID of the closest match
    """
    match_tuple = []
    # get match
    for name, idx in mapper.items():
        ratio = fuzz.ratio(name.lower(), fav_beer.lower())
        if ratio >= 60:
            match_tuple.append((name, idx, ratio))
    # sort
    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    if not match_tuple:
        print('Oops! No match is found')
        return -1
    if verbose:
        print('Found possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))
    return match_tuple[0][1]