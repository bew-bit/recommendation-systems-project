import numpy as np

def hit_rate(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    return (flags.sum() > 0) * 1

def hit_rate_at_k(recommended_list, bought_list, k=5):
    return hit_rate(recommended_list[:k], bought_list)

def precision(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    return flags.sum() / len(recommended_list)

def precision_at_k(recommended_list, bought_list, k=5):
    return precision(recommended_list[:k], bought_list)


def recall_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    bought_list = bought_list
    recommended_list = recommended_list[:k]
    flags = np.isin(bought_list, recommended_list)
    recall = flags.sum() / len(bought_list)
    return recall