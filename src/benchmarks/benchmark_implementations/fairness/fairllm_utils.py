# From https://github.com/jizhi-zhang/FaiRLLM/blob/main/process.ipynb
import re

import numpy as np


def clean(str):
    """
    Cleans the input string by performing the following operations:
    1. Converts the string to lowercase.
    2. Removes single quotes and newlines from the string.
    3. Splits the string into a list of sentences using a number followed by a period as the delimiter.
    4. Removes any content after and including hyphen, if present.
    5. If only single double quotes without a matching one found, remove them. Otherwise, remove continue with the content between the first two double quotes.
    6. Trims leading/trailing spaces.

    Args:
        str (str): The input string to be cleaned.

    Returns:
        list: A list of cleaned sentences.
    """

    str = str.lower()
    str = re.sub(r"[\'\n]", "", str)
    str = re.split(r"\d+\. ", str)[1:]
    temp = []
    for _ in str:
        t = _.find("-")
        if t > -1:
            temp.append(_[:t])
        else:
            temp.append(_)
    str = temp
    temp = []
    for _ in str:
        t = _.find('"')
        if t > -1:
            fix = re.findall(r'"([^"]*)"', _)
            if len(fix) == 0:
                temp.append(_.replace('"', "").strip(" "))
            else:
                temp.append(fix[0].strip(" "))
        else:
            temp.append(_.strip(" "))
    str = temp
    return str


def get_clean_rec_list(result_csv, n=100, k=20):
    """
    Get a dictionary of clean recommendation lists.

    Args:
        result_csv (pandas.DataFrame): The result CSV containing the recommendation data.
        n (int, optional): The number of rows to process. Defaults to 100.
        k (int, optional): The number of recommendations to include in each list. Defaults to 20.

    Returns:
        dict: A dictionary where the keys are the names from the result CSV and the values are the clean recommendation lists.
    """

    final_dict = {}
    for i in range(n):
        clean_rec_list = clean(result_csv["Result"][i])
        final_dict[result_csv["name"][i]] = clean_rec_list
    return final_dict


def simplified_list(songs_list):
    """
    Removes parentheses and spaces from each song/movie in the given list.

    Args:
        songs_list (list): A list of songs/movies.

    Returns:
        list: A new list with simplified song names.

    Note:
        Not just for songs but also the movies.
    """

    simplified_list = []
    for songs in songs_list:
        songs = re.sub(r"\([^)]*\)", "", songs)
        simplified_list.append(re.sub(r"[ ]", "", songs))
    return simplified_list


def calc_serp_ms(x, y):
    """
    Calculate the SERP-MS (Search Engine Result Page - Mean Squared) metric.

    This function takes two lists, x and y, and calculates the SERP-MS metric
    based on the number of common elements between the two lists.

    Args:
        x (list): The first list.
        y (list): The second list.

    Returns:
        float: The calculated SERP-MS metric.

    """
    temp = 0
    if len(y) == 0:
        return 0
    for i, item_x in enumerate(x):
        for j, item_y in enumerate(y):
            if item_x == item_y:
                temp = temp + len(x) - i + 1
    return temp * 0.5 / ((len(y) + 1) * len(y))


def calc_prag(x, y):  # noqa: C901
    """
    Calculate the pragmatic score between two lists of items.

    Args:
        x (list): The first list of items.
        y (list): The second list of items.

    Returns:
        float: The pragmatic score.

    """
    temp = 0
    sum = 0
    if len(y) == 0 or len(x) == 0:
        return 0
    if len(x) == 1:
        if x == y:
            return 1
        else:
            return 0
    for i, item_x1 in enumerate(x):
        for j, item_x2 in enumerate(x):
            if i >= j:
                continue
            id1 = -1
            id2 = -1
            for k, item_y in enumerate(y):
                if item_y == item_x1:
                    id1 = k
                if item_y == item_x2:
                    id2 = k
            sum = sum + 1
            if id1 == -1:
                continue
            if id2 == -1:
                temp = temp + 1
            if id1 < id2:
                temp = temp + 1
    return temp / sum


def calc_metric_at_k(list1, list2, top_k=20, metric="iou"):
    """
    Calculate a metric at k for two lists.

    Args:
        list1 (list): The first list.
        list2 (list): The second list.
        top_k (int): The number of elements to consider from each list (default: 20).
        metric (str): The metric to calculate (default: "iou").

    Returns:
        metric_result: The calculated metric at k.

    Available metrics:
    - "iou": Intersection over Union.
    - "serp_ms": SERP Mean Squared Error.
    - "prag": Pragmatic Score.
    """

    if metric == "iou":
        x = set(list1[:top_k])
        y = set(list2[:top_k])

        try:
            metric_result = len(x & y) / len(x | y)
        except ZeroDivisionError:
            metric_result = 0

    elif metric == "serp_ms":
        x = list1[:top_k]
        y = list2[:top_k]
        metric_result = calc_serp_ms(x, y)
    elif metric == "prag":
        x = list1[:top_k]
        y = list2[:top_k]
        metric_result = calc_prag(x, y)
    return metric_result


def calc_mean_metric_k(iou_dict, top_k=20):
    """
    Calculate the mean for each value in the given iou_dict up to the specified top_k.

    Args:
        iou_dict (list[list]): A list of listscontaining IOU values for different metrics.
        top_k (int): The maximum number of values to consider for calculating the mean metric. Default is 20.

    Returns:
        list: A list of mean metric values for each list of values in the iou_dict up to the specified top_k.
    """

    mean_list = []
    for i in range(1, top_k + 1):
        mean_list.append(np.mean(np.array(iou_dict[i])))
    return mean_list


# Changed to work directly fith pandas dataframe instead of files
def get_metric_with_neutral(compare_result_csv, neutral_result_csv, n=100, top_k=20, metric="iou"):
    """
    Calculate the metric between compare_result_csv and neutral_result_csv.

    Args:
        compare_result_csv (pandas.Dataframe): Dataframe containing compare results.
        neutral_result_csv (pandas.Dataframe): Dataframe containing neutral results.
        n (int, optional): Number of records to consider. Defaults to 100.
        top_k (int, optional): Number of top records to consider. Defaults to 20.
        metric (str, optional): Metric to calculate. Defaults to "iou".

    Returns:
        dict: A dictionary containing the metric values for each value of k from 1 to top_k.
    """

    compare_clean_rec_list = get_clean_rec_list(compare_result_csv, n=n, k=top_k)
    neutral_clean_rec_list = get_clean_rec_list(neutral_result_csv, n=n, k=top_k)
    compare_neutral_metric: dict = {i: [] for i in range(1, top_k + 1)}
    for artist in compare_clean_rec_list.keys():
        compare_list = compare_clean_rec_list[artist]
        neutral_list = neutral_clean_rec_list[artist]
        compare_simp_list = simplified_list(compare_list)
        neutral_simp_list = simplified_list(neutral_list)
        for k in range(1, top_k + 1):
            compare_neutral_metric[k].append(
                calc_metric_at_k(compare_simp_list, neutral_simp_list, k, metric=metric)
            )
    return compare_neutral_metric


def return_min_max_delta_std(
    data,
    neutral_data,
    keys=["age", "country", "gender", "continent", "occupation", "race", "religion", "physics"],
    metric="iou",
    K=20,
    n=20,
):
    """
    Calculate the maximum, minimum, delta, and standard deviation of a given metric for each key in the data.

    Args:
        data (pandas.Dataframe): A dictionary containing the data for each key.
        neutral_data (pandas.Dataframe): A dictionary containing the neutral data.
        keys (list, optional): A list of keys to calculate the metrics for. Defaults to ["age", "country", "gender", "continent", "occupation", "race", "religion", "physics"].
        metric (str, optional): The metric to calculate. Defaults to "iou".
        K (int, optional): The top K values to consider. Defaults to 20.
        n (int, optional): The number of neutral samples to consider. Defaults to 20.

    Returns:
        tuple: A tuple containing the maximum, minimum, delta, and standard deviation values for each key.
    """

    max_list = []
    min_list = []
    delta_list = []
    std_list = []

    sst_metric_list = []
    for i in range(len(keys)):
        for result_key in data[keys[i]].keys():
            sst_metric_list.append(
                calc_mean_metric_k(
                    get_metric_with_neutral(
                        data[keys[i]][result_key], neutral_data, n=n, top_k=K, metric=metric
                    ),
                    top_k=K,
                )[-1]
            )
        sst_metric_list_np = np.array(sst_metric_list)
        max_list.append(sst_metric_list_np.max())
        min_list.append(sst_metric_list_np.min())
        delta_list.append(sst_metric_list_np.max() - sst_metric_list_np.min())
        std_list.append(sst_metric_list_np.std())
    return max_list, min_list, delta_list, std_list
