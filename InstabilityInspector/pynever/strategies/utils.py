import pandas as pd


def return_df_dict(stars_dict: dict):
    data_dict = dict()
    keys = list(stars_dict.keys())

    for key, layer in stars_dict.items():
        if keys[-1] != key:
            lower_label = f"{key}_lower"
            upper_label = f"{key}_upper"

            lower, upper = get_lower_upper(layer.stars)
            data_dict[lower_label] = pd.Series(lower)
            data_dict[upper_label] = pd.Series(upper)

    df = pd.DataFrame(data_dict)
    return df

def get_lower_upper(stars):
    lower_list_of_lists = list()
    upper_list_of_lists = list()

    for star in stars:
        lower = list()
        upper = list()
        for i in range(star.center.shape[0]):
            lb, ub = star.get_bounds(i)
            lower.append(lb)
            upper.append(ub)
        lower_list_of_lists.append(lower)
        upper_list_of_lists.append(upper)

    absolute_lower_bounds = [min(x) for x in zip(*lower_list_of_lists)]
    absolute_upper_bounds = [max(x) for x in zip(*upper_list_of_lists)]

    return absolute_lower_bounds, absolute_upper_bounds
