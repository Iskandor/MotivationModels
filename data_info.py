from plots.dataloader import load_data2, prepare_data
import numpy as np
import pandas as pd
from glob import glob


res = {
    "name": "name",
    "re_median": "re_median",
    "re_90percentile": "re_90percentile",
    "re_std_dev": "re_std_dev",
    "re_mean": "re_mean",
    "norm_error_variance": "norm_error_variance",
    "norm_error_std_dev": "norm_error_std_dev",
    "norm_error_mean": "norm_error_mean",
    "norm_error_90percentile": "norm_error_90percentile",
    "relative_var_1": "relative_var_1"
    # "ri_variance": "ri_variance",
    # "ri_std_dev": "ri_std_dev"
}
df = pd.DataFrame(res, index=[0])
df.to_csv('hodnoty.csv', mode='a', header=False)



for path in glob("Q:\\Desktop\\skola MGR\\diplomka\\final_models\\*\\"):
    config = [
        {'env': 'gravitar', 'algorithm': 'ppo', 'model': 'rnd', 'id': '1'}
    ]

    data = []
    data.append(load_data2(path))

    #data = load_data2(path)
    #dict_keys(['steps', 'score', 're', 'ri', 'error'])


    all_data_re = []
    all_data_error = []
    all_data_ri = []


    for i in range(len(data[0]['re'])):
        all_data_re = np.append(all_data_re, data[0]['re'][i])
        all_data_error = np.append(all_data_error, data[0]['error'][i] / (data[0]['steps'][i] + 1))
        all_data_ri = np.append(all_data_ri, data[0]['ri'][i])

    normalized_error = (all_data_error - np.mean(all_data_error)) / np.std(all_data_error)


    print(all_data_error.shape)
    print(all_data_error)

    #meno folderu
    #path Q:\Desktop\skola MGR\diplomka\final_models\baseline - l2\
    i = path.rfind("\\", 0, -1)
    name = path[i+1:-1]


    res = {
        "name": name,
        "re_median": np.median(all_data_re),
        "re_90percentile": np.percentile(all_data_re, 90),
        "re_std_dev": np.std(all_data_re),
        "re_mean": np.format_float_scientific(np.mean(all_data_re)),
        "norm_error_variance": np.var(normalized_error),
        "norm_error_std_dev": np.std(normalized_error),
        "norm_error_mean" : np.mean(normalized_error),
        "norm_error_90percentile": np.percentile(normalized_error, 90),
        "relative_var_1" : np.format_float_scientific(np.var(all_data_error) / np.mean(all_data_error)),
        "var":  np.var(all_data_error),
        "mean": np.mean(all_data_error)
        # "ri_variance": np.format_float_scientific(np.var(all_data_ri)),
        # "ri_std_dev": np.format_float_scientific(np.std(all_data_ri))
    }


    df = pd.DataFrame(res, index=[0])
    df.to_csv('hodnoty.csv', mode='a', header=False)
