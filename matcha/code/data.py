from pathlib import Path
import json

from code.utils import X_START, X_END, Y_START, Y_END
import pandas as pd
from tqdm import tqdm
import numpy as np

def make_gt(chart_type, xs ,ys):
    
    x_str = X_START + ";".join(list(map(str, xs))) + X_END
    y_str = Y_START + ";".join(list(map(str, ys))) + Y_END

    ground_truth = '<' + chart_type + '>' + x_str + y_str

    ground_truth = ground_truth.replace('\n', ' ')

    return ground_truth

def round_float(value, decimals):

    if decimals < 0:
        decimals = 0

    format_str = '{:.' + str(decimals) + 'f}'
    val = format_str.format(value)

    # avoid -0.0 and 0.0000
    if float(val) == 0:
        return '0'

    return val

def is_nan(value):
    return isinstance(value, float) and str(value) == "nan"

def fix_int(x):
    if 'E+' in x[1:3]:
        return str(int(x[0]) * int(10**int(x[3:])))
    return x

def parse_numeric_ticks(vals):
    if any(['.' in str(v) for v in vals]):
        is_float = True
    else:
        is_float = False

    if is_float:
        vals = [float(v.replace(',','').replace('$','').replace('%','')) for v in vals]
    else:
        vals = [int(fix_int(v).replace(',','').replace('$','').replace('%','').replace(' ', '')) for v in vals]

    return vals

def calc_decimals(diff):
    return round(np.log10(1/diff)) + 3

def get_decimals(name, data, axis):

    text = {t['id']:t['text'] for t in data['text']}

    if axis == 'x-axis':
        if not data['chart-type'] in {'horizontal_bar', 'scatter'}:
            return None
        
        vals = [text[x['id']] for x in data['axes']['x-axis']['ticks']]

    elif axis == 'y-axis':
        if not data['chart-type'] in {'vertical_bar', 'histogram', 'scatter', 'line'}:
            return None
        
        vals = [text[y['id']] for y in data['axes']['y-axis']['ticks']]

        if name in ['345d26728a16', '7610fe70741b', '9402690e658c', 'e6ccc2703066']:
            vals = [v.replace('.', '') for v in vals]

        if name == '43f6a39f9750':
            vals = ['0', '100']

    vals = parse_numeric_ticks(vals)
    diff = max(vals) - min(vals)

    return calc_decimals(diff)

def preparte_gt(name, data, xs, ys):
    x_decimals = get_decimals(name, data, 'x-axis')
    y_decimals = get_decimals(name, data, 'y-axis')

    if not x_decimals is None:
        xs = [round_float(x, x_decimals) for x in xs]
    
    if not y_decimals is None:
        ys = [round_float(y, y_decimals) for y in ys]
    
    return xs, ys

def preparte_gt_without_ticks(vals):
    vals = [float(v) for v in vals]
    diff = max(vals) - min(vals)

    if diff == 0:
        return None
    
    decimals = calc_decimals(diff)

    vals = [round_float(v, decimals) for v in vals]

    return vals

def parse_annotations(filepath):

    filepath = Path(filepath)

    with open(filepath) as fp:
        data = json.load(fp)

    data_series = data["data-series"]
    xs, ys = [], []

    for d in data_series:
        xs.append(d["x"])

        # DON'T ADD NANs FOR HISTOGRAMS
        if not is_nan(d["y"]):
            ys.append(d["y"])
        else:
            data["chart-type"] = 'histogram'

    try:
        gt_xs, gt_ys = preparte_gt(filepath.stem, data, xs, ys)
    except:
        text = {t['id']:t['text'] for t in data['text']}
        print(filepath.stem, data['source'])
        print([text[x['id']] for x in data['axes']['x-axis']['ticks']])
        print([text[y['id']] for y in data['axes']['y-axis']['ticks']])
        gt_xs, gt_ys = xs, ys

    gt_string = make_gt(data['chart-type'], gt_xs, gt_ys)

    return {
        "ground_truth": gt_string,
        "x": xs,
        "y": ys,
        "chart-type": data["chart-type"],
        "id": filepath.stem,
        "source": data["source"],
    }

skip_wrong_ys = ['205971b3d411', '29a1afda7f60', '3288e7588c52', '41b5ad55fee2', '42e5ab14f4f1', '48189fe4c286', '4cfea6c541e9', '583eabadb51e', '750c3876563f', '7a46ec28cc2c', '835d25a0429f', '84b1b6aabe40', '85fc679ccf3c', '8861dde2ddc9', '8898e9907388', '944014b8ce8c', 'a3550cf6248e', 'a6aa31075bd8', 'aef80c8e09dd', 'b56e10126c31', 'bc4558e4d187', 'c72446dac578', 'd75a8f2a99db', 'd81a81b734d9', 'decb5cca4428', 'df3854b0bbba', 'ef141e2822d7', 'ff721b3ae2eb']
no_xs_or_ys = ['04296b42ba61', '3968efe9cbfc', '496de625e57a', '6a92d147f4d5', '6ce4bc728dd5', '733b9b19e09a', '89d24be7fcb0', 'aa9df520a5f2', 'd0cf883b1e13', 'e2ee063cb374']
extra_drop = ['43f6a39f9750', ]

def generate_train_dataset():

    data_dir = Path("../data/benetech-making-graphs-accessible/train")

    image_folder = data_dir / "images"

    json_paths = sorted((data_dir / "annotations").glob("*.json"))

    ids, image_paths, ground_truths, chart_types, sources, xs, ys = [], [], [], [], [], [], []

    for json_path in tqdm(json_paths):
        
        # histogram with missing x-tick
        if json_path.stem == '74220cf817a2':
            continue
        
        if json_path.stem in skip_wrong_ys or json_path.stem in no_xs_or_ys:
            continue

        ids.append(json_path.stem)
        image_paths.append(str(image_folder / f"{json_path.stem}.jpg"))

        parsed = parse_annotations(json_path)

        ground_truths.append(parsed['ground_truth'])
        chart_types.append(parsed['chart-type'])
        sources.append(parsed['source'])

        xs.append(parsed['x'])
        ys.append(parsed['y'])
    
    df = pd.DataFrame(zip(ids, image_paths, ground_truths, chart_types, sources), columns=['id', 'image_path', 'ground_truth', 'chart-type', 'source'])

    # drop scatter
    df = df[~df['chart-type'].isin(['scatter'])]

    train_df = df[df['source'] == 'generated'].reset_index(drop=True).sort_values(by='id')
    valid_df = df[df['source'] == 'extracted'].reset_index(drop=True).sort_values(by='id')

    index = [f"{id_}_x" for id_ in ids] + [f"{id_}_y" for id_ in ids]

    gt_df = pd.DataFrame(
        index=index,
        data={
            "data_series": xs + ys,
            "chart_type": chart_types * 2,
        },
    )
    
    gt_df['id'] = gt_df.index.map(lambda s: s.split('_')[0])
    gt_df = gt_df[gt_df['id'].isin(valid_df['id'])]
    gt_df = gt_df.drop(columns='id')

    return train_df, valid_df, gt_df

def add_generated_data(train_df):

    cnt = 15000

    chart_types = ['horizontal_bar'] * cnt
    chart_types += ['vertical_bar'] * cnt
    chart_types += ['line'] * cnt
    chart_types += ['histogram'] * cnt
    sources = ['my_generated'] * (4*cnt)
    placeholder = [None] * (4*cnt)

    df = pd.DataFrame(zip(placeholder, placeholder, placeholder, chart_types, sources), columns=train_df.columns)

    train_df = pd.concat([train_df, df]).reset_index(drop=True)

    return train_df