from tqdm import tqdm
from pathlib import Path
import json
import pickle
from code.data import parse_numeric_ticks, get_decimals, is_nan

data_dir = Path("../data/benetech-making-graphs-accessible/train")
json_paths = sorted((data_dir / "annotations").glob("*.json"))

def fix_int(x):
    if 'E+' in x[1:3]:
        return str(int(x[0]) * int(10**int(x[3:])))
    return x

parsed_data = []
roles = []
skipped = []

from collections import Counter
fix, total = [], []
cnt = 0

skip_wrong_ys = ['205971b3d411', '29a1afda7f60', '3288e7588c52', '41b5ad55fee2', '42e5ab14f4f1', '48189fe4c286', '4cfea6c541e9', '583eabadb51e', '750c3876563f', '7a46ec28cc2c', '835d25a0429f', '84b1b6aabe40', '85fc679ccf3c', '8861dde2ddc9', '8898e9907388', '944014b8ce8c', 'a3550cf6248e', 'a6aa31075bd8', 'aef80c8e09dd', 'b56e10126c31', 'bc4558e4d187', 'c72446dac578', 'd75a8f2a99db', 'd81a81b734d9', 'decb5cca4428', 'df3854b0bbba', 'ef141e2822d7', 'ff721b3ae2eb']
no_xs_or_ys = ['04296b42ba61', '3968efe9cbfc', '496de625e57a', '6a92d147f4d5', '6ce4bc728dd5', '733b9b19e09a', '89d24be7fcb0', 'aa9df520a5f2', 'd0cf883b1e13', 'e2ee063cb374']

def parse_val(v):
    if isinstance(v, str):
        return v.replace('\n', ' ')

    return v

for json_path in tqdm(json_paths):

    with open(json_path) as fp:
        data = json.load(fp)

        source = data['source']
        chart_type = data['chart-type']

        if json_path.stem in skip_wrong_ys or json_path.stem in no_xs_or_ys:
            skipped.append(chart_type)
            continue

        text = {t['id']:t['text'] for t in data['text']}

        chart_title = None

        axis_labels = []

        for t in data['text']:
            if t['role'] == 'chart_title':
                chart_title = t['text']
            elif t['role'] == 'axis_title':
                p = t['polygon']
                x_sum = p['x0'] + p['x1'] + p['x2'] + p['x3']
                axis_labels.append(((x_sum), t['text']))

        axis_labels.sort()
        if len(axis_labels) > 2:
            axis_labels = []

        for t in data['text']:
            roles.append(t['role'])

        x_ticks = [text[x['id']] for x in data['axes']['x-axis']['ticks']]
        y_ticks = [text[y['id']] for y in data['axes']['y-axis']['ticks']]

        xs = [parse_val(d['x']) for d in data['data-series']]
        ys = [parse_val(d['y']) for d in data['data-series']]

        if is_nan(ys[-1]):
            chart_type = 'histogram'

        if chart_type == 'horizontal_bar':
            xs, ys = ys, xs
            x_ticks, y_ticks = y_ticks, x_ticks
            y_ticks.reverse()

        """
        if chart_type != 'scatter' and (len(xs) != len(x_ticks) or any([x != y for x,y in zip(xs, x_ticks)])):
            #if len(set(x_ticks).difference(set(xs))) > 0:
            if chart_type == 'vertical_bar' and len(xs) == len(x_ticks):
                print(json_path.stem, chart_type, source)
                print(xs)
                print(x_ticks)
                print()
                fix.append(json_path.stem)
                cnt +=1
                continue
        """

        if len(y_ticks) == 0 or len(x_ticks) == 0:
            if source == 'extracted':
                print(json_path.stem)
                raise Exception()
            else:
                skipped.append(chart_type)
                fix.append(json_path.stem)
                continue

        if '345d26728a16' in str(json_path) or '7610fe70741b' in str(json_path) or '9402690e658c' in str(json_path) or 'e6ccc2703066' in str(json_path):
            y_fixed = [y.replace('.', '') for y in y_ticks]
            y_ticks_vals = parse_numeric_ticks(y_fixed)
        else:
            y_ticks_vals = parse_numeric_ticks(y_ticks)

        diff = max(y_ticks_vals) - min(y_ticks_vals)
        diff *= 0.2

        diff = max(diff, 0.2 * max(y_ticks_vals))
        
        if source != 'extracted' and (max(ys) > max(y_ticks_vals) + diff or min(ys) + diff < min(y_ticks_vals)):
            print(json_path.stem, source)
            print(ys)
            print(y_ticks_vals)
            print()
            fix.append(json_path.stem)

        if chart_type != 'scatter' and (len(xs) != len(x_ticks) or any([x != y for x,y in zip(xs, x_ticks)])):
            cnt += 1
            continue

        total.append(chart_type)

        x_decimals = get_decimals(json_path.stem, data, 'x-axis')
        y_decimals = get_decimals(json_path.stem, data, 'y-axis')

        if chart_type == 'horizontal_bar':
            x_decimals, y_decimals = y_decimals, x_decimals

        parsed_data.append({
            'id': json_path.stem,
            'source': source,
            'chart_type': chart_type,
            'x_ticks': x_ticks,
            'x_ticks_vals': list(range(len(x_ticks))),
            'y_ticks': y_ticks,
            'y_ticks_vals': y_ticks_vals,
            'xs': xs,
            'xs_vals': list(range(len(xs))),
            'ys': ys,
            'chart_title': chart_title,
            'axis_labels': axis_labels,
            'x_decimals': x_decimals,
            'y_decimals': y_decimals,
            })

pickle.dump(parsed_data, open('parsed_annotations.p', 'wb'))
print(Counter(roles))
print(len(parsed_data))

print('SKIPPED GENERATED:', Counter(skipped))
print('cnt', cnt)
print(fix)

print('TOTAL:', Counter(total))