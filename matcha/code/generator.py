import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pickle
from PIL import Image
import io
import random
import copy
from code.data import make_gt
import numpy as np
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import matplotlib.colors as mcolors
from code.data import round_float
from scipy.interpolate import CubicSpline

class BaseGraphGenerator():
    def __init__(self, chart_types = ['vertical_bar'], allowed_ids=None):
        data = pickle.load(open('parsed_annotations.p', 'rb'))

        if allowed_ids is not None:
            allowed_ids = set(allowed_ids)
            data = [data_sample for data_sample in data if data_sample['chart_type'] in chart_types and data_sample['id'] in allowed_ids]
        else:
            print('ALLOWED IDS IS NONE!')
            data = [data_sample for data_sample in data if data_sample['chart_type'] in chart_types]
        
        #data = [data_sample for data_sample in data if data_sample['chart_type'] in chart_types and data_sample['source'] == 'generated']
        # this drops histograms
        extra_data = [data_sample for data_sample in data if data_sample['chart_type'] in chart_types and data_sample['source'] in ['extracted', 'external'] and len(data_sample['xs']) == len(data_sample['ys'])]
        data = data + extra_data + extra_data

        x_avg, y_avg = 0, 0
        x_cnt, y_cnt = 0, 0

        for data_sample in data:
            axis_labels = data_sample['axis_labels']

            if len(axis_labels) >= 2:
                
                y_avg += axis_labels[0][0]
                y_cnt += 1

                x_avg += axis_labels[1][0]
                x_cnt += 1

        x_avg /= x_cnt
        y_avg /= y_cnt

        self.data = data
        self.x_avg = x_avg
        self.y_avg = y_avg

        self.mpl_styles = ['Solarize_Light2', '_classic_test_patch', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10']
        self.fonts = ['Aclonica', 'Arimo', 'Calligraffitti', 'Cherry Cream Soda', 'Chewy', 'Coming Soon', 'Cousine', 'Crafty Girls', 'DejaVu Sans', 'DejaVu Sans Mono', 'DejaVu Serif', 'Fontdiner Swanky', 'Homemade Apple', 'Irish Grover', 'Just Another Hand', 'Kosugi', 'Kosugi Maru', 'Kranky', 'Maiden Orange', 'Montez', 'Mountains of Christmas', 'Open Sans Hebrew', 'Open Sans Hebrew Condensed', 'Rancho', 'Redressed', 'Roboto', 'Roboto Mono', 'Roboto Slab', 'Rochester', 'STIXGeneral', 'Satisfy', 'Schoolbell', 'Slackey', 'Smokum', 'Special Elite', 'Sunshiney', 'Syncopate', 'Tinos', 'Ultra', 'Unkempt', 'Yellowtail', 'cmb10', 'cmr10', 'cmss10', 'cmtt10']
        self.patterns = [None, None, None, None, '/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*', '//', '\\\\', '||', '--', '++', 'xx', 'oo', 'OO', '..', '**', '/o', '\\|', '|*', '-\\', '+o', 'x*', 'o-', 'O|', 'O.', '*-']
        self.colors = [c for c in mcolors.CSS4_COLORS.keys() if c not in {'whitesmoke', 'white', 'w', 'snow', 'mistyrose', 'seashell', 'linen', 'bisque', 'oldlace', 'floralwhite', 'cornsilk', 'lemonchiffon', 'ivory', 'belge', 'lightyellow', 'lightgoldenrodyellow', 'honeydew', 'mintcream', 'azure', 'aliceblue', 'ghostwhite', 'lavander', 'lavanderblush'}]

    def get_number_of_decimals(self, y):
        if not '.' in y:
            return 0
    
        return len(y.split('.')[-1])

    def get_max_decimals(self, ys):
        return max([self.get_number_of_decimals(y) for y in ys])

    def is_numerical(self, xs):
        try:
            xs = [float(x) for x in xs]
            return True
        except:
            return False

    def is_int_numerical(self, xs):
        try:
            xs = [int(x) for x in xs]
            return True
        except:
            return False

    def is_same_dist_numerical(self, xs):
        try:
            xs = [int(x) for x in xs]
            dist = np.diff(xs)

            if np.all(dist == dist[0]):
                return True
            else:
                return False
        except:
            pass
        try:
            xs = [float(x) for x in xs]
            dist = np.diff(xs)

            if np.all(dist == dist[0]):
                return True
            else:
                return False
        except:
            pass

        return False

    def generate_random_colors(self, count=1, low=40, high=235):

        if random.random() < 0.3:
            max_colors = 1
        else:
            max_colors = min(len(self.colors), round(count ** random.random()))

        if random.random() < 0.7:
            colors = random.sample(self.colors, k=max_colors)
        else:
            colors = []
            for _ in range(max_colors):

                red = random.randint(low, high) / 255
                green = random.randint(low, high) / 255
                blue = random.randint(low, high) / 255

                colors.append((red, green, blue))

        return random.choices(colors, k=count)
    
    def generate_edge_color(self, low=40, high=235):

        if random.random() < 0.2:
            return 'none'

        if random.random() < 0.7:
            return random.choice(list(mcolors.CSS4_COLORS.keys()))

        red = random.randint(low, high) / 255
        green = random.randint(low, high) / 255
        blue = random.randint(low, high) / 255
        alpha = random.random()

        return (red, green, blue, alpha)

    def generate_random_patterns(self, count=1):

        if random.random() < 0.3:
            max_patterns = 1
        else:
            max_patterns = min(len(self.patterns), round(count ** random.random()))

        patterns = random.sample(self.patterns, max_patterns)
        return random.choices(patterns, k=count)

    def set_random_style(self):
        pass
    
    def get_x_alignment(self, rotation):
        if rotation == 0 or rotation == 90:
            return {
                'ha': 'center',
                'va': 'top',
            }
        
        elif rotation > 0:
            return {
                'ha': 'right',
                'va': 'top',
                'rotation_mode': 'anchor',
            }
        
        else:
            return {
                'ha': 'left',
                'va': 'top',
                'rotation_mode': 'anchor',
            }
        
    def get_y_alignment(self, rotation):
        if rotation == 0:
            return {
                'ha': 'right',
                'va': 'center',
            }
        
        else:
            return {
                'ha': 'right',
                'va': 'center_baseline',
            }
            

    def rotate_ticks(self, x_ticks, y_ticks):
        x_rotation, y_rotation = 0, 0
        # x-ticks rotate
        x_ticks_sum = len(''.join(x_ticks))

        if random.random() < 0.07:
            x_rotation = 90
        else:
            x_rotation = max(0, int((x_ticks_sum-35)*1.4))

            if x_rotation == 0 and random.random() < 0.3:
                x_rotation += random.randint(1, 10)
                x_rotation *=  random.choice([-1,1])

            elif x_rotation > 0 and random.random() < 0.7:
                x_rotation += random.randint(1, 10)
                x_rotation *=  random.choice([-1,1])

        if abs(x_rotation) <= 5:
            x_rotation = 0

        if x_rotation != 0:    
            if x_rotation < -30:
                x_rotation *= -1
            
            if x_rotation >= 60:
                if x_rotation <= 80:
                    x_rotation = random.randint(60, 70)
                else:
                    x_rotation = 90

        # y-ticks rotate
        if random.random() < 0.2:
            y_rotation = random.randint(5,30)
        
        if random.random() < 0.25 and len(''.join(y_ticks)) <= 25:
            y_rotation = 90
        
        plt.xticks(rotation = x_rotation, **self.get_x_alignment(x_rotation))
        plt.yticks(rotation = y_rotation, **self.get_y_alignment(y_rotation))

    def set_chart_axes_titles(self, chart_title, axis_labels, is_multiline=False):

        ax = plt.gca()    

        if chart_title:
            if len(chart_title) > 100:
                ct = chart_title.split(' ')
                ct = ' '.join(ct[:len(ct)//2]) + '\n' + ' '.join(ct[len(ct)//2:])
                chart_title = ct
            min_font, max_font = 8, 17
            if is_multiline:
                min_font, max_font = 8, 13
            ax.set_title(chart_title, fontdict={'fontsize': random.randint(min_font, max_font)}, pad=random.randint(10,20))

        if len(axis_labels) == 2:
            ax.set_xlabel(axis_labels[1][1], linespacing=0.8)
            ax.set_ylabel(axis_labels[0][1], linespacing=0.8)
        elif len(axis_labels) == 1:
            if abs(self.x_avg-axis_labels[0][0]) < abs(self.y_avg-axis_labels[0][0]):
                ax.set_xlabel(axis_labels[0][1], linespacing=0.8)
            else:
                ax.set_ylabel(axis_labels[0][1], linespacing=0.8)

    def set_style_params(self):

        font_size = random.randint(10, 14)

        if random.random() < 0.85:
            font_weight = "normal"
        else:
            font_weight = "bold"

        if random.random() < 0.85:
            font_style = "normal"
        else:
            font_style = "italic"

        font = random.choice(self.fonts)

        if font == 'cmr10':
            plt.rcParams["axes.formatter.use_mathtext"] = True

        plt.rcParams['font.size'] = font_size
        plt.rcParams['font.family'] = font
        plt.rcParams["font.style"] = font_style
        plt.rcParams['font.weight'] = font_weight

        if random.random() < 0.3:
            plt.margins(x=random.random()*0.15, y=random.random()*0.15)

    def randomize_spines_and_ticks(self):

        ax = plt.gca()

        if random.random() < 0.2:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            if random.random() < 0.5:
                ax.spines['bottom'].set_visible(False)
            else:
                ax.spines['bottom'].set_visible(True)
        else:
            ax.spines['right'].set_visible(True)
            ax.spines['top'].set_visible(True)
            ax.spines['bottom'].set_visible(True)

        if random.random() < 0.2:
            ax.xaxis.set_ticks_position('none')
        if random.random() < 0.2:
            ax.yaxis.set_ticks_position('none')

    def drop_x_ticks(self, sample_data):

        if len(sample_data['x_ticks']) != len(sample_data['xs']) or len(sample_data['xs']) < 4:
            return sample_data
    
        min_drops = 1
        max_drops = max(1, round(len(sample_data['xs']) * 0.25))

        ids = sorted(random.sample(sample_data['x_ticks_vals'], len(sample_data['x_ticks_vals'])-random.randint(min_drops, max_drops)))

        sample_data['x_ticks'] = list(np.array(sample_data['x_ticks'])[ids])
        sample_data['x_ticks_vals'] = list(range(len(ids)))

        sample_data['xs_vals'] = list(range(len(ids)))
        sample_data['xs'] = list(np.array(sample_data['xs'])[ids])
        sample_data['ys'] = list(np.array(sample_data['ys'])[ids])

        return sample_data
    
    def add_x_ticks_noise(self, sample_data):

        if sample_data['chart_type'] == 'horizontal_bar' or len(sample_data['x_ticks']) != len(sample_data['xs']):
            return sample_data
        
        is_int_numerical = self.is_int_numerical(sample_data['xs']) 

        # modify strings less often
        if not is_int_numerical and random.random() < 0.8:
            return sample_data

        if random.random() < 0.25:
            first_pos = random.randint(0, len(sample_data['xs'])-1)
            second_pos = first_pos + random.choice([-2,-1,1,2])

            if second_pos >= 0 and second_pos < len(sample_data['xs']):
                sample_data['xs'][first_pos] = sample_data['xs'][second_pos]
                sample_data['x_ticks'][first_pos] = sample_data['x_ticks'][second_pos]
        
        elif random.random() < 0.25:
            first_pos = random.randint(0, len(sample_data['xs'])-1)
            second_pos = random.randint(0, len(sample_data['xs'])-1)
            sample_data['xs'][first_pos] = sample_data['xs'][second_pos]
            sample_data['x_ticks'][first_pos] = sample_data['x_ticks'][second_pos]
        
        # char level modifications just for numbers
        if not is_int_numerical:
            return sample_data
        
        if random.random() < 0.5:
            pos = random.randint(0, len(sample_data['xs'])-1)
            modify = random.randint(0,2)

            # add char
            if modify == 0:
                new_x = sample_data['xs'][pos]
                char_pos = random.randint(0, len(new_x))

                new_x = new_x[:char_pos] + new_x[random.randint(0,len(new_x)-1)] + new_x[char_pos:]
                sample_data['xs'][pos] = new_x
                sample_data['x_ticks'][pos] = new_x
            # remove char
            elif modify == 1 and len(sample_data['xs'][pos]) > 1:
                new_x = sample_data['xs'][pos]
                char_pos = random.randint(0, len(new_x)-1)

                new_x = new_x[:char_pos] + new_x[char_pos+1:]
                sample_data['xs'][pos] = new_x
                sample_data['x_ticks'][pos] = new_x
            # replace with neighbour
            elif modify == 2:
                new_x = sample_data['xs'][pos]
                char_pos = random.randint(0, len(new_x))

                tmp_pos = char_pos + random.choice([-1,1])

                if tmp_pos >= 0 and tmp_pos < len(new_x):
                    new_x = new_x[:char_pos] + new_x[tmp_pos] + new_x[char_pos+1:]
                    sample_data['xs'][pos] = new_x
                    sample_data['x_ticks'][pos] = new_x

        return sample_data

    def pre_generate(self, sample_data):
        pass

    def post_generate(self, sample_data):
        pass

    def drop_to_min(self, sample_data):

        # set some values to minimal - especially start and end
        if random.random() < 0.2:
            min_val = min(min(sample_data['ys']), min(sample_data['y_ticks_vals']))
            if random.random() < 0.3 and len(sample_data['ys']) >= 2:
                vs = random.sample(list(range(len(sample_data['ys']))), random.randint(1, len(sample_data['ys']) // 2))
                for v in vs:
                    sample_data['ys'][v] = min_val
            elif random.random() < 0.5:
                sample_data['ys'][0] = min_val
            else:
                sample_data['ys'][-1] = min_val

    def make_big_or_small(self, sample_data):

        try:
            y_ticks = [int(y) for y in sample_data['y_ticks']]
        except:
            y_ticks = [float(y) for y in sample_data['y_ticks']]
    
        y_ticks_vals = sample_data['y_ticks_vals']
        ys = sample_data['ys']
        decimals = sample_data['y_decimals']

        max_decimals = self.get_max_decimals(sample_data['y_ticks'])

        decimals_removed = 0

        if max_decimals == 0:
            while all([y // 10 * 10 == y for y in y_ticks]):
                decimals_removed += 1
                y_ticks = [y // 10 for y in y_ticks]

        if decimals_removed > 0:
            y_ticks_vals = [y / 10 ** decimals_removed for y in y_ticks_vals]
            ys = [y / 10 ** decimals_removed for y in ys]
            decimals += decimals_removed

        v = random.randint(-12, 12)

        decimals = decimals - v
        max_decimals = max_decimals - v

        v = 10 ** v

        y_ticks = [y * v for y in y_ticks]
        y_max = max([abs(y) for y in y_ticks])

        if y_max > 1e11 or y_max < 1e-11:
            return
    
        y_ticks_vals = [y * v for y in y_ticks_vals]
        ys = [y * v for y in ys]

        sample_data['y_ticks'] = [round_float(y, max_decimals) for y in y_ticks]
        sample_data['y_ticks_vals'] = y_ticks_vals
        sample_data['ys'] = ys
        sample_data['y_decimals'] = decimals

    def generate(self):

        sample_data = random.choice(self.data) #[x for x in self.data if x['id'] == '03a15ca4ece6'][0] #
        sample_data = copy.deepcopy(sample_data)

        multiline = False

        if ';'.join(sample_data['x_ticks']) == ';'.join(sample_data['xs']):

            # add multiline text
            if random.random() < 0.2 and not self.is_numerical(sample_data['x_ticks']):

                words = []
                for x in sample_data['x_ticks']:
                    words.extend(x.split(' '))
                
                x_ticks = []

                tick_count = random.randint(4,7)
                if tick_count > len(sample_data['x_ticks']):
                    tick_count = len(sample_data['x_ticks'])

                sample_data['x_ticks'] = sample_data['x_ticks'][:tick_count]
                sample_data['x_ticks_vals'] = list(range(len(sample_data['x_ticks'])))
                sample_data['xs'] = sample_data['xs'][:tick_count]
                sample_data['xs_vals'] = list(range(len(sample_data['xs'])))

                sample_data['ys'] = random.sample(sample_data['ys'], tick_count)

                for _ in range(len(sample_data['x_ticks'])):
                    cnt = random.randint(1,min(3, len(words)))
                    x_ticks.append(random.sample(words, cnt))
                
                xs = ['\n'.join(x) for x in x_ticks]
                x_ticks = ['\n'.join(x) for x in x_ticks]
            
                sample_data['x_ticks'] = x_ticks
                sample_data['xs'] = xs      

                multiline = True

            # long xs
            elif random.random() < 0.2:
                sample_data['x_ticks'] = sample_data['x_ticks'] + sample_data['x_ticks']
                sample_data['x_ticks_vals'] = list(range(len(sample_data['x_ticks'])))
                sample_data['xs'] = sample_data['xs'] + sample_data['xs']
                sample_data['xs_vals'] = list(range(len(sample_data['xs'])))

                sample_data['ys'] = sample_data['ys'] + sample_data['ys']
                if random.random() < 0.7:
                    random.shuffle(sample_data['ys'])

                if random.random() < 0.7:
                    random.shuffle(sample_data['xs'])
                    sample_data['x_ticks'] = sample_data['xs'][:]
    

        # big and small values
        if random.random() < 0.1 and self.is_numerical(sample_data['y_ticks']) and not '-' in ''.join(sample_data['y_ticks']):
            self.make_big_or_small(sample_data)
            
        self.pre_generate(sample_data)

        if random.random() < 0.01:
            all_y_ticks = ''.join(sample_data['y_ticks'])
            if '%' not in all_y_ticks and '$' not in all_y_ticks:
                if random.random() < 0.5:
                    sample_data['y_ticks'] = [y + '%' for y in sample_data['y_ticks']]
                elif random.random() < 0.5:
                    sample_data['y_ticks'] = [y + '$' for y in sample_data['y_ticks']]
                else:
                    sample_data['y_ticks'] = ['$' + y for y in sample_data['y_ticks']]

        if sample_data['chart_type'] != 'line':
            drop_val = 0.02
        else:
            drop_val = 0.05

        if random.random() < drop_val:
            sample_data = self.drop_x_ticks(sample_data)

        if random.random() < drop_val:
            sample_data = self.add_x_ticks_noise(sample_data)

        chart_title = sample_data['chart_title']
        axis_labels = sample_data['axis_labels']
        
        mpl_style = random.choice(self.mpl_styles)
        plt.rcdefaults()
        
        with plt.style.context(mpl_style):

            self.set_style_params()
            
            self.generate_graph(sample_data)

            self.set_chart_axes_titles(chart_title, axis_labels, is_multiline=multiline)
            self.rotate_ticks(sample_data['x_ticks'], sample_data['y_ticks'])
            self.randomize_spines_and_ticks()

            font = random.choice(['cmb10', 'cmr10', 'cmss10', 'cmtt10'])
            fontsize = random.randint(8,13)

            if random.random() < 0.1:
                ax = plt.gca()
                ax.set_xlabel('X', fontsize = fontsize, font=font)
                ax.xaxis.set_label_coords(1.02 + random.random()*0.02, 0.005 + random.random()*0.01)
            
            if random.random() < 0.1:
                ax = plt.gca()
                ax.set_ylabel('Y', fontsize = fontsize, rotation = 0, font=font)
                ax.yaxis.set_label_coords(0.005 + random.random()*0.01, 1.0 + random.random()*0.02)

            buf = io.BytesIO()
            plt.savefig(buf, format='jpg', bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)

            width, height = img.size

            max_size = max(width, height)
            div = max_size / random.randint(448, 576)

            width, height = round(width/div), round(height/div)
            img = img.resize((width, height))
            
            buf.close() 
            plt.close()

            self.post_generate(sample_data)
            
            return img, self.make_gt(sample_data)
    
    def make_gt(self, sample_data):
        raise NotImplemented()

    def generate_graph(self):
        raise NotImplemented()
    
    def get_gt_x(self, sample_data):

        xs = sample_data['xs']

        if not sample_data['x_decimals'] is None:
            xs = [round_float(x, sample_data['x_decimals']) for x in xs]
        
        return xs

    def get_gt_y(self, sample_data):

        ys = sample_data['ys']
        
        if not sample_data['y_decimals'] is None:
            ys = [round_float(y, sample_data['y_decimals']) for y in ys]
        
        return ys

class HorizontalBarGenerator(BaseGraphGenerator):

    def make_gt(self, sample_data):
        return make_gt('horizontal_bar', self.get_gt_y(sample_data), self.get_gt_x(sample_data))
    
    def drop_bar(self, sample_data):

        x_ticks_vals = np.array(sample_data['x_ticks_vals'])

        if random.random() < 0.1:
            x_ticks_vals = sorted(np.random.choice(x_ticks_vals, size=random.randint((len(x_ticks_vals)+1)//2,len(x_ticks_vals)), replace=False))

        sample_data['xs'] = list(np.array(sample_data['xs'])[x_ticks_vals])
        sample_data['xs_vals'] = list(np.array(sample_data['xs_vals'])[x_ticks_vals])

        sample_data['ys'] = list(np.array(sample_data['ys'])[x_ticks_vals])
        sample_data['x_ticks_vals'] = list(x_ticks_vals)

    def post_generate(self, sample_data):
        sample_data['xs'].reverse()
        sample_data['ys'].reverse()

    def generate_graph(self, sample_data):

        # shuffle xs
        random.shuffle(sample_data['ys'])

        self.drop_to_min(sample_data)

        #self.drop_bar(sample_data)

        xs_vals = sample_data['xs_vals']
        x_ticks_vals = sample_data['x_ticks_vals']
        xs = sample_data['xs']

        y_ticks = sample_data['y_ticks']
        y_ticks_vals = sample_data['y_ticks_vals']
        ys = sample_data['ys']

        ax = plt.gca()

        height = random.random()*0.57+0.4
        if height >= 0.94:
            height = 1.0

        patterns = self.generate_random_patterns(len(ys))

        ax.barh(xs_vals, ys, height=height, color=self.generate_random_colors(len(ys)), zorder=3, edgecolor=self.generate_edge_color(), hatch=patterns)
        ax.set_yticks(ticks=x_ticks_vals, labels=xs)    
        ax.set_xticks(y_ticks_vals, labels=y_ticks)
        #ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlim(min(y_ticks_vals + ys), max(y_ticks_vals + ys))
    
class VerticalBarGenerator(BaseGraphGenerator):

    def make_gt(self, sample_data):
        return make_gt('vertical_bar', self.get_gt_x(sample_data), self.get_gt_y(sample_data))

    def drop_bar(self, sample_data):

        x_ticks_vals = np.array(sample_data['x_ticks_vals'])

        if random.random() < 0.2:
            x_ticks_vals = sorted(np.random.choice(x_ticks_vals, size=random.randint((len(x_ticks_vals)+1)//2,len(x_ticks_vals)), replace=False))

        sample_data['xs'] = list(np.array(sample_data['xs'])[x_ticks_vals])
        sample_data['xs_vals'] = list(np.array(sample_data['xs_vals'])[x_ticks_vals])

        sample_data['ys'] = list(np.array(sample_data['ys'])[x_ticks_vals])
        sample_data['x_ticks_vals'] = list(x_ticks_vals)

    def generate_graph(self, sample_data):

        # shuffle ys
        random.shuffle(sample_data['ys'])

        self.drop_to_min(sample_data)

        #self.drop_bar(sample_data)

        xs_vals = sample_data['xs_vals']
        x_ticks_vals = sample_data['x_ticks_vals']
        xs = sample_data['xs']

        y_ticks = sample_data['y_ticks']
        y_ticks_vals = sample_data['y_ticks_vals']
        ys = sample_data['ys']

        ax = plt.gca()

        width = random.random()*0.57+0.4
        if width >= 0.94:
            width = 1.0

        patterns = self.generate_random_patterns(len(ys))

        ax.bar(xs_vals, ys, width=width, color=self.generate_random_colors(len(ys)), zorder=3, edgecolor=self.generate_edge_color(), hatch=patterns)
        ax.set_xticks(ticks=x_ticks_vals, labels=xs)
        ax.set_yticks(y_ticks_vals, labels=y_ticks)

        ax.set_ylim(min(y_ticks_vals + ys), max(y_ticks_vals + ys))

class LineGenerator(BaseGraphGenerator):

    def __init__(self, chart_types, allowed_ids):
                
        super().__init__(chart_types=chart_types, allowed_ids=allowed_ids)

        # add more numerical xs to lines
        numerical_data = [d for d in self.data if self.is_numerical(d['xs'])]
        #same_dist_numerical = [d for d in self.data if self.is_same_dist_numerical(d['xs'])]
        #print(len(numerical_data), len(same_dist_numerical))
        self.data = self.data + numerical_data + numerical_data + numerical_data

    def make_gt(self, sample_data):
        return make_gt('line', self.get_gt_x(sample_data), self.get_gt_y(sample_data))
    
    def drop_line(self, sample_data):

        start = 0
        end = len(sample_data['xs_vals'])

        if end-start > 4 and random.random() < 0.6:
            start += random.randint(1,2)
        
        if end-start > 4 and random.random() < 0.6:
            end -= random.randint(1,2)

        sample_data['xs_vals'] = sample_data['xs_vals'][start:end]
        ys = np.array(sample_data['ys'])
        sample_data['ys'] = list(ys[sample_data['xs_vals']])

    def get_numerical(self, xs):
        try:
            xs = [float(x) for x in xs]
            return True, xs
        except:
            return False, None

    def pre_generate(self, sample_data):
        y_ticks = sample_data['y_ticks']
        y_ticks_vals = sample_data['y_ticks_vals']
        ys = sample_data['ys']

        if not self.is_numerical(y_ticks) or '-' in ''.join(y_ticks):
            return
        
        # negatives
        if random.random() < 0.05:
            y_ticks = [['-' + y, y][float(y) == 0] for y in y_ticks]
            y_ticks.reverse()
            y_ticks_vals = [-y for y in y_ticks_vals]
            y_ticks_vals.reverse()
            ys = [-y for y in ys]

        # negatives and positives
        elif random.random() < 0.05:
            # keep same decimals
            decimals = self.get_max_decimals(y_ticks)
            if len(y_ticks_vals) % 2 == 0:
                y_mean = sum(y_ticks_vals[:-1]) / len(y_ticks_vals[:-1])
            else:
                y_mean = sum(y_ticks_vals) / len(y_ticks_vals)

            y_ticks = [round_float(float(y)-y_mean, decimals) for y in y_ticks]
            y_ticks_vals = [y-y_mean for y in y_ticks_vals]
            ys = [y-y_mean for y in ys]
        
        # negatives v2
        elif random.random() < 0.05:
            # keep same decimals
            decimals = self.get_max_decimals(y_ticks)
            y_max = max(y_ticks_vals)

            y_ticks = [round_float(float(y)-y_max, decimals) for y in y_ticks]
            y_ticks_vals = [y-y_max for y in y_ticks_vals]
            ys = [y-y_max for y in ys]

        # rounding gives multiple same values
        if len(set(y_ticks)) != len(y_ticks):
            return
        
        sample_data['y_ticks'] = y_ticks
        sample_data['y_ticks_vals'] = y_ticks_vals
        sample_data['ys'] = ys

    def generate_graph(self, sample_data):

        # shuffle ys
        if random.random() < 0.3:
            random.shuffle(sample_data['ys'])

        if random.random() < 0.3:
            self.drop_line(sample_data)
            drop_line = True
        else:
            drop_line = False

        #self.drop_to_min(sample_data)

        xs_vals = sample_data['xs_vals']
        x_ticks_vals = sample_data['x_ticks_vals']
        xs = sample_data['xs']

        numerical, numerical_xs = self.get_numerical(xs)

        y_ticks = sample_data['y_ticks']
        y_ticks_vals = sample_data['y_ticks_vals']
        ys = sample_data['ys']

        ax = plt.gca()

        if random.random() < 0.2:
            marker = random.choice(['o', '8', 'H', 'D'])
        else:
            marker = None

        linestyle = '-'
        if random.random() < 0.05:
            linestyle = '--'
            marker = None

        drop_start, drop_end = False, False
        xs_vals = xs_vals[:]

        if numerical and len(xs_vals) >= 2 and len(x_ticks_vals) >= 2 and random.random() < 0.4:
            xs_vals[0] += 0.05 + random.random()*0.9
            drop_start = True

        if numerical and len(xs_vals) >= 2 and len(x_ticks_vals) >= 2 and random.random() < 0.4:
            xs_vals[-1] -= 0.05 + random.random()*0.9
            drop_end = True

        if numerical and marker:
            if drop_start or drop_end or random.random() < 0.15:
                new_xs = np.linspace(xs_vals[0], xs_vals[-1], round(len(xs_vals)*(random.random()+1)))
                new_ys = np.interp(new_xs, xs_vals, ys)
            else:
                if random.random() < 0.2:
                    new_xs = np.linspace(xs_vals[0], xs_vals[-1], len(xs_vals)*2)
                    new_ys = np.interp(new_xs, xs_vals, ys)
                else:
                    new_xs = xs_vals
                    new_ys = ys
        else:
            new_xs = xs_vals
            new_ys = ys

        if not marker and random.random() < 0.3:
            cs = CubicSpline(new_xs, new_ys)
            new_xs = np.linspace(min(new_xs), max(new_xs), 200)
            new_ys = cs(new_xs)

        ax.plot(new_xs ,new_ys, color=self.generate_random_colors()[0], linestyle=linestyle, marker=marker, linewidth=random.random()*5+1)
        
        ax.set_xticks(ticks=x_ticks_vals, labels=xs)
        ax.set_yticks(y_ticks_vals, labels=y_ticks)

        ax.set_ylim(min(y_ticks_vals + ys), max(y_ticks_vals + ys))

        if numerical and random.random() < 0.4:
            ax.xaxis.set_major_locator(MultipleLocator(1.0))
            ax.xaxis.set_minor_locator(MultipleLocator(random.choice([0.1, 0.2, 0.25, 0.5])))
            length=random.randint(3,5)
            ax.tick_params(which='minor', length=length)
            ax.tick_params(which='major', length=random.randint(length,length+2))

        if drop_line:
            xs = np.array(sample_data['xs'])
            xs = xs[sample_data['xs_vals']]
        
        if drop_start:
            ys = ys[1:]
            xs = xs[1:]
        
        if drop_end:
            ys = ys[:-1]
            xs = xs[:-1]

        sample_data['xs'] = list(xs)
        sample_data['ys'] = list(ys)

class HistogramGenerator(BaseGraphGenerator):

    def __init__(self, chart_types, allowed_ids):
                
        super().__init__(chart_types=chart_types, allowed_ids=allowed_ids)

        same_dist_numerical = [d for d in self.data if self.is_same_dist_numerical(d['xs'])]
        self.data = same_dist_numerical

    def make_gt(self, sample_data):
        return make_gt('histogram', self.get_gt_x(sample_data), self.get_gt_y(sample_data))

    def generate_graph(self, sample_data):

        xs_vals = sample_data['xs_vals']
        x_ticks_vals = sample_data['x_ticks_vals']
        xs = sample_data['xs']

        y_ticks = sample_data['y_ticks']
        y_ticks_vals = sample_data['y_ticks_vals']
        ys = sample_data['ys']

        # drop first or last y
        if random.random() < 0.5:
            ys = ys[1:]
        else:
            ys = ys[:-1]

        sample_data['ys'] = ys

        self.drop_to_min(sample_data)

        extra_args = {}

        ax = plt.gca()

        extra_args['histtype'] = random.choice(['bar', 'barstacked', 'step', 'stepfilled'])

        if 'bar' in extra_args['histtype']:
            width = random.random() + 0.7
            if width >= 0.94:
                width = 1.0
            
            extra_args['rwidth'] = width

        if random.random() < 0.9:
            extra_args['edgecolor'] = self.generate_edge_color()
        else:
            extra_args['edgecolor'] = 'none'

        if random.random() < 0.3:

            if random.random() < 0.8:
                pattern = self.generate_random_patterns()[0]
                extra_args['hatch'] = pattern
            
            ax.hist(xs_vals[:-1], bins=xs_vals, weights=ys, facecolor=self.generate_random_colors()[0], zorder=3, fill=True, **extra_args)
        else:
            patterns = self.generate_random_patterns(len(ys))
            colors = self.generate_random_colors(len(ys))

            for idx in range(len(ys)):
                extra_args['hatch'] = patterns[idx]
                extra_args['color'] = colors[idx]

                weights = [0] * len(ys)
                weights[idx] = ys[idx]

                ax.hist(xs_vals[:-1], bins=xs_vals, weights=weights, zorder=3, fill=True, **extra_args)

        ax.set_xticks(ticks=x_ticks_vals, labels=xs)    
        ax.set_yticks(y_ticks_vals, labels=y_ticks)

        ax.set_ylim(min(y_ticks_vals + ys), max(y_ticks_vals + ys))

class GraphGenerator():
    def __init__(self, allowed_ids=None):
        self.generators = {
            'horizontal_bar': HorizontalBarGenerator(chart_types = ['vertical_bar', 'horizontal_bar'], allowed_ids=allowed_ids),
            'vertical_bar': VerticalBarGenerator(chart_types = ['vertical_bar', 'horizontal_bar'], allowed_ids=allowed_ids),
            'line': LineGenerator(chart_types = ['line'], allowed_ids=allowed_ids),
            'histogram': HistogramGenerator(chart_types = ['vertical_bar', 'horizontal_bar'], allowed_ids=allowed_ids),
        }
    
    def generate(self, chart_type):
        if chart_type not in self.generators:
            raise NotImplemented()
    
        return self.generators[chart_type].generate()