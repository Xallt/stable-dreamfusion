from main import main, parse_args
import itertools

if __name__ == '__main__':
    exp_grid = {
        'text': [
            "an HDR photo of a frog",
            "a cute elephant with a long trunk and ivory tusks",
            "an HDR photo of a hamburger",
            "a stack of pancakes with maple syrup and butter"
        ],
        'encoding': [
            'hashgrid',
            'frequency'
        ],
        'network': [
            'nerf',
            'neus'
        ]
    }

    keywords = ['frog', 'elephant', 'hamburger', 'pancakes']

    keys, values = zip(*exp_grid.items())
    value_grid = itertools.product(*values)
    value_grid = [{k: v for k, v in zip(keys, value_tuple)} for value_tuple in value_grid]

    for value_args in value_grid:
        text_keyword = None
        for keyword in keywords:
            if keyword in value_args['text']:
                text_keyword = keyword
                break
        exp_name = f"{text_keyword}_{value_args['encoding']}_{value_args['network']}"
        exp_path = f"exp_trials/{exp_name}"
        args = parse_args([
            '--text', value_args['text'],
            '--encoding', value_args['encoding'],
            '--network', value_args['network'],
            '--W', '256',
            '--H', '256',
            '--workspace', exp_path,
            '--batch_size', '4',
            '-O2',# '--dummy'
        ])
        main(args)
