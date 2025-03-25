#!/usr/bin/env python3
import argparse
from utils.misc import parse_args

def train_parse_args() -> argparse.ArgumentParser:
    parser = parse_args(parser_name = 'Training')
    parser.add_argument('--monitor_dir',
                        type=str, default='monitor_train',
                        help='directory to save monitoring plots (default: ./train)')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--epochs',
                        type=int, default=20,
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--batch_size',
                        type=int,
                        default=10,
                        help='batch size for training (default: 10)')
    parser.add_argument('--num_workers',
                        type=int,
                        default = 4,
                        help='number of workers (default: 4)')

    # Augmentation arguments
    parser.add_argument('--augmentation_types',
                        nargs='+',
                        help='''List of augmentation types to use.
                                Valid options: "fog", "noise", "gaussian_blur", "color_jitter", "horizontal_flip", "rain_effect"
                                If specified, augmentation is enabled.
                                Example: --augmentation_types fog noise horizontal_flip''')
    parser.add_argument('--augmentation_prob',
                        type=float,
                        default=0.25,
                        help='Probability of applying augmentation to a video (default: 0.25)')
    parser.add_argument('--horizontal_flip_prob',
                        type=float,
                        default=0.5,
                        help='Probability of flipping a video horizontally (default: 0.5)')
    #model argument
    parser.add_argument('--model_type', 
                        type = str,
                        default = 'baseline',
                        help = 'Type of model (default: baseline)',
                        choices = ['timesformer', 'baseline', 'accidentxai', 'swintransformer']
                        )
    #optimizer argument
    parser.add_argument('--optimizer',
                        type = str,
                        default = 'radam',
                        help = 'option of optimizer(default: radam)'
                        )
    parser.add_argument('--print_freq', type = int, default = 4, help = 'Frequency for printing')
    parser.add_argument('--training_dir', type = str, help = 'folder for extracted train directory') 
    parser.add_argument('--training_csv', type = str, help = 'extracted clip csv file for training dataset')
    parser.add_argument('--validation_dir', type = str, help = 'extracted clip folder for validation dataset')
    parser.add_argument('--validation_csv', type = str, help = 'extracted clip csv file for validation dataset')
    parser.add_argument('--evaluation_dir', type = str, help = 'extracted clip folder evaluatio  dataset')
    parser.add_argument('--evaluation_csv', type = str, help = 'extracted clip csv file for evaluation dataset')
    _args = parser.parse_args()
    return _args
