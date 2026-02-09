"""
DEASRec Example Run Script
This script demonstrates how to run the DEASRec model with RecBole.
"""

import argparse
from recbole.quick_start import run_recbole


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='DEASRec', help='Model name')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='Dataset name')
    parser.add_argument('--config_files', type=str, default=None, help='Config files')
    
    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    
    run_recbole(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
