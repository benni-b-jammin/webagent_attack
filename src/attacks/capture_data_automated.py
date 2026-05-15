"""
Name: capture_data_automated.py
Description: Completely automated data capture of HTML ax trees to JSON file format
Authors: Benji Lawrence and Nolan Schlacht
Last updated: 14/05/2026
"""

import multiprocessing
import argparse

from src.utils.website_data import get_web_data
from src.utils.data_processing import submit_goal_object_batch

def main():
    ap = argparse.ArgumentParser(description="Capture Data (Automated)")
    ap.add_argument(
        '--n_batch_splits',
        type=int,
        default=1,
        help="Number of batch splits to use for OpenAI batch API, which has a limit of file sizes. Recommend 1 per 5000 websites"
    )
    ap.add_argument(
        '--batch_ids',
        type=str,
        nargs='+',
        help="Batch ids of the goal object to retrieve. Wait 24 hours after submitting to OpenAI."
    )
    args=ap.parse_args()

    # Get website data
    # Run on 4 processes
    with multiprocessing.Pool(4) as pool:
        pool.map(get_web_data, range(4))
    # Set goals
    ids = submit_goal_object_batch(args.n_batch_splits)
    print("Saving batch ids for retrieval later...")
    # Write ids to a file and pause program for 24 hours before getting goals
    # Get goals