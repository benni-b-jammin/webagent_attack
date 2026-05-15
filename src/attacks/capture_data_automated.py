"""
Name: capture_data_automated.py
Description: Completely automated data capture of HTML ax trees to JSON file format
Authors: Benji Lawrence and Nolan Schlacht
Last updated: 14/05/2026
"""

import multiprocessing
import argparse

import argparse

from src.utils.website_data import (
    DEFAULT_TARGET_WEBSITES,
    DEFAULT_URLS_PER_QUERY,
    get_website_data,
)

from src.utils.data_processing import DEFAULT_N_SEARCH_QUERIES

parser = argparse.ArgumentParser(
    description="Prepare website observation JSON files for the web-agent pipeline"
)

parser.add_argument(
    "instruction",
    choices=["get_webs"],
    help="Current supported step: retrieve candidate websites and capture their observation JSONs.",
)

parser.add_argument(
    "--query_types_file",
    type=str,
    required=True,
    help="Path to a text file containing website/page types, one per line.",
)

parser.add_argument(
    "--n_websites",
    type=int,
    default=DEFAULT_TARGET_WEBSITES,
    help="Number of website JSON files to save.",
)

parser.add_argument(
    "--n_search_queries",
    type=int,
    default=DEFAULT_N_SEARCH_QUERIES,
    help="Number of OpenAI-generated search queries to create from the input file.",
)

parser.add_argument(
    "--urls_per_query",
    type=int,
    default=DEFAULT_URLS_PER_QUERY,
    help="Number of URLs to request from SerpApi per query.",
)

parser.add_argument(
    "--out_dir",
    type=str,
    default="src/data/datasets/auto_data",
    help="Directory in which to save captured website JSON files.",
)

args = parser.parse_args()

if args.instruction == "get_webs":
    get_website_data(
        query_types_file=args.query_types_file,
        n_websites=args.n_websites,
        n_search_queries=args.n_search_queries,
        urls_per_query=args.urls_per_query,
        out_dir=args.out_dir,
    )
else:
    raise ValueError("Invalid instruction")