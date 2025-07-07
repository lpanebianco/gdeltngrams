# gdeltngrams/ingestion.py
import csv
import gzip
import json
import os
import re
import tempfile
from collections import defaultdict
from functools import partial
from io import BytesIO
from typing import Callable, Dict, List, Tuple, Union, Optional
import logging

import requests
from tqdm import tqdm

from .multiprocess import *
__all__ = ['ingestion']

import requests
import gzip
import traceback

def load_webngram(timestamp, output_dir=".", language_filter=None, url_filter=None):
    url = f"http://data.gdeltproject.org/gdeltv3/webngrams/{timestamp}.webngrams.json.gz"
    
    try:
        response = requests.get(url)
        if response.status_code == 404:
            pass#print(f"{timestamp} webngram not found.")
            return
        response.raise_for_status()
    except requests.exceptions.RequestException as req_err:
        logging.error(f"HTTP error while accessing {url}: {req_err}")
        traceback.print_exc()
        return

    try:
        decompressed = gzip.decompress(response.content).decode("utf-8")
    except (OSError, UnicodeDecodeError) as decompress_err:
        print(f"Error decompressing or decoding the response for {timestamp}: {decompress_err}")
        traceback.print_exc()
        return

    try:
        data = [json.loads(line) for line in decompressed.splitlines()]
    except json.JSONDecodeError as json_err:
        print(f"JSON parsing error for {timestamp}: {json_err}")
        traceback.print_exc()
        return

    # Apply filters
    if language_filter is not None:
        data = [record for record in data if record.get("lang", "") == language_filter]
    if url_filter is not None:
        data = [record for record in data if url_filter in record.get("url", "")]

    try:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{timestamp}.webngrams.json")
        if os.path.exists(output_path):
            logging.warning(f"File {output_path} already exists. Skipping save.")
            return
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(decompressed)
        logging.info(f"Webngrams for timestamp {timestamp} filtered and saved to {output_path}")
    except OSError as file_err:
        logging.error(f"File system error while saving {timestamp}: {file_err}")
        traceback.print_exc()


def ingestion(dates: Union[str, List[str]], hours: Optional[str]=None, output_dir=".", language_filter: Optional[str] = None, url_filter: Optional[str] = None) -> None:
    """
    This function imports data from the GDELT Web News NGrams 3.0 Dataset, which provides near-real-time global news content.

    Parameters:
    - dates (str or List[str]): a single date (YYYYMMDD) or a list of such elements.
    - hours (str or List[str], optional): a single hour (HH) or list of hours to filter. Default is None (no hour filtering).
    - output_dir (str, optional): directory to save the GDELT JSON data. Default is "." (current directory).
    - language_filter (str, optional): language code to filter articles. Default is None (no language filtering).
    - url_filter (str, optional): substring that must appear in the source URL. Default is None (no URL filtering).
    Returns:
    - None. Writes output to given JSON and CSV paths.
    """
    if not isinstance(dates, list):
        dates = [dates]
    dates = [int(d) for d in dates]

    if hours is None:
        hours = range(24)
    else:
        if not isinstance(hours, list):
            hours = [hours]
        hours = [int(h) for h in hours]
        
    timestamps = [
        f"{d}{h:02d}{m:02d}00"
        for d in dates
        for h in hours
        for m in range(60)
    ]
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    logging.info(f"Starting ingestion process...")
    for time in timestamps:
        load_webngram(
            time, 
            output_dir, 
            language_filter, 
            url_filter)
    logging.info(f"Ingestion process completed.")

