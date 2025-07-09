# gdeltngrams/ingestion.py
import gzip
import json
import os
from typing import List, Union, Optional
import logging
import requests
from tqdm import tqdm
import requests
import gzip
import traceback

from .multiprocess import *
__all__ = ['ingestion']

def load_webngram(timestamp, output_dir=".", language_filter=None, url_filter=None):
    url = f"http://data.gdeltproject.org/gdeltv3/webngrams/{timestamp}.webngrams.json.gz"
    
    try:
        response = requests.get(url)
        if response.status_code == 404:
            pass
            return
        response.raise_for_status()
    except requests.exceptions.RequestException as req_err:
        logging.error(f"HTTP error while accessing {url}: {req_err}")
        traceback.print_exc()
        return

    try:
        decompressed = gzip.decompress(response.content).decode("utf-8")
    except (OSError, UnicodeDecodeError) as decompress_err:
        logging.error(f"Error decompressing or decoding the response for {timestamp}: {decompress_err}")
        traceback.print_exc()
        return

    try:
        data = [json.loads(line) for line in decompressed.splitlines()]
    except json.JSONDecodeError as json_err:
        logging.error(f"JSON parsing error for {timestamp}: {json_err}")
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


DateLike = Union[int, str]
DateInput = Union[DateLike, List[DateLike]]
HourInput = Optional[Union[DateLike, List[DateLike]]]

def ingestion(
    dates: DateInput,
    hours: HourInput = None,
    output_dir: str = ".",
    language_filter: Optional[str] = None,
    url_filter: Optional[str] = None,
) -> None:
    """
    Import data from the GDELT Web News NGrams 3.0 Dataset.

    Args:
        dates (int, str, list of int or str): One or more dates in YYYYMMDD format.
        hours (int, str, list of int or str, optional): One or more hours in HH format. Defaults to None (all hours).
        output_dir (str, optional): Directory to save output files. Defaults to ".".
        language_filter (str, optional): ISO 639-1 two-letter language code to filter articles. Defaults to None.
        url_filter (str, optional): Substring to filter URLs. Defaults to None.

    Returns:
        None: Writes JSON files to the specified output directory.
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

