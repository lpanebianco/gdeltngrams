# multiprocess/__init__.py
import json
import csv
from collections import defaultdict
import pandas as pd
import re
import os
from typing import Callable, Dict, List, Tuple, Union, Optional
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import logging

__all__ = ['multiprocess']
#Get the input data from the following URL, provided as an example: http://data.gdeltproject.org/gdeltv3/webngrams/20250316000100.webngrams.json.gz and extract the json.

def reconstruct_sentence(fragments: List[str], positions: List[int] = None) -> str:
    """
    Reconstructs text by merging a group of overlapping fragments.
    Begins with the first fragment and iteratively finds the best matching unused fragment
    based on word overlap, appending or prepending it as needed.
    If position data is provided, it uses that to prioritize fragments that are not overly distant
    in original text order to improve reconstruction accuracy.
    Returns a single string that represents the reconstructed text.
    """

    if not fragments:
        return ""
    if len(fragments) == 1:
        return fragments[0]

    # Create position mapping if provided
    pos_map = {}
    if positions:
        pos_map = {i: pos for i, pos in enumerate(positions)}

    # Split all fragments into words once
    words_list = [fragment.split() for fragment in fragments]
    result_words = words_list[0]
    used = {0}

    while len(used) < len(fragments):
        best_overlap = 0
        best_fragment = -1
        best_is_prefix = False

        for i in range(len(fragments)):
            if i in used:
                continue

            words = words_list[i]

            # Check suffix matching prefix (append operation)
            min_len = min(len(result_words), len(words))
            # Only if positions allow (current fragment position >= first fragment position)
            # Allowance for small position differences
            if positions is None or pos_map[i] + 10 >= pos_map[0]:
                for k in range(min_len, 0, -1):
                    if result_words[-k:] == words[:k] and k > best_overlap:
                        best_overlap = k
                        best_fragment = i
                        best_is_prefix = False
                        break

            # Check prefix matching suffix (prepend operation)
            # Only if positions allow (current fragment position <= first fragment position)
            # Allowance for small position differences
            if positions is None or pos_map[i] - 10 <= pos_map[0]:
                for k in range(min_len, 0, -1):
                    if result_words[:k] == words[-k:] and k > best_overlap:
                        best_overlap = k
                        best_fragment = i
                        best_is_prefix = True
                        break

        if best_fragment == -1:
            break

        if best_is_prefix:
            result_words = words_list[best_fragment][:-best_overlap] + result_words
        else:
            result_words = result_words + words_list[best_fragment][best_overlap:]

        used.add(best_fragment)

    return ' '.join(result_words)


def remove_overlap(text: str) -> str:
    """
    Removes duplicated content that appears at both the beginning and end of a reconstructed text,
    which can occur when overlapping fragments are merged improperly.
    Checks for the longest matching prefix and suffix in the text, and removes the repeated segment.
    Returns the cleaned version of the input text.
    """

    if len(text) < 2:
        return text

    # Maximum possible overlap length to check (half the text length)
    max_check_len = len(text) // 2
    max_overlap_len = 0

    # Find the longest string overlap from beginning and end
    for i in range(1, max_check_len + 1):
        if text[:i] == text[-i:]:
            max_overlap_len = i

    # Remove the overlap from the beginning if found
    if max_overlap_len > 0:
        return text[max_overlap_len:]

    return text


def process_article(url_entries_tuple, language_filter="en"):
    """Process a single article - designed to be run in parallel"""
    url, entries = url_entries_tuple # entries is a list of dictionaries
        
    entries.sort(key=lambda x: x['pos'])

    sentences = [entry['sentence'] for entry in entries]
    positions = [entry['pos'] for entry in entries]

    # Calculate group positions
    group_positions = [pos for _, pos in zip(sentences, positions)]

    # Simplified: treat all sentences as one group
    reconstructed_sentences = reconstruct_sentence(sentences, group_positions)
    text = remove_overlap(reconstructed_sentences)

    # Clean and format text
    textok = text.replace("|", " ").replace('"', " ").strip()
    textok = re.sub(r'\s+', ' ', textok)  # Remove extra spaces

    return {
        "url": url,
        "text": textok,
        "date": entries[0]['date'][:10]
    }

def load_and_filter_data(input_file, keywords=None, language_filter="en", url_filter=None):
    """
    Load data from file and filter by language, URL, and keywords preserving original URL order.

    Parameters:
        input_file (str): Path to a JSONL file.
        keywords (str or List[str], optional): Case-insensitive keyword(s).
        language_filter (str, optional): Language code to filter articles. Default is "en".
        url_filter (str, optional): Substring to filter articles by URL. Default is None (no filtering).
    Returns:
        transformed_articles: a dictionary where each key is a URL and the value is a list of cleaned entries,
    each containing a reconstructed sentence and associated metadata (date, language, type, and sentence position).
        url_order: list of URLs in the same order as they appear in the transformed_articles dictionary
        input_articles_num: total number of URLs within input_file
    """
    # Track the order of URLs as they first appear in the file
    valid_entries = []
    
    with open(input_file, "r", encoding="utf-8") as file:
        for i, line in enumerate(file, 1):
            try:
                entry = json.loads(line)
                valid_entries.append(entry)
            except json.JSONDecodeError as e:
                logging.warning(f"Skipping line {i} due to JSON error: {e}")
    
    # Create a DataFrame from valid entries
    df = pd.DataFrame(valid_entries)
    input_articles_num = len(df["url"].unique())
    
    # Proceed with filtering
    df = df[df['lang'] == language_filter]
    
    if url_filter is not None:
        df = df[df['url'].str.contains(url_filter, na=False)]

    # Combine pre, ngram, and post into a single sentence
    df['sentence'] = df['pre'].fillna('') + ' ' + df['ngram'].fillna('') + ' ' + df['post'].fillna('')
    df['sentence'] = df['sentence'].str.strip() #remove leading and trailing whitespace

    # If an entry appears at the beginning of an article (position is less than 20) 
    # and contains " / ", it is assumed to include
    # incorrectly appended content from the article's end, which is removed.
    mask = pd.to_numeric(df['pos'], errors='coerce') < 20
    df.loc[mask & df['sentence'].str.contains(" / "), 'sentence'] = (
        df.loc[mask & df['sentence'].str.contains(" / "), 'sentence']
        .apply(lambda s: " / ".join(s.split(" / ")[1:]) if len(s.split(" / ")) > 1 else s)
    )
    # Select and reorder final columns
    df = df[['url', 'date', 'lang', 'type', 'pos', 'sentence']]
    # Case no keywords provided
    if keywords is None:
        # Group by URL and convert to dict
        articles = (
            df
            .drop(columns='url')  # Drop before grouping to avoid warning
            .groupby(df['url'])
            .apply(lambda g: g.to_dict(orient='records'))
            .to_dict()
        )
        # List of unique urls
        url_order = list(articles.keys())
        
        return articles, url_order, input_articles_num

    if not isinstance(keywords, list):
        keywords = [keywords]
    keywords = [str(k) for k in keywords]
    keywords_lower = [k.lower() for k in keywords]

    # Create a boolean mask for rows where sentence contains any keyword (case-insensitive)
    mask = df['sentence'].str.lower().apply(lambda s: any(kw in s for kw in keywords_lower))
    # Get URLs that have at least one matching sentence
    urls_to_keep = df.loc[mask, 'url'].unique()
    # Filter df to keep rows with those URLs
    df = df[df['url'].isin(urls_to_keep)]
    # Group by URL and convert to dict
    articles = (
        df
        .drop(columns='url')  # Drop before grouping to avoid warning
        .groupby(df['url'])
        .apply(lambda g: g.to_dict(orient='records'))
        .to_dict()
    )
    # List of unique urls
    url_order = list(articles.keys())

    return articles, url_order, input_articles_num


def multiprocess(input_path: str, output_file: str, language_filter: str = "en",
                                 url_filter: Optional[str] = None, num_processes: Optional[int] = None,
                                 keywords: Optional[Union[str, List[str]]] = None,
                                 text_condition: Optional[Callable[[str], bool]] = None) -> None:
    """
    Reads one or more line-based JSON files (a single file or a folder) and processes
    articles in parallel using multiprocessing. All results are written to a single output file.

    Parameters:
        input_path (str): Path to a JSONL file or a directory containing multiple JSONL files.
        output_file (str): Path to the output file where results will be saved.
        language_filter (str, optional): Language code to filter articles. Default is "en".
        url_filter (str, optional): Substring to filter articles by URL. Default is None (no filtering).
        num_processes (int, optional): Number of processes for multiprocessing. Default is None, which uses all available CPU cores.
        keywords (str or List[str], optional): Case-insensitive keyword(s) to filter text content before processing. Each keyword should contain one or two space-separated words. Longer keywords are accepted but may result in missing articles. Default is None (no keyword filtering).
        text_condition (Callable[[str], bool], optional): Function to filter the reconstructed text. Takes a string and returns True to keep or False to discard. Default is None (no filtering).
    Returns:
        None. Writes processed results to the specified output file.
    """
    input_files = [input_path]
    if os.path.isdir(input_path):
        input_files = [os.path.join(input_path, f) for f in os.listdir(input_path)
                       if os.path.isfile(os.path.join(input_path, f)) and (f.endswith('.json') or f.endswith('.jsonl'))]
    input_files.sort()

    # If num_processes is not specified, use process CPU count
    if num_processes is None:
        try:
            num_processes = os.process_cpu_count() # new in Python 3.13
        except AttributeError:
            num_processes = mp.cpu_count()
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )

    file_exists = os.path.exists(output_file)
    is_empty = (not file_exists) or (os.path.getsize(output_file) == 0)
    if is_empty:
        for input_file in input_files:
            try:
                logging.info(f"Loading and filtering {input_file} using {num_processes} processes...")
                # Load and filter data, capturing original URL order
                articles, url_order, file_total_articles = load_and_filter_data(input_file, keywords, language_filter, url_filter)
            
                # Create lookup by URL for later reordering
                url_index = {url: idx for idx, url in enumerate(url_order)}
            
                # Prepare list of work items
                work_items = list(articles.items())
                # Skip this file if no actual content to process
                if not work_items or all(not v for v in articles.values()):
                    logging.warning(f"No valid articles to process in {input_file}. Skipping.")
                    continue
                total_articles = len(work_items)
                logging.info(f"Processing {total_articles}/{file_total_articles} articles from {input_file}...")

            
                # Create a pool of worker processes
                with mp.Pool(processes=num_processes) as pool:
                    # Process articles in parallel with progress tracking
                    process_func = partial(process_article, language_filter=language_filter)
                    results = []
            
                    # Use imap to process chunks and track progress
                    with tqdm(total=total_articles, desc="Processing articles") as pbar:
                        for result in pool.imap_unordered(process_func, work_items, chunksize=10):
                            results.append(result)
                            pbar.update(1)
            
                # Sort results based on the original URL order
                sorted_results = sorted(results, key=lambda x: url_index.get(x['url'], float('inf')))
            
                # Write results to file
                file_exists = os.path.exists(output_file)
                is_empty = (not file_exists) or (os.path.getsize(output_file) == 0)
    
                with open(output_file, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file, delimiter="|", quoting=csv.QUOTE_NONE)
                    if is_empty:
                        writer.writerow(["Text", "Date", "URL"])    
                    if text_condition is not None:
                        for article in sorted_results:
                            if text_condition(article['text']):
                                writer.writerow([article['text'], article['date'], article['url']])
                    else:
                        for article in sorted_results:
                            writer.writerow([article['text'], article['date'], article['url']])    
            except Exception as e:
                logging.error(f"Error in processing {input_file}: {e}")  
    else:
        logging.info(f"Processing not executed: {output_file} already exists.")
