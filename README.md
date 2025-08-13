# gdeltngrams

This repository provides a Python package for accessing the GDELT Web News NGrams 3.0 API and locally reconstructing full-text news articles. No web scraping is performed.

To learn more about the dataset, please visit the official announcement: https://blog.gdeltproject.org/announcing-the-new-web-news-ngrams-3-0-dataset/

## Installation

After cloning the package from the [GitHub repository](https://github.com/lpanebianco/gdeltngrams), open the command line and run:

```python
pip install -e .
```

To verify the installation, run Python and type:

```python
import gdeltngrams as gdn
```

If you don't get any error messages, then your installation has been successful.

## Example Usage

Data ingestion example: 

```python
gdn.ingestion(
    dates = "20250101", # YYYYMMDD
    hours = "00", 
    output_dir = 'ingestion_folder', 
    language_filter = "en", 
    url_filter = None)
```

Data processing example:

```python
gdn.multiprocess(
    input_path = 'ingestion_folder', 
    output_file = 'example.gdeltnews.webngrams.csv', 
    language_filter = "en",
    url_filter = None, 
    num_processes = None,
    keywords = ["Trump", "der Leyen"],  
    text_condition = lambda text: len(text) > 300 and "israel" in text.lower()  
) 
```

```keywords``` filters articles that contain at least one of the specified keywords (max 2 words suggested). This filtering is applied before multiprocessing.  

```text_condition``` keeps only articles satisfying the given condition, and is applied after multiprocessing.  

For further details, see the [Jupyter notebook](https://github.com/lpanebianco/gdeltngrams/blob/main/gdeltngrams_guide.ipynb).

## Credits

This package was developed with reference to [iandreafc/gdeltnews](https://github.com/iandreafc/gdeltnews), which provided the main multiprocessing function.
