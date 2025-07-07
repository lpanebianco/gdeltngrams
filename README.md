# gdeltngrams

This repository provides a Python package for accessing the GDELT Web News NGrams 3.0 API and reconstructing full-text news articles. 

To learn more about the dataset, please visit the official announcement: https://blog.gdeltproject.org/announcing-the-new-web-news-ngrams-3-0-dataset/

## Installation

After cloning the package from the [GitHub repository](https://github.com/lpanebianco/gdeltngrams), open the command line and run:

```python
python setup.py install
```

To ensure the package has been properly installed run python and type:

```python
import gdeltgrams as gdgrams
```

If you don't get any error messages, then your installation has been successful.

## Code and Example Usage

Data ingestion example:

```python
gdgrams.ingestion(
    dates = "20250101", 
    hours = "00", 
    output_dir = 'ingestion_folder', 
    language_filter = "en", 
    url_filter = None)
```

Data processing example:

```python
gdgrams.multiprocess(
    input_path = 'ingestion_folder', 
    output_file = 'example.gdeltnews.webngrams.csv', 
    language_filter = "en",
    url_filter = None, 
    num_processes = None,
    keywords = ["Trump", "der Leyen"],
    text_condition = lambda text: len(text) > 300 and "israel" in text.lower() 
) 
```

Further details on the [Jupyter notebook](https://github.com/lpanebianco/gdeltngrams/blob/main/gdeltngrams_guide.ipynb).

## Credits

This package was developed with reference to [gdeltnews](https://github.com/iandreafc/gdeltnews), which provided the main multiprocessing function.
