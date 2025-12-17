"""
A script to load and print the contents of a pickle cache file.

Usage:
    python find_funny_stuff.py | sed 's/\\n/\n/g' | grep -C 5 "India had declared a national emergency in 1975. However, the text explicitly states 1976"

It outputs the entire content of the specified pickle file to the console. Then you can use tools like `sed` and `grep` to search for specific strings within the output.
`sed` is used here to replace escaped newline characters with actual newlines, making the output more readable. `grep` is then used to search for a specific string and display it along with a few lines of context.
"""

import pickle
from pprint import pprint
from termcolor import colored
from textwrap import wrap

with open("data/answers_llama3_8b.pkl", "rb") as f:
    cache = pickle.load(f)

for k, v in cache.items():
    print(colored(f"Key: {"\n".join(wrap(k, subsequent_indent="  "))}", 'green'))
    pprint(v)
