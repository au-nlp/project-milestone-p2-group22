"""
Translator will try to translate all docs in SMOL (SMOLDOC) from
English to Danish. It uses eTranslation from EU.

The API for the service can be found at
https://language-tools.ec.europa.eu/dev-corner/etranslation/rest-v2/introduction
"""


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(
        description="Automatically translate documents in SMOL."
    )
    parser.add_argument(
        "endpoint",
        type=str,
        help="the callback endpoint",
        default="translator.jehaj.dk",
    )
