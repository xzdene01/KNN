# From Fajcik

import os
import subprocess
from corpy.morphodita import Tagger
from topmost.preprocessing.preprocessing import Tokenizer


class CzechLemmatizedTokenizer(Tokenizer):
    def __init__(self,
                 stopwords=None,
                 keep_num=False,
                 keep_alphanum=False,
                 strip_html=False,
                 no_lower=False,
                 min_length=3,
                 cache_dir="./models",
                 **kwargs):
        if stopwords is None:
            stopwords = []  # Default value for stopwords

        stopwords += forbidden_words

        # Add the parameters to kwargs if they are expected by the superclass
        kwargs.update({
            'stopwords': stopwords,
            'keep_num': keep_num,
            'keep_alphanum': keep_alphanum,
            'strip_html': strip_html,
            'no_lower': no_lower,
            'min_length': min_length
        })

        # Define the path to the Morphodita model
        self.model_cache_dir = cache_dir
        self.model_path = os.path.join(self.model_cache_dir,
                                       "czech-morfflex2.0-pdtc1.0-220710",
                                       "czech-morfflex2.0-pdtc1.0-220710.tagger")

        # Check if the Morphodita model is available, if not, download it
        if not os.path.exists(self.model_path):
            self.download_morphodita_model()

        # Initialize the Morphodita Tagger
        self.tagger = Tagger(self.model_path)

        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)

    def download_morphodita_model(self):
        """Download and unzip the Morphodita model if not present."""
        os.makedirs(self.model_cache_dir, exist_ok=True)
        print("Morphodita model not found. Downloading...")

        # Shell commands to download and unzip the Morphodita model
        commands = [
            "curl --remote-name https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-4794/czech-morfflex2.0-pdtc1.0-220710.zip",
            f"unzip czech-morfflex2.0-pdtc1.0-220710.zip -d {self.model_cache_dir}",
            "rm czech-morfflex2.0-pdtc1.0-220710.zip"  # Clean up the zip file after extraction
        ]

        # Execute the shell commands
        for command in commands:
            subprocess.run(command, shell=True, check=True)

    def tokenize(self, text):
        """
        Tokenize and lemmatize the input text using spaCy.
        This function only performs lemmatization and skips other NLP components.

        Args:
            text (str): The input text to process.

        Returns:
            list: A list of lemmatized tokens.
        """
        # Clean text
        text = self.clean_text(text, self.strip_html, self.lower)

        # Process the text using corpy pipeline
        tokens = [token.lemma for token in self.tagger.tag(text)]

        # Remove -[num] suffixes from tokens
        tokens = [token.split("-")[0] if not token.startswith("-") else token for token in tokens]
        tokens = [token.split("_")[0] if not token.startswith("_") else token for token in tokens]

        # Drop stopwords
        tokens = [token for token in tokens if token not in self.stopword_set]

        # Remove numeric tokens
        tokens = [token for token in tokens if not token.isdigit()]

        # drop short tokens
        if self.min_length > 0:
            tokens = [t if len(t) >= self.min_length else '_' for t in tokens]

        return tokens

forbidden_words = [
    "ó"
]