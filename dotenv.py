class Dotenv:
    def __init__(self, filepath):
        self.filepath = filepath
        self.variables = self.load_dotenv()

    def load_dotenv(self):
        with open(self.filepath) as f:
            return dict(
                line.strip().split("=", 1)
                for line in f
                if line.strip() and not line.startswith("#")
            )

    def get(self, key, default=None):
        return self.variables.get(key, default)
