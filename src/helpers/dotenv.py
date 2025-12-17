import os


class Dotenv:
    def __init__(self, filepath: str = None):
        if filepath is None:
            dirpath = os.path.dirname(os.path.abspath(__file__))
            env_path = os.path.join(os.path.dirname(os.path.dirname(dirpath)), ".env")
            self.filepath = env_path
        else:
            self.filepath = filepath
        self.variables = self.load_dotenv()

    def load_dotenv(self):
        with open(self.filepath) as f:
            return dict(
                line.strip().split("=", 1)
                for line in f
                if line.strip() and not line.startswith("#")
            )

    def get(self, key: str, default: str | None = None) -> str | None:
        return self.variables.get(key, default)
