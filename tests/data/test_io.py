import os
from embmplxrec.data import io as io

class TestIO:
    def test_unique_filename(self):
        filepath = "./temporary_file.extension"
        expected = "./temporary_file_1.extension"
        actual = io.unique_filename_extension(filepath)
        assert actual == expected

    def test_safe_save(self):
        # Create some dummy data
        data = "Hello, world!"

        # Specify filepath
        filepath = "./test.tmp"
        # Create obstruction at filepath
        os.system(f"touch {filepath}")

        # Expected safety filepath
        unique_filepath = "./test_1.tmp"

        # Check safe_save works
        io.safe_save(data, filepath)
        assert os.path.isfile(unique_filepath)
