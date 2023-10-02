from os.path import join as jn

import pytest

from EMB.mplxio import readers
# =========== TESTS ===========
class TestReaders:
    def test_from_edgelist(self):
        mplx = readers.from_edgelist(jn('tests', 'mplxio', 'valid.edgelist'))
        assert isinstance(mplx, dict)

    def test_from_edgelists_not_exists(self):
        with pytest.raises(FileNotFoundError):
            readers.from_edgelist(jn('nonexistent', 'missing.data'))

    def test_from_edgelist_delimiter(self):
        mplx = readers.from_edgelist(jn('tests', 'mplxio', 'valid_delim.edges'), ',')
        assert isinstance(mplx, dict)

    def test_from_edgelist_baddelim(self):
        with pytest.raises(TypeError):
            readers.from_edgelist(jn('tests', 'mplxio', 'valid.edgelist'), 1)

class TestReadersHelpers:
    def test_file_exists(self):
        assert readers._check_exists(jn('tests', 'mplxio', 'test_writers.py')) is True

    def test_file_not_exists(self):
        assert readers._check_exists(jn('this', 'file', 'does_not.exists')) is False

    def test_process_line_whitespace(self):
        expected = ['some', 'example', 'text']
        actual = readers._process_line("some example text")

        assert actual == expected

    def test_process_line_nonws_sep(self):
        expected = ['some', 'example-text']
        actual = readers._process_line("some_example-text", '_')

        assert actual == expected

    def test_process_line_no_sep_present(self):
        expected = ["some example text"]
        actual = readers._process_line("some example text", "agh")

        assert actual == expected

    def test_process_line_badsep(self):
        with pytest.raises(TypeError):
            readers._process_line('test', 1)
