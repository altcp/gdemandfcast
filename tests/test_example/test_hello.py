"""Tests for hello function."""
from gdemandfcast.ai import execute


# Test Manual ML
def test_execute_manualml():

    train = "./gdemandfcast/data/Train Data.xlsx"
    test = "./gdemandfcast/data/Test Data.xlsx"
    df = execute(train, test, 3).frm()
    print(df)


# Test Auto ML
def test_execute_automl():

    train = "./gdemandfcast/data/Train Data.xlsx"
    test = "./gdemandfcast/data/Test Data.xlsx"
    df = execute(train, test, 3, "auto").frm()
    print(df)
