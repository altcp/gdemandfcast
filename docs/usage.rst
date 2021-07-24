=====
Usage
=====

To use gdemandfcast in a project::

    import gdemandfcast.ai as gdf

    train = '/path/to/train.xlsx'
    test = '/path/to/test.xlsx'
    num_of_lags = 3
    pred = gdf.execute(train, test, num_of_lags).frm()
    print(pred)
