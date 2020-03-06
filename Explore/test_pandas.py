

import pandas as pd
import matplotlib.pyplot as plt


def test():
    import glob
    cwd = os.path.realpath(os.getcwd())
    pickle_files = glob.glob('*.pd')
    dfs = []
    for file in pickle_files:
        print(file)
        df = pd.read_pickle(os.path.join(cwd, file))
        df['timestamp'] = pd.to_datetime(df.date + " " + df.time)
        dfs.append(df)

    bdf = pd.concat(dfs, sort=True, ignore_index=True)
    for df in dfs[1:]:
        bdf = bdf.append(df, sort=True, ignore_index=True)
        
    i = 0
    for i, df, file in zip(it.count(), dfs, pickle_files):
        fig = plt.figure(i)
        fig.clf()
        ax = df.sog.plot()
        ax = df.speed_water_referenced.plot(ax=ax)
        fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')

def test2():
    # Returns the first 5 rows of the dataframe. To override the default, you may insert a
    # value between the parenthesis to change the number of rows returned. Example:
    # df.head(10) will return 10 rows.
    df.head()

    # Returns the last 5 rows of the dataframe. You may insert a value between the parenthesis
    # to change the number of rows returned.
    df.tail()

    # Returns a tuple representing the dimensions. For example, an output of (48, 14)
    # represents 48 rows and 14 columns.
    df.shape

    # Provides a summary of the data including the index data type, column data types,
    # non-null values and memory usage.
    df.info()

    # Provides descriptive statistics that summarizes the central tendency, dispersion,
    # and shape
    df.describe()

    bdf.memory_usage()

    bdf.desired_mode = bdf.desired_mode.astype(‘category’)
    

    
