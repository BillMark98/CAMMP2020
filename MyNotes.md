
# python learn

## import module

* [here](https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder),
to import a file in the `path` use 

```python
import sys
sys.path.insert(0,path)
import myModule
```

## basic python

* [create sequence between 2 values](https://stackoverflow.com/questions/18265935/python-create-list-with-numbers-between-2-values)

```python
list(range(low,high))
```

because in `python3`, `range` will return an iterator

* [create distinct randomly picked sequence](https://stackoverflow.com/questions/8505651/non-repetitive-random-number-in-numpy)

```python
np.random.choice(range(lowerBound, upperBound), num , replace= False)
```

* [run python script](https://www.knowledgehut.com/blog/programming/run-python-scripts) or [here](https://realpython.com/run-python-scripts/#:~:text=To%20run%20Python%20scripts%20with,python3%20hello.py%20Hello%20World!) or [here](https://stackoverflow.com/questions/17247471/how-to-run-a-python-script-from-idle-interactive-shell)
Note that in python shell, using the syntax

```python
if __name__ == (__file__):
    do_something
```

will cause the problem `python name __file__ is not defined`, see [here](https://stackoverflow.com/questions/16771894/python-nameerror-global-name-file-is-not-defined)

* [test float equal](https://www.quora.com/How-does-one-correctly-compare-two-floats-in-Python-to-test-if-they-are-equal)

```python
abs(a-b) < epsilon
```

* boolean array for indexing array
need to remember that the syntax is like the following

```python
    boolArr = (arr < upBound) & (arr > lowBound)
```

Note that the parenthesis are essential, else will cause the problem [`truth value of an array with more than one element is ambiguous`](https://stackoverflow.com/questions/10062954/valueerror-the-truth-value-of-an-array-with-more-than-one-element-is-ambiguous)

* [indexing array based on a list](https://stackoverflow.com/questions/3179106/python-select-subset-from-list-based-on-index-set)

## os



## fileIO

* [check if file exists](https://stackoverflow.com/questions/82831/how-do-i-check-whether-a-file-exists-without-exceptions)

```python
import os
os.path.isfile(fileName)
```

## list manipulation

* [get unique value](https://stackoverflow.com/questions/12897374/get-unique-values-from-a-list-in-python),
use `set(myList)`

## pandas

* [index error using axes](https://stackoverflow.com/questions/28442285/indexerror-when-plotting-subplots-in-pandas/40168092)
basically, it's because if it is `N * 1` or `1 * N` only one variable needs to be specified, so instead of
`myDataframe.plot(ax = [i,0])` use `myDataframe.plot(ax = [i])`

* to create a one row dataframe, need also to put each element as array, like

```python
df = pd.DataFrame({
    "myName":["one"],
    "myVal": [1]
})
```

doing this withoug an array, will generate an error message `ValueError: If using all scalar values, you must must pass an index`

* [convert dictionary to dataframe](https://stackoverflow.com/questions/18837262/convert-python-dict-into-a-dataframe) or [documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.from_dict.html)

* [multiple plots in different figures](https://www.kite.com/python/answers/how-to-plot-multiple-graphs-on-a-single-figure-using-pandas-in-python)

* [use list to select rows](https://stackoverflow.com/questions/12096252/use-a-list-of-values-to-select-rows-from-a-pandas-dataframe),
use 

```python
df_sub = df[df['A'].isin(myList)]
``` 

* [plot separately](https://stackoverflow.com/questions/22483588/how-can-i-plot-separate-pandas-dataframes-as-subplots)
something like:

```python
    fig = plt.figure()
    ax = fig.add_subplot(221)
    plt.plot(x,y)

    ax = fig.add_subplot(222)
    plt.plot(x,z)
    ...

    plt.show()
```

or similarly

```python
    plt.subplot(nrow, ncol, count)
    rvVariables.plot.hist(bins = bins ) 
```

* [add global titles](https://stackoverflow.com/questions/19614400/add-title-to-collection-of-pandas-hist-plots)

```python
    fig,axes = plt.subplots(nrow,ncol)
    fig.tight_layout(pad = 5.0)
    fig.suptitle("total number of particles chosen: " + str(numTrajectories), fontsize = 20)
```

* [how pandas use matplotlib](http://jonathansoma.com/lede/algorithms-2017/classes/fuzziness-matplotlib/how-pandas-uses-matplotlib-plus-figures-axes-and-subplots/)

* [as_index in groupby](https://stackoverflow.com/questions/41236370/what-is-as-index-in-groupby-in-pandas)

## numpy

* [numpy.select](https://numpy.org/doc/stable/reference/generated/numpy.select.html)

## matplotlib

* [x labels overlapping](https://stackoverflow.com/questions/26700598/matplotlib-showing-x-tick-labels-overlapping)

* [too many indices for array ](https://stackoverflow.com/questions/49809027/matplotlib-subplots-too-many-indices-for-array)
analogous problem in plotting in `pandas`, if the axes is one-dimensional, just use one dimension for the axes argument

* [global legends](https://stackoverflow.com/questions/7526625/matplotlib-global-legend-and-title-aside-subplots)

```python
    fig, axs = plt.subplots()
    fig.suptitle("Global title")
``` 

* [set legends for several lines](https://stackoverflow.com/questions/33322349/set-legend-for-plot-with-several-lines-in-python)

* [set spacing between plots](https://www.kite.com/python/answers/how-to-set-the-spacing-between-subplots-in-matplotlib-in-python)

```python
fig,ax = plt.subplots()
fig.tight_layout(pad = 5.0)
```

* [improve spacing between plots](https://stackoverflow.com/questions/6541123/improve-subplot-size-spacing-with-many-subplots-in-matplotlib)

* [equivalent of hold on](https://stackoverflow.com/questions/21465988/python-equivalent-to-hold-on-in-matlab)
the command `plt.hold(True)` is deprecated, now simply add plot use command `plt.plot(...)` and at the end `plt.show()` will plot all lines
into one figure

## re

* [split string into chars and numbers](https://stackoverflow.com/questions/430079/how-to-split-strings-into-text-and-number)

```python
r = re.compile("([a-zA-Z]+)([0-9]+)")
m = r.match("foobar231")
m.group(1) # 'foobar'
m.group(2) # '231'
```

## statistics

* [pandas normality test](https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/), [ways to test normality](https://towardsdatascience.com/6-ways-to-test-for-a-normal-distribution-which-one-to-use-9dcf47d8fa93)

* [linear regression](https://realpython.com/linear-regression-in-python/#implementing-linear-regression-in-python)

```python
model = LinearRegression().fit(x, y)
print('intercept:', model.intercept_)
print('slope:', model.coef_)  # the slope
```

## jupyter

* use `stdin` to import file written in `vim`

## To Learn

### git

* git pull a single file from master, maybe [here](https://stackoverflow.com/questions/3334475/git-how-to-update-checkout-a-single-file-from-remote-origin-master)

### awk

* awk modify variable, maybe [here](https://stackoverflow.com/questions/34866678/modify-a-shell-variable-inside-awk-block-of-code#:~:text=1%20Answer&text=No%20program%20%2D%2D%20in%20awk,modify%20its%20own%20variables%20itself.)

* set different delimiter, [here](https://stackoverflow.com/questions/51866972/can-i-use-a-different-delimiter-in-different-code-blocks-in-awk)

### windows

* [install python](https://www.howtogeek.com/197947/how-to-install-python-on-windows/) and [here](https://datatofish.com/add-python-to-windows-path/)

### python

#### general

* [python not working in command line](https://stackoverflow.com/questions/13596505/python-not-working-in-command-prompt)

#### os 

* [os.walk to certain level](https://stackoverflow.com/questions/42720627/python-os-walk-to-certain-level)

* [search specific files](https://www.kite.com/python/answers/how-to-search-for-specific-files-in-subdirectories-in-python)

#### pandas

* [rename columns](https://stackoverflow.com/questions/11346283/renaming-columns-in-pandas) or [here](https://stackoverflow.com/questions/29442370/how-to-correctly-read-csv-in-pandas-while-changing-the-names-of-the-columns)

I think sth like `pd.read_csv(fileName, header = 0, columns = myColumnName)`