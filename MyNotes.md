
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
Note that in python shell, using will cause the problem `python name __file__ is not defined` ,because the syntax:

```python
if __name__ == (__file__):
    do_something
```

* [test float equal](https://www.quora.com/How-does-one-correctly-compare-two-floats-in-Python-to-test-if-they-are-equal)

```python
abs(a-b) < epsilon
```

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

## jupyter

* use `stdin` to import file written in `vim`