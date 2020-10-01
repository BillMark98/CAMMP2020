
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

* [run python script](https://www.knowledgehut.com/blog/programming/run-python-scripts) or [here](https://realpython.com/run-python-scripts/#:~:text=To%20run%20Python%20scripts%20with,python3%20hello.py%20Hello%20World!)
Note that in python shell, using 


will cause the problem `python name __file__ is not defined` ,because the syntax:

```python
if __name__ == (__file__):
    do_something
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

```

## numpy

* [numpy.select](https://numpy.org/doc/stable/reference/generated/numpy.select.html)

## re

* [split string into chars and numbers](https://stackoverflow.com/questions/430079/how-to-split-strings-into-text-and-number)

```python
r = re.compile("([a-zA-Z]+)([0-9]+)")
m = r.match("foobar231")
m.group(1) # 'foobar'
m.group(2) # '231'
```