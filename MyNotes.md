
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

* [test type](https://stackoverflow.com/questions/2225038/determine-the-type-of-an-object)
use `type()` to tell direct type and `isinstance()` to find out the inheritance.

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

* [remove sublists from a list](https://stackoverflow.com/questions/2514961/remove-all-values-within-one-list-from-another-list/30353802)

```python
    [x for x in arr if x not in sublist]
```

* [delete element from a list suppress ValueError if element not exists](https://stackoverflow.com/questions/4915920/how-to-delete-an-item-in-a-list-if-it-exists) or [here](https://stackoverflow.com/questions/9915339/how-can-i-ignore-valueerror-when-i-try-to-remove-an-element-from-a-list)
either hard code, or convert to a set and use `myset.discare(elem)` or 

```python
try:
    arr.remove(elem)
except ValueError:
    pass
```

* [dictionary comprehension](https://www.datacamp.com/community/tutorials/python-dictionary-comprehension?utm_source=adwords_ppc&utm_campaignid=898687156&utm_adgroupid=48947256715&utm_device=c&utm_keyword=&utm_matchtype=b&utm_network=g&utm_adpostion=&utm_creative=229765585186&utm_targetid=aud-299261629574:dsa-473406574715&utm_loc_interest_ms=&utm_loc_physical_ms=9044818&gclid=Cj0KCQjw5eX7BRDQARIsAMhYLP9ayrB3oG8wubLGGadQEb6JsaZZiSH3isVBMsMDEd3rYM81e4pyLvYaArrSEALw_wcB)

```python
    temp = {key: f(key) for key in origList}
```

* formatting string for several varibles, need to specify the number

```python
a = 2
b = 3
print("hello: {:.2f}, yes:{1:.4f}".format(a,b))  # wrong!
```

will generate error:

```python
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: cannot switch from automatic field numbering to manual field specification
```

use instead `print("hello: {0:.2f}, yes:{1:.4f}".format(a,b))`

* [floating point epsilon](https://stackoverflow.com/questions/9528421/value-for-epsilon-in-python)

```python
import sys
sys.float_info.epsilon
```

also [interesting posts](https://stackoverflow.com/questions/34611858/machine-epsilon-in-python?lq=1) ,introduce `np.spacing` returns the next floating point

* [string startswith](https://www.tutorialspoint.com/python/string_startswith.htm)

```python
mystr.startswith("a")
```

* [find the index of a boolean array where value is True](https://stackoverflow.com/questions/36941294/find-the-index-of-a-boolean-array-whose-values-are-true)

```python
boolArr = np.array([ True, False, False,  True,  True, False])
np.where(boolArr)
boolArr.nonzero()
```

both returns `(array([0, 3, 4]),)`

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

* [read_csv index_col = 0,None,False](https://stackoverflow.com/questions/29862864/different-read-csv-index-col-none-0-false-in-pandas)

`index_col = 0` will treat the first column as index, whereas `None` and `False` will treat the first column as a data column and will add
an extra `Unnamed: 0` column for indexing.  Another interesting thing from the link is the usage:

```python
import io
import pandas as pd
t = """,a,b
0,hello,pandas"""
pd.read_csv(io.StringIO(t))
```

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

* [concatenate two dataframes](https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html),
  
```python
    pd.concat([df1,df2],axis = 1) # axis = 1, means concatenate horizontally
```

* [rename columns of a dataframe](https://www.datacamp.com/community/tutorials/python-rename-column?utm_source=adwords_ppc&utm_campaignid=898687156&utm_adgroupid=48947256715&utm_device=c&utm_keyword=&utm_matchtype=b&utm_network=g&utm_adpostion=&utm_creative=332602034352&utm_targetid=aud-748597547652:dsa-429603003980&utm_loc_interest_ms=&utm_loc_physical_ms=9044818&gclid=Cj0KCQjw5eX7BRDQARIsAMhYLP-_t7a5ryrHzsKN9XX5XP6XcjUyzhLY42C2yvbhl8EhiAClktFsv74aAqjhEALw_wcB) or [here](https://cmdlinetips.com/2018/03/how-to-change-column-names-and-row-indexes-in-pandas/)

use 

```python
    df.rename(columns = {'oldname1': 'newname1'}, inplace = True) # True will do the change in the original df, False otherwise
```

* [drop column and ignore error if column not existant](https://stackoverflow.com/questions/59116716/df-drop-if-it-exists)
use 

```python
df = df.drop(["some column"], axis = 1, error = "ignore")
```

* [update row with condition](https://stackoverflow.com/questions/36909977/update-row-values-where-certain-condition-is-met-in-pandas/36910033)
something like: 

```python
df.loc[df["colA"] == "someThing", ["colB","colC"]] = 1 
```

* [scatter plot for different groups](https://stackoverflow.com/questions/21654635/scatter-plots-in-pandas-pyplot-how-to-plot-by-category)

```python
groups = df.groupby("molecule")

for name, group in groups:
    plt.plot(group.chol_percentage, group.diffusion_coefficient, marker = 'o', linestyle = '', ms = 12, label = name)
plt.legend()
plt.show()
```

* [list as value](https://stackoverflow.com/questions/41500359/create-pandas-dataframe-with-list-as-values-in-rows/41501032)

```python
df = pd.DataFrame([{
        'name':"hi",
        "value":[1,2,3]
    }])
df
```

Output:

```bash
name	value
0	hi	[1, 2, 3]
```

* [apply mean on list, which is an entry in a pandas](https://stackoverflow.com/questions/26806054/how-to-use-lists-as-values-in-pandas-dataframe)

two methods,

1. divide lists into columns. And group by conditions to calculate the mean
2. use `apply` 

```python
df['listvalue'].apply(numpy.mean)
```

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

* [cannot perform reduce with flexible type](https://stackoverflow.com/questions/43442415/cannot-perform-reduce-with-flexible-type/43443554)
like:

```python
TypeError                                 Traceback (most recent call last)
<ipython-input-16-30c02975f4be> in <module>
----> 1 diffAna.plotAlpha(DOPC_alpha)

~/gruppe_1/students_code/pythonSource/diffusionAnalysis.py in plotAlpha(dfAlpha, ms)
    176     print(yerror)
    177     # plt.plot(x,y)
--> 178     plt.errorbar(x,y,yerr = [yerror,yerror] , fmt = ' ',ms = ms,ecolor= 'magenta')
    179     plt.xlabel("cholesterol_concentration")
    180     plt.xticks([0,25,50])
/usr/lib/python3/dist-packages/matplotlib/pyplot.py in errorbar(x, y, yerr, xerr, fmt, ecolor, elinewidth, capsize, barsabove, lolims, uplims, xlolims, xuplims, errorevery, capthick, data, **kwargs)
   2547         uplims=False, xlolims=False, xuplims=False, errorevery=1,
   2548         capthick=None, *, data=None, **kwargs):
-> 2549     return gca().errorbar(
   2550         x, y, yerr=yerr, xerr=xerr, fmt=fmt, ecolor=ecolor,
   2551         elinewidth=elinewidth, capsize=capsize, barsabove=barsabove,
/usr/lib/python3/dist-packages/matplotlib/__init__.py in inner(ax, data, *args, **kwargs)
   1599     def inner(ax, *args, data=None, **kwargs):
   1600         if data is None:
-> 1601             return func(ax, *map(sanitize_sequence, args), **kwargs)
   1602 
   1603         bound = new_sig.bind(ax, *args, **kwargs)
/usr/lib/python3/dist-packages/matplotlib/axes/_axes.py in errorbar(self, x, y, yerr, xerr, fmt, ecolor, elinewidth, capsize, barsabove, lolims, uplims, xlolims, xuplims, errorevery, capthick, **kwargs)
   3428                 xo, _ = xywhere(x, lower, noylims & everymask)
   3429                 lo, uo = xywhere(lower, upper, noylims & everymask)
-> 3430                 barcols.append(self.vlines(xo, lo, uo, **eb_lines_style))
   3431                 if capsize > 0:
   3432                     caplines.append(mlines.Line2D(xo, lo, marker='_',
/usr/lib/python3/dist-packages/matplotlib/__init__.py in inner(ax, data, *args, **kwargs)
   1599     def inner(ax, *args, data=None, **kwargs):
   1600         if data is None:
-> 1601             return func(ax, *map(sanitize_sequence, args), **kwargs)
   1602 
   1603         bound = new_sig.bind(ax, *args, **kwargs)
/usr/lib/python3/dist-packages/matplotlib/axes/_axes.py in vlines(self, x, ymin, ymax, colors, linestyles, label, **kwargs)
   1199 
   1200         if len(x) > 0:
-> 1201             minx = x.min()
   1202             maxx = x.max()
   1203             miny = min(ymin.min(), ymax.min())
/usr/lib/python3/dist-packages/numpy/core/_methods.py in _amin(a, axis, out, keepdims, initial, where)
     32 def _amin(a, axis=None, out=None, keepdims=False,
     33           initial=_NoValue, where=True):
---> 34     return umr_minimum(a, axis, None, out, keepdims, initial, where)
     35 
     36 def _sum(a, axis=None, dtype=None, out=None, keepdims=False,
TypeError: cannot perform reduce with flexible type
```

Usually, it's caused by there are some not numerical values in the x, or y, or yerror. Check all values are numerical values, in particular, not
string like `'50'`

## re

* [split string into chars and numbers](https://stackoverflow.com/questions/430079/how-to-split-strings-into-text-and-number)

```python
r = re.compile("([a-zA-Z]+)([0-9]+)")
m = r.match("foobar231")
m.group(1) # 'foobar'
m.group(2) # '231'
```

## statistics

* [scikit.metrics.mean_squared_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)

the `mean_squared_error` calulate `1/n * \sum_{i = 1}^{n} (x(i) - y(i))^2` in particular, to calculate the `sigma^2` for the linear regression,
remember to multiply ` n/(n-2)`

[slides for linear regression error analysis](https://www2.isye.gatech.edu/~yxie77/isye2028/lecture12.pdf)

* [pandas normality test](https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/), [ways to test normality](https://towardsdatascience.com/6-ways-to-test-for-a-normal-distribution-which-one-to-use-9dcf47d8fa93)

* [linear regression](https://realpython.com/linear-regression-in-python/#implementing-linear-regression-in-python)

```python
model = LinearRegression().fit(x, y)
print('intercept:', model.intercept_)
print('slope:', model.coef_)  # the slope
```

* remember to do a fit, need to add `()` before the `LinearRegression` 

```python
    7 model = LinearRegression.fit(tLog,msdLog)   # <---------- forget parenthes
    8 print("k:{0:4e}, b: {1:4e}".format(model.coef_, model.intercept_))
TypeError: fit() missing 1 required positional argument: 'y'
```

Correct version is `model = LinearRegression().fit(tLog,msdLog)`

* [normal quantile](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html)

```python
    quantile_95 = norm.ppf(0.95)
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

* [check nan](https://stackoverflow.com/questions/944700/how-can-i-check-for-nan-values)

* [python not working in command line](https://stackoverflow.com/questions/13596505/python-not-working-in-command-prompt)

```python
/gruppe_1/students_code/pythonSource/statisticsAnalysis.py in findLowUpIndex(arr, lowBound, upBound, startIndex, endIndex, userIsKing)
    690     if (startIndex < 0):
    691         tempStart = 0
--> 692     arrSlice = arr[tempStart : endIndex]
    693     arrSlice_low = min(arrSlice)
    694     arrSlice_max = max(arrSlice)
UnboundLocalError: local variable 'tempStart' referenced before assignment
```

#### os 

* [os.walk to certain level](https://stackoverflow.com/questions/42720627/python-os-walk-to-certain-level)

* [search specific files](https://www.kite.com/python/answers/how-to-search-for-specific-files-in-subdirectories-in-python)

#### pandas

* [rename columns](https://stackoverflow.com/questions/11346283/renaming-columns-in-pandas) or [here](https://stackoverflow.com/questions/29442370/how-to-correctly-read-csv-in-pandas-while-changing-the-names-of-the-columns)

I think sth like `pd.read_csv(fileName, header = 0, columns = myColumnName)`

* [add new columns and rows](https://www.geeksforgeeks.org/adding-new-column-to-existing-dataframe-in-pandas/)

```python
df["newcolumn"] = listLikeStructure # add column
df.insert(listLikeStructure)  # add new row
```

* [error bar](https://stackoverflow.com/questions/21469620/how-to-do-linear-regression-taking-errorbars-into-account)
also [here](https://towardsdatascience.com/the-quick-and-easy-way-to-plot-error-bars-in-python-using-pandas-a4d5cca2695d) and [here](https://stackoverflow.com/questions/23000418/adding-error-bars-to-grouped-bar-plot-in-pandas)

* [fill a column with values](https://stackoverflow.com/questions/34811971/how-do-i-fill-a-column-with-one-value-in-pandas/34811984)

* warning:
`A value is trying to be set on a copy of a slice from a DataFrame.`
[SettingWithCopyWarning](https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas) and  [here](https://www.dataquest.io/blog/settingwithcopywarning/)

#### numpy

* [interesting posts](https://stackoverflow.com/questions/34611858/machine-epsilon-in-python?lq=1) ,introduce `np.spacing` returns the next floating point

#### matplotlib

* [tick label scientifc](https://matplotlib.org/3.3.1/api/_as_gen/matplotlib.axes.Axes.ticklabel_format.html)

### theory

* [python mittag-leffler instability](https://stackoverflow.com/questions/48645381/instability-in-mittag-leffler-function-using-numpy), 
I think it's caused by the divergence of the series representation of the functioin for |z| > 1

* [confidence interval of a function](https://stats.stackexchange.com/questions/55441/how-to-calculate-the-confidence-interval-of-a-function-of-a-combination-of-two-l)

* richardson extrapolation