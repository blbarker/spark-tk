def freakify(self):
    """
    augment column names

    >>> f = tc.frame.create([[1, "a"], [2, "b"]], [("number", int), ("letter", str)])

    >>> f.inspect()
    [#]  number  letter
    ===================
    [0]       1  a
    [1]       2  b

    >>> f.freakify()

    >>> f.column_names
    ['the_freaking_number', 'the_freaking_letter']

    """
    self.rename_columns(dict([(name, "the_freaking_" + name) for name in self.column_names]))

