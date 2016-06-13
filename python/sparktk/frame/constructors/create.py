from sparktk.lazyloader import implicit

def create(data, schema=None, tc=implicit):
    """creates a Frame from given data, schema"""
    if tc is implicit:
        implicit.error('tc')
    from sparktk.frame.frame import Frame
    return Frame(tc, data, schema)
