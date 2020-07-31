def get_matlab_compatible_properties(obj):
    names = dir(obj)
    return [v for v in names if not v.startswith('_')]
