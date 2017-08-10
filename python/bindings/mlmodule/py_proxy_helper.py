def get_matlab_compatible_properties(obj):
    vs = vars(obj)
    return [v for v in vs if not v.startswith('_')]
