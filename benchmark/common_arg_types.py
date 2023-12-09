def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def list_of_floats(arg):
    return list(map(float, arg.split(',')))

def list_of_strings(arg):
    return arg.split(',')
