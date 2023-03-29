

def escape_bracket_globe(path):
    return path.replace('[', '[[').replace(']', ']]').replace('[[', '[[]').replace(']]', '[]]')
