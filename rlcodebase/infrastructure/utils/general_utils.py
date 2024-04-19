

def escape_bracket_globe(path):
    return path.replace('[', '[[').replace(']', ']]').replace('[[', '[[]').replace(']]', '[]]')


def merge_lists(ls):
    merged_list = []
    for l in ls:
        merged_list.extend(l)

    return merged_list
