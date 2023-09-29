import os

def tree(path, max_files):
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 4 * level
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        if len(files) > max_files:
            print('{}[{} entries exceeds filelimit, not opening dir]'.format(subindent, len(files)))
        else:
            for f in files:
                print('{}{}'.format(subindent, f))

tree('data/dlib', 10)
