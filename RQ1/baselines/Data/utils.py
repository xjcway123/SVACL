import re
# regex to remove empty lines
def remove_empty_lines(text):
    return re.sub(r'^$\n', '', text, flags=re.MULTILINE)


# regex to remove comments from a file
def remove_comments(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)


# regex to remove space before newLine character
def remove_space_before_newline(text):
    return re.sub(r'\s+$', '', text, flags=re.MULTILINE)


# regex to remove space after newLine character
def remove_space_after_newline(text):
    return re.sub(r'^\s+', '', text, flags=re.MULTILINE)