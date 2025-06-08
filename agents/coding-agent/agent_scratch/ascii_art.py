import random

def generate_ascii_art():
    art = ''
    for _ in range(10):
        line = ''
        for _ in range(10):
            if random.random() < 0.5:
                line += '*'
            else:
                line += ' '
        art += line + '\n'
    return art