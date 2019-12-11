from tqdm import tqdm

TEST_FILE = 'data/test_pairs.txt'
NEW_FILE = 'data/test_pairs_1.txt'


def generate():
    with open(TEST_FILE) as file:
        lines = file.readlines()

    new_lines = []
    for line in tqdm(lines):
        tokens = line.split()
        imagepath1 = tokens[0]
        imagepath1 = imagepath1[:imagepath1.index('/')] + '/0.jpg'
        imagepath2 = tokens[1]
        type = int(tokens[2])

        new_line = '{} {} {}\n'.format(imagepath1, imagepath2, type)
        new_lines.append(new_line)

    with open(NEW_FILE, 'w') as file:
        file.writelines(new_lines)


if __name__ == "__main__":
    generate()
