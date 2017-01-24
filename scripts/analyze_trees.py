import pandas as pd
import sys
import pydot
import argparse
from PIL import Image
from tqdm import tqdm


def convert_binary_bracketing(parse):
    transitions = []
    tokens = []
    for word in parse.split(' '):
        if word[0] != "(":
            if word == ")":
                transitions.append(1)
            else:
                # Downcase all words to match GloVe.
                tokens.append(word.lower())
                transitions.append(0)
    return tokens, transitions


def create_tree(words, transitions):
    template_start = """
    digraph G {
        nodesep=0.4; //was 0.8
        ranksep=0.5;
    """
    template_end = """
    }
    """
    template = ""
    template += template_start
    buf = list(reversed(words))
    stack = []
    leaves = []
    for i, t in enumerate(transitions):
        if t == 0:
            stack.append((i+1,t))
            leaves.append(str(i+1))
            template += '{node[label = "%s"]; %s;}\n' % (str(buf.pop()), str(i+1))
        else:
            right = stack.pop()
            left = stack.pop()
            top = i + 1
            stack.append((top, (left, right)))
            template += "{} -> {};\n".format(top, left[0])
            template += "{} -> {};\n".format(top, right[0])
    template += "{rank=same; %s}" % ("; ".join(leaves))
    template += template_end
    return stack, template


def print_tree(words, transitions, output_file):
    _, tree = create_tree(words, transitions)
    graphs = pydot.graph_from_dot_data(tree)
    with open(output_file, 'wb') as f:
        f.write(graphs[0].create_jpeg())
    return graphs[0]


def combine_trees(files, output_file):
    images = map(Image.open, files)
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('L', (total_width, max_height))
    new_im.paste(255, (0, 0, total_width, max_height))

    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]

    new_im.save(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument('--no-combine', dest="combine", action="store_false", default=True)
    args = parser.parse_args()

    print(args)

    with open(args.input) as f:
        for i, line in tqdm(enumerate(f)):
            line = line.strip()

            given, predicted = line.split('\t')

            # Given
            given_output_file = "{}/given.{:05}.jpg".format(args.out_dir, i)
            words, transitions = convert_binary_bracketing(given)
            print_tree(words, transitions, given_output_file)

            # Predicted
            predicted_output_file = "{}/predicted.{:05}.jpg".format(args.out_dir, i)
            words, transitions = convert_binary_bracketing(predicted)
            print_tree(words, transitions, predicted_output_file)
            
            if args.combine:
                files = [given_output_file, predicted_output_file]
                combined_output_file = "{}/combined.{:05}.jpg".format(args.out_dir, i)
                combine_trees(files, combined_output_file)
