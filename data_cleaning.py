import collections
import matplotlib.pyplot as plt
import os
import pickle
import re
import time

from collections import defaultdict
from pprint import pprint

input_file = "Wikipedia_LLM_orig.txt"
output_file = "Wikipedia_LLM_clean.txt"
token_file = "Wikipedia_LLM_tokens.pickle"
data_dir = "data"


start_time = time.time()
statistics = defaultdict(lambda: 0)
token_count = defaultdict(lambda: 0)


def remove_latex(line) -> str:
    def _one_char_case(matchobj) -> str:
        statistics['LaTeX_one_char_fixed'] += 1
        return matchobj.group(0)[-2]

    # Deal with the one char LaTeX
    line = re.sub(r'{\\displaystyle [a-zA-Z]}', _one_char_case, line)

    # Skip / Remove lines which still have laTex. Formulas are too complicated for now.
    if line.find('{\\displaystyle') != -1:
        statistics['latex_lines_removed'] += 1
        return ''

    # Return the fixed and cleaned line
    return line


def remove_citation(line) -> str:
    line = re.sub(r'\[([0-9]|[1-9][0-9]|[1-9][0-9][0-9])\]', '', line)
    line = line.replace("[citation needed]", " ")
    line = line.replace("[unreliable source?]", " ")
    line = line.replace("[better source needed]", " ")

    #print("Line without references:\n", line)
    return line


def split_into_sentence(line) -> list:
    # Save some special wordings and abbreviation to make them survive the splitting into sentences:
    line = line.replace('u.s.', 'u_s_')
    line = line.replace('et al.', 'et_al')
    line = line.replace('i.e.', 'i_e_')
    line = line.replace('e.g.', 'e_g_')

    # Split into sentences:
    sentences = re.split('\\. |\\? |! ', line)

    # Remove dots which survived (end of line/paragraph mostly)
    for i in range(len(sentences)):
        while len(sentences[i]) > 1 and sentences[i][-1] == '.':
            sentences[i] = sentences[i][:-1]

    # Bring back i.e., e.g. and et_al.:
    sentences = [sentence.replace('et_al', 'et al.').replace('i_e_', 'i.e.') \
                     .replace('e_g_', 'e.g.').replace('u_s_', 'u.s.')
                 for sentence in sentences]
    # Make sure, every sentence ends with a . bu

    return sentences

def punctuation_to_token(sentence) -> str:
    def _one_char_case(matchobj) -> str:
        statistics['punctuation_count'] += 1
        return ' ' + matchobj.group(0)[0] + ' '

    #
    operators = [',', ';', ':', '-', '"', "'", '(', ')', '[', ']', '$', '%', '?', '!']
    pattern = '|'.join(map(re.escape, sorted(operators, reverse=True)))
    sentence = re.sub(pattern, _one_char_case, sentence)

    # Remove additional white spaces
    items = sentence.split(' ')
    tokens = []
    for item in items:
        if not item or len(item) < 1:
            continue
        token_count[item] += 1
        tokens.append(item)

    sentence = ' '.join(tokens)

    return sentence


def process_input_line(line) -> list:
    #print("Incoming line:\n", line)

    # Simplifications!
    line = line.lower()
    # Tab to space
    line = line.replace('\t', ' ')
    # To keep it simple, we keep only one type of "hyphen"
    line = line.replace("—", "-")

    # Remove references
    line = remove_citation(line)

    # Create sentences
    sentences = split_into_sentence(line)

    # Deal with LaTeX:
    sentences = [remove_latex(sentence) for sentence in sentences]

    # deal with punctuation (Satzzeichen)
    sentences = [punctuation_to_token(sentence) for sentence in sentences]

    # Strip white spaces and remove empty lines
    sentences = [sentence.strip() for sentence in sentences if len(sentence)>5]

    return sentences


sentence_count = 0
sentence_length = collections.Counter()
with open(os.path.join(data_dir, input_file), 'r') as oif:
    with open(os.path.join(data_dir, output_file), 'w+') as oof:
        i = 0
        while in_line := oif.readline():
            in_line = in_line.strip()
            out_lines = process_input_line(in_line)
            for oline in out_lines:

                n_tokens = oline.count(' ') + 1
                if  n_tokens < 4:
                    statistics['short_lines'] += 1
                    continue
                sentence_length[n_tokens] += 1

                if n_tokens > 70:
                    print("Long long sentence:")
                    print(oline)

                # Keep only sentences (lines) with at lest four word
                oof.write(oline + "\n")
                sentence_count += 1

# Create token dict and save it
token_to_id = {key: i+1 for i, key in enumerate(sorted(token_count.keys()))}
token_to_id['.'] = 0        # End of Line Marker!

with open(os.path.join(data_dir, token_file), 'wb') as of:
    pickle.dump(token_to_id, of)


#pprint(token_to_id)
#pprint(token_count)

pprint(statistics)
#pprint(sentence_length)

plt.figure()
plt.title("Sentence length (num tokens per line)")
plt.bar(sentence_length.keys(), sentence_length.values())


print("We have a total of", sum(token_count.values()), "tokens.")
print("We have", len(token_count), "different tokens.")
print("The form", sentence_count, "sentences.")

print("Script runtime:", time.time() - start_time, "seconds")
plt.show()
