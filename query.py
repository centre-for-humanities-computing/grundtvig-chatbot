import random, argparse, sys, json
import numpy as np
from keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="hdf5 model file")
parser.add_argument("-s","--seed", default=12345)
parser.add_argument("-l","--length", default=200, help="Characters to sample")
parser.add_argument("-t","--temperature", default=0.8, help="Sampling temperature")
parser.add_argument("-q","--query", help="sampling text start query")
parser.add_argument("-v", "--vocab")
args = parser.parse_args()

np.random.seed(args.seed)

# load model
model = load_model('/Users/oliverjarvis/Arbejde/rnn-2/output/best.h5s')
# summarize model.
model.summary()

chars = []
# load chars
with open("/Users/oliverjarvis/Arbejde/rnn-2/tokens.json") as f:
    chars = json.loads(f.read())

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

def sample(preds, temperature=args.temperature):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_from_input(input_text):
    # Function invoked at end of each epoch. Prints generated text.
    print("Generating from input: ", input_text)

    generated = ''
    sentence = input_text
    generated += sentence

    for i in range(args.length):
        x_pred = np.zeros((1, len(input_text), len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.
        
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, args.temperature)
        next_char = indices_char[next_index]

        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()

generate_from_input("Kære mand. Godmorgen, har du det godt?  ")
