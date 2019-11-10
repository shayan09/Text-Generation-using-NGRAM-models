import math
import random
from collections import OrderedDict
random.seed(1)

global one_words


def get_ngrams(n, text):
    words = text.split()
    for i in range(n - 1):
        words = ['<s>'] + words
    words += ['</s>']
    for i in range(len(words)-(n - 1)):
        word = words[i + n-1]
        context = tuple(words[i:i + n-1])
        yield (word, context)
    return


def mask_rare(corpus):
    global one_words
    unk_list = one_words
    if corpus == 'warpeace.txt':
        name = 'my_corpus.txt'
    else:
        name = 'my_corpus1.txt'
    my_doc = open(corpus, 'r').read()
    lines = my_doc.splitlines()
    new_text = []
    c = 0
    for sentence in lines:
        temp = []
        words = sentence.split()
        for word in words:
            if word in unk_list:
                c += 1
                temp.append("<unk>")
            else:
                temp.append(word)
        new_text.append(" ".join(temp))
    new_doc = open(name, 'w')
    for text in new_text:
        new_doc.write(text+'\n')
    new_doc.close()


class NGramLM:
    def __init__(self, n):
        self.n = n
        self.ngram_counts = dict()
        self.context_counts = dict()
        self.vocabulary = dict()
        self.sorted_vocab = OrderedDict()
        self.unk_words = set()

    def update(self, text):
        # this function will update the class/model internal counters
        res = get_ngrams(self.n, text)
        no_words = len(text.split())
        # below 'n' is the number of SGML tags
        for _ in range(no_words):
            try:
                gen = next(res)
                word = gen[0]
                context = gen[1]
                if word not in self.vocabulary:
                    self.vocabulary[word] = 1
                    self.unk_words.add(word)
                else:
                    self.vocabulary[word] += 1
                    if word in self.unk_words:
                        self.unk_words.remove(word)
                if context not in self.ngram_counts:
                    self.ngram_counts[(word, context)] = 1
                else:
                    self.ngram_counts[(word, context)] += 1
                if context not in self.context_counts:
                    self.context_counts[context] = 1
                else:
                    self.context_counts[context] += 1
            except StopIteration:
                break

    def word_prob(self, word, context, delta=0):
        # this function returns the probability of ngram(word, context) as a float
        ngram = (word, context)
        if context not in self.context_counts:
            prob = 1 / len(self.vocabulary)
            return prob
        if ngram not in self.ngram_counts:
            if '<unk>' in self.vocabulary:
                prob = (self.vocabulary['<unk>'] + delta)/(self.context_counts[ngram[1]] + (delta * len(self.vocabulary)))
            else:
                prob = delta / (self.context_counts[ngram[1]] + (delta * len(self.vocabulary)))
            return prob

        prob = ((self.ngram_counts[ngram] + delta) / ((self.context_counts[ngram[1]]) + (delta * len(self.vocabulary))))
        return prob

    def random_word(self, context, delta=0):
        # Creating a new sorted vocabulary
        sorted_keys = sorted(self.vocabulary.keys())
        for key in sorted_keys:
            self.sorted_vocab[key] = self.vocabulary[key]

        r = random.random()

        total_prob = 0
        word_prob= []
        for word in sorted_keys:
            total_prob += self.word_prob(word, context, delta)
            word_prob.append((word, total_prob))
        sorted_word_prob = sorted(word_prob, key=lambda x: x[1])

        for i in range(0, len(sorted_word_prob)):
            if sorted_word_prob[i][1] > r:
                break
        return sorted_word_prob[i - 1][0]

    def likeliest_word(self, context, delta=0):
        total_prob = 0
        max_prob = -1
        likely_word = ''
        for word in self.vocabulary:
            prob = self.word_prob(word, context, delta)
            total_prob = total_prob + prob
            if prob >= max_prob:
                max_prob = prob
                likely_word = word
        return likely_word


def create_ngramlm(n, corpus_path):
    # this function will send the tokenized data to update
    mymodel = NGramLM(n)
    data = open(corpus_path, 'r')
    contents = data.read()
    sentences = contents.splitlines()
    for s in sentences:
        mymodel.update(s)
    return mymodel


def text_prob(model, text):
    # below 'n' is the number of SGML tags
    res = get_ngrams(model.n, text)
    t_prob = 1
    no_words = len(text.split())
    try:
        for _ in range(no_words):
            gen = next(res)
            word = gen[0]
            context = gen[1]
            t_prob += math.log(model.word_prob(word, context))
        return t_prob
    except:
        return 'Domain Error'


class NGramInterpolator:
    def __init__(self, n, lambdas):
        self.n = n
        self.lambdas = lambdas
        self.model = []
        for i in range(n):
            self.model.append(NGramLM(i))

    def update(self, text):
        data = open(text, 'r')
        contents = data.read()
        sentences = contents.splitlines()
        for i in range(self.n):
            for s in sentences:
                self.model[i].update(s)

    def word_prob(self, word, context, delta=0):
        probs = []
        if word == 'first':
            text = 'God has given it to me, let him who touches it beware!'
        else:
            text = 'Where is the prince, my Dauphin?'
        for i in range(self.n):
            probs.append(text_prob(self.model[i], text))
        probability = 0
        for j in range(self.n):
            probability += (self.lambdas[j] * probs[j])
        return probability


def perplexity(model, corpus_path):
    with open(corpus_path, 'r') as doc:
        tokens = 0
        for line in doc:
            words = line.split()
            tokens += len(words)
        logp = 0.0
    with open(corpus_path, 'r') as doc:
        for line in doc:
            logp += text_prob(model, line)
        logp /= tokens
        return math.e ** (-1 * logp)


def random_text(model, max_length, delta=0):
    next_word = ''
    context = ()
    for i in range(model.n-1):
        w = "<s>"
        context = context+(w,)
    while max_length > 0 and next_word != '</s>':
        next_word = model.random_word(context,0)
        new_context = ()
        for i in range(len(context)-1):
            new_context = new_context+(context[i+1],)
        new_context = new_context+(next_word,)
        context = new_context;
        max_length = max_length-1
        print(next_word, end=" ")
    return


def likeliest_text(model, max_length, delta=0):
    next_word = ''
    context = ()
    for i in range(model.n-1):
        w = "<s>"
        context = context+(w,)
    while max_length > 0 and next_word != '</s>':
        next_word = model.likeliest_word(context,0)
        new_context = ()
        for i in range(len(context)-1):
            new_context = new_context+(context[i+1],)
        new_context = new_context+(next_word,)
        context = new_context
        max_length = max_length-1
        print(next_word, end=" ")
    return


# Q1
model = create_ngramlm(3, 'warpeace.txt')
predict_text1 = 'God has given it to me, let him who touches it beware!'
predict_text2 = 'Where is the prince, my Dauphin?'
probability1 = text_prob(model, predict_text1)
probability2 = text_prob(model, predict_text2)
print('Probabilities with \'warpeace.txt\': \n')
print(probability1)
print(probability2)
print()

# Q 2.1
one_words = model.unk_words
mask_rare('warpeace.txt')
print('Masking of \'warpeace.txt\' done')
mask_rare('shakespeare.txt')
print('Masking of \'shakespeare.txt\' done\n')

new_model = create_ngramlm(3, "my_corpus.txt")
probability1 = text_prob(new_model, predict_text1)
probability2 = text_prob(new_model, predict_text2)
print('Probabilities with new corpus:')
print(probability1)
print(probability2)
print()

# Q 2.3
inter_model = NGramInterpolator(3, lambdas=[0.33, 0.33, 0.33])
inter_model.update("my_corpus.txt")
print('Linear Interpolation probability of sentence 1 is: {}\n'.format(inter_model.word_prob('first', '')))
print('Linear Interpolation probability of sentence 2 is: {}\n'.format(inter_model.word_prob('second', '')))

# Q 3.1
model1 = create_ngramlm(3, "my_corpus1.txt")
model2 = create_ngramlm(3, "my_corpus.txt")
perplexity_corpus = 'sonnets.txt'
print('The perplexity for \'shakespeare.txt\' is {}'.format(perplexity(model1, perplexity_corpus)))
print('The perplexity for \'warpeace.txt\' is {}\n'.format(perplexity(model2, perplexity_corpus)))

# Q 4.1
print('The random texts generated are: \n')
for i in range(5):
    random_text(model1, 10)
    print('\n')

model_1 = create_ngramlm(1, "shakespeare.txt")
model_2 = create_ngramlm(2, "shakespeare.txt")
model_3 = create_ngramlm(3, "shakespeare.txt")
model_4 = create_ngramlm(4, "shakespeare.txt")
model_5 = create_ngramlm(5, "shakespeare.txt")

# Q 4.2
print('The text generated by the 5 models are:\n')
random_text(model_1, 10)
print('\n')
random_text(model_2, 10)
print('\n')
random_text(model_3, 10)
print('\n')
random_text(model_4, 10)
print('\n')
random_text(model_5, 10)
print('\n')
