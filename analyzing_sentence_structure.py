import nltk

# source: https://www.nltk.org/book/ch08.html
# Also some discussion of material from: https://web.stanford.edu/~jurafsky/slp3/12.pdf

# generative grammar; consider a "language" to be
# nothing more but an enormous collection of all
# grammatical sentences, and a grammar to be a
# formal notation that can be used to generate
# the members of this set:
# * rules a.k.a. productions, for listing out
#   the different ways to generate sentences;
# * non-terminal symbols in the CFG, representing
#   types of nodes that may dominate lexical items
#   or other nodes;
# * terminal symbols, i.e. lexical items
# Because productions may result in recursive forms,
# either in themselves or when combined, this
# perspective can account for the infinite
# possibilities of grammatical sentences.

# 1.2 Ubiquitous ambiguity

'''
"While hunting in Africa, I shot an elephant in
my pajamas. How he got into my pajamas, I don't
know."
- Animal Crackers (1930), a Groucho Marx film
'''

groucho_grammar = nltk.CFG.fromstring("""
  S   -> NP VP
  PP  -> P  NP
  NP  -> Det N | Det N PP | 'I'
  VP  -> V NP  | VP PP
  Det -> 'an' | 'my'
  N   -> 'elephant' | 'pajamas'
  V   -> 'shot'
  P   -> 'in'
  """)

groucho_sent = ['I', 'shot', 'an', 'elephant', 'in', 'my',
        'pajamas']
groucho_parser = nltk.ChartParser(groucho_grammar)
groucho_trees = list(groucho_parser.parse(groucho_sent))

# for tree in groucho_parser.parse(groucho_sent)
#    print(tree)

# Our context-free grammar allows 'sent' to be
# parsed in two different ways!

# 2. What's the use of Syntax?
# 2.1 Beyond n-grams

# Constituent structure
# * words combine with other words to form units
# * just like we can classify (POS-tagging) words,
#   we observe that a generative grammar in which
#   constituents belong to different classes can
#   account efficiently & adequately for various
#   grammatical phenomena

# substitutability
# well- vs. ill-formedness of coordinate structures

# 3. Context Free Grammar
# 3.1 A Simple Grammar

# left-hand side of 1st production:
# grammar's start-symbol
# * All well-formed trees must ultimately have
#   this symbol as their root label.
# * In NLTK - CFG's are defined in the
#   nltk.grammar module.

grammar1 = nltk.CFG.fromstring("""
  S   -> NP VP
  VP  -> V NP  | V NP PP
  PP  -> P NP
  V   -> "saw" | "ate" | "walked"
  NP  -> "John" | "Mary" | "Bob" | Det N | Det N PP
  Det -> "a" | "an" | "the" | "my"
  N   -> "man" | "dog" | "cat" | "telescope" | "park"
  P   -> "in" | "on" | "by" | "with"
""")

# sent = "Mary saw Bob".split()
# rd_parser = nltk.RecursiveDescentParser(grammar1)
# for tree in rd_parser.parse(sent):
#     print(tree)

'''
Just 1 parse is given - no structural ambiguity:

(S (NP Mary) (VP (V saw) (NP Bob)))
'''

# Interlude with nltk.app.rdparser() ;
# compile with 'C-c C-c' then run in interactive
# session to bring up the GUI.
# in 'Edit > Edit Grammar' (?) dropdown, can
# put in a rule for e.g.
# VP   -> V PP PP
# production
# (e.g. "ran from the park to the telescope")


# 3.2 Writing Your Own Grammars

'''
If you're interested in experimenting with writing
CFG's, you'll find it helpful to create and edit
your grammar in a text file, say
 mygrammar.cfg
You can then load it into NLTK and parse with it
as follows:
'''
grammar1 = nltk.data.load('file:mygrammar.cfg')

str1 = "Mary saw Bob"
str2 = "Mary saw the light"
str3 = "Bob knew that Mary saw the light"
str4 = "Bob believed the proof that Mary saw the light"

rd_parser = nltk.RecursiveDescentParser(grammar1)

def parse_with_grammar_1(str):
    sent = str.split()
    for tree in rd_parser.parse(sent):
        print(tree)

# 3.3 Recursion in Syntactic Structure
# A grammar is said to be *recursive* if a category
# occurring on the left hand side of a production
# also appears on the right hand side of some production.

'''
We just saw an example in the parse for str4!

S, the sentence category, appears on LHS of the top-level
production (i.e. the one that says that the root of every
well-formed sentence must be 'S') *and* it also appears
on RHS of the production 'SCOMP -> COMP S', the production
for complementized sentences:

S     -> NP   (V       (Det NCOMP SCOMP))
         Bob  believed the  proof (...)

SCOMP -> COMP S
         that Mary saw the light

'''

'''
That example was *indirect*, but there are also examples
of *direct recursion* on a category: e.g.
  Nom    -> Adj Nom
in the following:
'''

grammar2 = nltk.CFG.fromstring("""
  S  -> NP VP
  NP -> Det Nom | PropN
  Nom -> Adj Nom | N
  VP -> V Adj | V NP | V S | V NP PP
  PP -> P NP
  PropN -> 'Buster' | 'Chatterer' | 'Joe'
  Det -> 'the' | 'a'
  N -> 'bear' | 'squirrel' | 'tree' | 'fish' | 'log'
  Adj  -> 'angry' | 'frightened' |  'little' | 'tall'
  V ->  'chased'  | 'saw' | 'said' | 'thought' | 'was' | 'put'
  P -> 'on'
  """)

'''
(setting aside
4   Parsing With Context Free Grammar
4.1   Recursive Descent Parsing
4.2   Shift-Reduce Parsing
4.3   The Left-Corner Parser
4.4   Well-Formed Substring Tables
in the interest of time
)
'''

# jumping to -
# 6   Grammar Development
# 6.1   Treebanks and Grammars

# Brief intro for the Tree class used here.
'''
Handy resource on constituent-based syntactic parsing essentials,
including manipulation of Tree instance objects:

https://www.cs.bgu.ac.il/~elhadad/nlp16/NLTK-PCFG.html
'''
tag_str = '(S (NP (DT the) (NN cat)) (VP (VBD ate) (NP (DT a) (NN cookie))))'
tag_tree = nltk.Tree.fromstring(tag_str)
# tag_tree.draw()
# print(tag_tree)
# tag_tree.label()
# tag_tree[0] # tree's 1st child
# tag_tree[1] # tree's 2nd child
# tag_tree.height()
# tag_tree.leaves()

'''
>>> tag_tree[1]
Tree('VP', [Tree('VBD', ['ate']), Tree('NP', [Tree('DT', ['a']), Tree('NN', ['cookie'])])])
>>> tag_tree[1, 1]
Tree('NP', [Tree('DT', ['a']), Tree('NN', ['cookie'])])
>>> tag_tree[1, 1, 0]
Tree('DT', ['a'])
'''

# The corpus module defines the treebank corpus
# reader, which contains a 10% sample of the
# Penn Treebank corpus.
from nltk.corpus import treebank
t = treebank.parsed_sents('wsj_0001.mrg')[0]

'''
We can use this data to help develop a grammar.
For example, this program uses a simple filter to find verbs
that take sentential complements.
Assuming we already have a production of the form
VP -> Vs S, this information enables us to identify particular
verbs that would be included in the expansion of Vs 
'''
def filter(tree):
    '''
    filter is the predicate that, given a tree,
    returns whether or not the tree is a VP node
    containing an S among the nodes it dominates
    (so long as the input tree is an nltk.Tree)
    '''
    child_nodes = [child.label() for child in tree
                   if isinstance(child, nltk.Tree)]
    return (tree.label() == 'VP') and ('S' in child_nodes)

vcomp_subtrees = [subtree for tree in treebank.parsed_sents()
                          for subtree in tree.subtrees(filter)]

'''
e.g.

>>> vcomp_subtrees[435]

Tree('VP',
  [Tree('VBZ', ['declines']),
   Tree('S',
     [Tree('NP-SBJ', [Tree('-NONE-', ['*-1'])]),
      Tree('VP', [Tree('TO', ['to']),
                  Tree('VP', [Tree('VB', ['comment']),
                  Tree('PP-CLR', [Tree('IN', ['on']),
                  Tree('NP', [Tree('DT', ['the']),
                  Tree('NN', ['arrangement'])])])])])])])

This is using the UPenn tagset, as I was able to confirm by
using the built-in nltk.help.upenn_tagset(..) method
(there's one for each tagset supported within nltk library):

e.g.


>>> nltk.help.upenn_tagset('VBZ')
'''

# here's a filter that identifies trees
# a) labeled 'VP', i.e. verb phrases, and
# b) that have a child note 'S', i.e. dominate a sentential complement
#    to the VP verb
def filter_ncomp(tree):
    child_nodes = [child.label() for child in tree
                   if isinstance(child, nltk.Tree)]
    return (tree.label() == 'NP') and ('S' in child_nodes) 

ncomp_subtrees = [subtree for tree in treebank.parsed_sents()
                          for subtree in tree.subtrees(filter_ncomp)]

# Some characteristic nouns of this sort:
# appointment effort decision race chance option duty attempt right

'''
>>> len(ncomp_subtrees)
>>> 145
>>> ncomp_subtrees[10].leaves()
['an', 'appointment', '*', 'to', 'see', 'the', 'venerable', 'Akio', 'Morita', ',', 'founder', 'of', 'Sony']
>>> ncomp_subtrees[20].leaves()
['any', 'efforts', '*ICH*-1']
>>> ncomp_subtrees[30].leaves()
['a', 'decision', '*', 'to', 'switch', 'to', 'more', 'economical', 'production', 'techniques']
>>> ncomp_subtrees[40].leaves()
['a', 'race', '*ICH*-2']
>>> ncomp_subtrees[50].leaves()
['its', 'chance', '*', 'to', 'be', 'in', 'the', 'telephone', 'business']
>>> ncomp_subtrees[60].leaves()
['the', 'option', '*', 'to', 'redeem', 'the', 'shares', 'before', 'a', 'conversion', 'takes', 'place']
>>> ncomp_subtrees[70].leaves()
['a', 'duty', 'upon', 'the', 'government', '*', 'to', 'assure', 'easy', 'access', 'to', 'information', 'for', 'members', 'of', ...]
>>> ncomp_subtrees[80].leaves()
['some', 'breathtaking', 'attempts', '*ICH*-1']
>>> ncomp_subtrees[90].leaves()
['the', 'president', "'s", 'right', '*', 'to', 'perform', 'the', 'duties', 'and', 'exercise', ...]
>>> 
'''

'''
Discussing a worked problem using the nltk Penn-treebank subset.
How can I find verbs which pair with THAT-complementized sentences?
'''

# 1. Get all the labels that a 1-terminal subtree equal to ['that'] can ever be associated with
that_labels = set(subtree.label() for tree in treebank.parsed_sents()
                                  for subtree in tree.subtrees()
                                  if subtree[0] == 'the')

'''
>>> that_labels
{'NNP', 'DT', 'CD', 'JJ'}
'''

# 2. Okay, still not clear which of these corresponds to the complementizer use of 'that'
#    - check out some examples?
trees_having_that = [tree for tree in treebank.parsed_sents()
                          if 'that' in tree.leaves()]

# An example of what I was looking for:
tree_with_SBAR_IN_that = trees_having_that[59]
'''
>>> list(trees_having_that[59].subtrees(lambda s: 'that' in s.leaves()))
[Tree('S', [Tree('CC', ['Yet']), Tree('NP-SBJ', [Tree('JJ', ['many']), Tree('NNS', ['economists'])]), Tree('VP', ...)
>>> tree_with_SBAR_IN_that = trees_having_that[59]
>>> tree_with_SBAR_IN_that.draw()
This tree (which is the tree of a full sentence in the corpus being treebanked) is seen to have a subtree
of form:
(VP
  (VBG predicting)
  (SBAR
    (IN that)
    (S ..)
  )
)
so in the end, we're looking for VP's that dominate an SBAR in our treebank.
'''

# 3. subtree filter for VP's dominating an SBAR
def filter_vp_dom_sbar(tree):
    child_nodes = [child.label() for child in tree
                   if isinstance(child, nltk.Tree)]
    return (tree.label() == 'VP') and ('SBAR' in child_nodes) 

vdoms_subtrees = [subtree for tree in treebank.parsed_sents()
                          for subtree in tree.subtrees(filter_vp_dom_sbar)]

'''
By the way, what *are* these part-of-speech tags in the first place?
Where do they come from, and how can we look up what each tag
stands for?
'''

pos_sample = "Many people know that 4 out of 5 dentists recommend sugarless chewing gum"
pos_words = nltk.word_tokenize(pos_sample)
pos_tags = nltk.pos_tag(pos_words)
# nltk.help.upenn_tagset(..)

'''
6.2 Pernicious Ambiguity

From an initial, rudimentary grammar, you can build out a treebank
by using the grammar to attempt parses for sentences in the corpus,
having human supervision to correct the parses and add rules as needed.

Unfortunately, as the coverage of the grammar increases and the length
of the input sentences grows, the number of parse trees grows rapidly.
In fact, it grows at an astronomical rate.

This is the point made in Jurafsky as follows:

'The sentences in a treebank implicitly constitute a grammar of the
language represented by the corpus being annotated. [...] The grammar
used to parse the Penn Treebank is relatively flat, resulting in
very many and very long rules. For example, among the approximately
4,500 different rules for expanding VPs are separate rules for PP
sequences of any length and every possible arrangement of verb
arguments: [...] Various facts about the treebank grammars, such as
their large numbers of flat rules, pose problems for probabilistic
parsing algorithms. For this reason, it is common to make various
modifications to a grammar extracted from a treebank. We discuss these
further in Chapter 14.' (pgs. 17 - 18)
'
'''

# "Let's explore this issue with the help of a simple example" (nltk book)
fish_gram = nltk.CFG.fromstring("""
S -> NP V NP
NP -> NP Sbar
Sbar -> NP V
NP -> 'fish'
V -> 'fish'
""")

'''
"Fish fish." (Fish like to fish.)
"Fish fish fish." (Fish like to fish for other fish.)
You could even have (indexing for ease of discussion:)
"Fish1 fish2 fish3 fish4 fish5 ."
(Fish1 that other fish2 fish3 for are in the habit
of fishing4 fish5 themselves.)
'''

# We'll use the NLTK chart parser this time:
tokens = ["fish"] * 5
cp = nltk.ChartParser(fish_gram)
fish_parses = [tree for tree in cp.parse(tokens)]

'''
>>> len(fish_parses)
2
>>> fish_parses[0].draw()
>>> fish_parses[1].draw()

FACT: For a fish-sentence of length 23, the number of possible parse trees
using our grammar is over 208K (!)
Structural ambiguity (illustrated by the fish example), as well as lexical
ambiguity (ubiquitous in natural language), are problems for broad-coverage
parsing.
'''

'''
FOLLOW-UP:
* parsing techniques / algorithms?
* probabilistic context free grammars?
* lexical-functional / combinatory categorial grammar?
'''
