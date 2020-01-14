# Abby Bechtel
# Grace Tsai
# Karl Hickel

#!/usr/bin/env python

"""
This program reads in a text file containing lyrics and creates a first order Markov Model to generate new  song lyrics.
It does this by creating a dictionary containing each word found in the lyric document as a key and a list of words that have been found to follow the key word as the value. A random word is chosen to initialize the model and then the dictionary is used to randomly select the next word taking into account the probability that it is the next word.
"""

import numpy as np
import string

def process(fileName):
    """Reads the file and returns a processed list of words from the file"""
    file = open(fileName , "r")
    text = [word.translate(str.maketrans('', '', string.punctuation)) for line in file for word in line.lower().split()]
    return text

def bigrams(text):
    #Source: https://towardsdatascience.com/simulating-text-with-markov-chains-in-python-1a27e6d13fc6
    for i in range(len(text)-1):
        yield (text[i], text[i+1])


def genBiDict(text):
    """Takes a list of words as an argument and returns a dictionary of all the words in the list and the corrisponding words that follow"""
    wordBigrams = bigrams(text)
    biDict = {}

    for word1, word2 in wordBigrams:
        if word1 in biDict.keys():
            biDict[word1].append(word2)
        else: 
            biDict[word1]  = [word2]
    return biDict
            
def genSong(bigramDict, text):
    """Randomly generates a song given the dictionary of bigrams and a list of words. Initializes with a random word"""
    song = [np.random.choice(text)]
    nWords = 75
    for i in range(nWords):
        song.append(np.random.choice(bigramDict[song[-1]]))
    lyrics = ' '.join(song)
    
    return lyrics
    
if __name__ =="__main__":
    import sys
    text = process(sys.argv[1])
    probDict = genBiDict(text)
    print(genSong(probDict, text))
