#!/usr/bin/env python3

import sys
import fileinput

filename = sys.argv[1]
known_words = [word.replace("\n", "") for word in open("query.wl", "r").readlines()]

with open("./querywords/peer_words.wl", "r") as f:
    unknown = [line.split(None, 1)[0] for line in f][0]

final_word = "EOS"

with fileinput.FileInput(filename, inplace=True) as file:
    for line in file:
        line_arr = line.split()
        new_line = line
        for word in line_arr:
            if word == final_word:
                break
            if word not in known_words:
                index = new_line.index(word)
                new_line = new_line[:index] + unknown + new_line[index + len(word) :]
        print(new_line, end="")
