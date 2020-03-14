import json
import re
import string


with open("training-data/utils/samples/js_MDN_Code.json") as f:
    code_blocks_list = json.load(f)

for code_blocks in code_blocks_list:
    for code_block in code_blocks[1:]:
        code_block = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', "URL", code_block)
        code_block = re.sub(
            r"(\/\*([\s\S]*?)\*\/)|(\/\/(.*)$)", "", code_block, flags=re.M
        )
        code_block = code_block.replace("\n ", " ")
        code_block = re.sub(r" \n(.)", r"\g<1>", code_block)
        code_block = re.sub(r"\n(}.*|\).*)", r"\g<1>", code_block)
        # code_block = re.sub(r"(.)\n(.)", r"\g<1> \g<2>", code_block)
        code_block = "BOS " + code_block.translate(
            str.maketrans({key: " {0} ".format(key) for key in string.punctuation})
        )
        code_block = code_block.replace("\n", " EOS\nBOS ")
        code_block = re.sub(r" {2,}", " ", code_block).rstrip() + " EOS\n"
        code_block = code_block.replace("BOS EOS\n", "")

        print(code_block)
