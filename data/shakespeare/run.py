import sys
import re
for line in sys.stdin:
    line = re.sub("\s+", " ", line)
    line = line.strip()
    line = re.sub("<A NAME=speech.+?><b>(.+?)<.+", r"[\1]", line)
    line = re.sub("<A NAME=[0-9].+?>", "\t", line)
    line = re.sub("<p><i>(.+?)<.+", r"\t(\1)", line)
    line = re.sub("<.+?>", "", line)
    line = re.sub("\t +","\t", line)
    if line == "":
        continue
    print(line)
