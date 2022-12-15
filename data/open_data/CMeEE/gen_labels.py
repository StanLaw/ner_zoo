import sys
import os
import json

filename = sys.argv[1]
labels = set()

with open(filename, 'r') as f:
	for line in f.readlines():
		tmp = json.loads(line.strip())
		for one in tmp["labels"]:
			labels.add(one)


savefile = os.path.join(os.path.dirname(filename), "labels.txt")
with open(savefile, 'w') as f:
	f.write("<E>\n")
	f.write("<B>\n")
	for one in labels:
		f.write(one + "\n")
