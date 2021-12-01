from sys import argv

with open(argv[1]) as f:
	lines = [i.strip().split(',')[:4] for i in f.readlines()]

a2b = {}
b2a = {}

from tqdm import tqdm

for i, (a, b, c, d) in enumerate(tqdm(lines)):
	if 'West' in d: continue
	if a2b.get(a, b) != b:
		print('a2b', i + 1, a, b, c, d)
		# quit()
	if b2a.get(b, a) != a:
		print('b2a', i + 1, a, b, c, d)
		# quit()
	a2b[a] = b
	a2b[b] = a
