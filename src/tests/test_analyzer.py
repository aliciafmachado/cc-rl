from cc_rl.data.Analyzer import Analyzer

analyzer = Analyzer(datasets='emotions')
analysis = analyzer.analyze()

print(analysis['exact_match'])
print(analysis['hamming'])
print(analysis['reward'])
print(analysis['time'])
print(analysis['n_nodes'])
