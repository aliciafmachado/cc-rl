from cc_rl.data.Analyzer import Analyzer

analyzer = Analyzer(datasets='emotions')
analysis = analyzer.analyze()

print('EXACT MATCH:')
print(analysis['exact_match'])
print('EXACT MATCH REWARD:')
print(analysis['exact_match_reward'])
print('HAMMING:')
print(analysis['hamming'])
print('HAMMING REWARD:')
print(analysis['hamming_reward'])
print('TIME:')
print(analysis['time'])
print('NUMBER OF NODES:')
print(analysis['n_nodes'])
