import pickle as pkl

results = {}

with open('senteval_results/baseline', 'rb') as file:
    results['baseline'] = pkl.load(file)

with open('senteval_results/lstm', 'rb') as file:
    results['lstm'] = pkl.load(file)

with open('senteval_results/bilstm', 'rb') as file:
    results['bilstm'] = pkl.load(file)

with open('senteval_results/maxbilstm', 'rb') as file:
    results['maxbilstm'] = pkl.load(file)

encoder_types = ['baseline', 'lstm', 'bilstm', 'maxbilstm']

print("Results on the STS14 multilingual textual similarity task:")

for encoder_type in encoder_types:
    print("\n############################")
    print(encoder_type.upper(), "encoder:")
    for task in results[encoder_type]:
        print('\n' + task + '-------------\n')
        for measure in results[encoder_type][task]:
            print(measure + ":", results[encoder_type][task][measure])
    print()
