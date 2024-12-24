import pickle

with open('../lib/models/spamfilter_model.pkl', 'rb') as f:
    model = pickle.load(f)