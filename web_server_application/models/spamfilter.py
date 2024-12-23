import pickle

with open('../lib/models/spamfilter_model.pkl', 'rb') as f:
    model = pickle.load(f)

y_predict_loaded = model.predict_proba(["ko có class diagram nên ko bt code nó theo cái chiều nào đi cho dễ xem"])
print(y_predict_loaded)