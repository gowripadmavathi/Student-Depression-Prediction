import pickle
model=pickle.load(open("model.pkl","rb"))
scaler=pickle.load(open("scaler.pkl","rb"))
print(model)
print(scaler)
print(model.get_params())