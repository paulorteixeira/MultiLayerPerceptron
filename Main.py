from MultiLayerPerceptron import MLP_treinamento,MLP_teste

MLP_treinamento(ampdigitos=90, vsai=10, entradas=266, neur=250, limiar=0.0, alfa=0.005, errotolerado=0.05)
print(MLP_teste(path='9_157',vsai=10,neur=250,limiar=0.0))