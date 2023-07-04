from MultiLayerPerceptron import MLP_treinamento,MLP_teste
import os
from Interface import graf
currentDir = os.getcwd()
#MLP_treinamento(ampdigitos=90, vsai=10, entradas=266, neur=250, limiar=0.0, alfa=0.005, errotolerado=0.05,currentDir=currentDir)
#print(MLP_teste(path='9_157',vsai=10,neur=250,limiar=0.0,currentDir=currentDir))

graf(currentDir)
