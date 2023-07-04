import numpy as np
import random as rd
import matplotlib.pyplot as plt
import os


def MLP_treinamento(ampdigitos: int, vsai: int,  entradas:int, neur:int, limiar:float, alfa:float, errotolerado:float):
    currentDir = os.getcwd()
    os.chdir(currentDir+'\\DadosParaTreino')

    randomSeed = rd.randint(0,1000000)
    rd.seed(randomSeed)

    listaciclo   = []
    listaerro    = []
    listaVarErro = []

    amostras    = ampdigitos*vsai


    x  = np.zeros((amostras,entradas))
    k2 = '_'
    k4 = '.txt'
    contagem = 0
    ordem = np.zeros(amostras)

    def f(x):
        return 2*x-1

    for m in range(vsai):
        k1 = str(m)
        for n in range(ampdigitos):
            k3a = n+1
            k3 = str(k3a)
            nome = k1+k2+k3+k4
            entrada = np.loadtxt(nome)
            x[contagem,:] = f(entrada[:])
            ordem[contagem] = m
            contagem = contagem+1

    ordem = ordem.astype('int')

    t = [[ 1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [-1, 1,-1,-1,-1,-1,-1,-1,-1,-1],
        [-1,-1, 1,-1,-1,-1,-1,-1,-1,-1],
        [-1,-1,-1, 1,-1,-1,-1,-1,-1,-1],
        [-1,-1,-1,-1, 1,-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1, 1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1,-1, 1,-1,-1,-1],
        [-1,-1,-1,-1,-1,-1,-1, 1,-1,-1],
        [-1,-1,-1,-1,-1,-1,-1,-1, 1,-1],
        [-1,-1,-1,-1,-1,-1,-1,-1,-1, 1]]


    vanterior = np.zeros((entradas,neur))
    aleatorio = 0.2

    for i in range(entradas):
        for j in range(neur):
            vanterior[i][j] = rd.uniform(-aleatorio,aleatorio)

    v0anterior = np.zeros((1,neur))
    for i in range(neur):
        v0anterior[0][j] = rd.uniform(-aleatorio, aleatorio)


    wanterior = np.zeros((neur,vsai))
    for i in range(neur):
        for j in range(vsai):
            wanterior[i][j] = rd.uniform(-aleatorio,aleatorio)

    w0anterior = np.zeros((1,vsai))
    for i in range(vsai):
        w0anterior[0][j] = rd.uniform(-aleatorio, aleatorio)


    vnovo       = np.zeros((entradas,neur))
    v0novo      = np.zeros((1,neur))
    wnovo       = np.zeros((neur,vsai))
    w0novo      = np.zeros((1,vsai))

    zin         = np.zeros((1,neur))
    z           = np.zeros((1,neur))

    deltinhak   = np.zeros((vsai,1))
    deltaw0     = np.zeros((vsai,1))
    deltinha    = np.zeros((1,neur))
    xaux        = np.zeros((1,entradas))
    h           = np.zeros((vsai,1))
    target      = np.zeros((vsai,1))
    deltinha2   = np.zeros((neur,1))

    ciclo = 0
    errototal = 10000000

    variacaoerro = 1000

    while errotolerado<errototal:
        errototal = 0
        
        for padrao in range(amostras):
            for j in range(neur):
                zin[0][j] = np.dot(x[padrao,:],vanterior[:,j])+v0anterior[0][j]
            z = np.tanh(zin)
            yin = np.dot(z,wanterior) +w0anterior
            y = np.tanh(yin)


            for m in range(vsai):
                h[m][0] = y[0][m]

            for m in range(vsai):
                target[m][0] = t[m][ordem[padrao]]

            errototal = errototal+np.sum(0.5*((target-h)**2))

            
            deltinhak = (target-h)*(1+h**2)     
            deltaw = alfa*(np.dot(deltinhak,z))
            deltaw0 = alfa*deltinha
            deltinhain = np.dot(np.transpose(deltinhak),np.transpose(wanterior))
            deltinha = deltinhain*(1+z**2)  

            for m in range(neur):
                deltinha2[m][0] = deltinha[0][m]
            for k in range(entradas):
                xaux[0][k] = x[padrao][k]
            deltav = alfa*np.dot(deltinha2,xaux)
            deltav0 = alfa*deltinha

            vnovo   = vanterior+np.transpose(deltav)
            v0novo  = v0anterior +np.transpose(deltav0)
            wnovo   = wanterior+np.transpose(deltaw)
            w0novo  = w0anterior +np.transpose(deltaw0)


            vanterior  = vnovo
            v0anterior = v0novo
            wanterior  = wnovo
            w0anterior = w0novo

        ciclo = ciclo +1 
        listaciclo.append(ciclo)
        listaerro.append(errototal)
        listaVarErro.append(errototal-variacaoerro)
        print('Ciclo\t Erro\t\t\t varerr')
        print(ciclo,'\t', errototal,'\t\t',errototal-variacaoerro)
        variacaoerro = errototal

    plt.plot(listaciclo,listaerro)
    plt.plot(listaciclo,listaVarErro)
    plt.xlabel('ciclo')
    plt.ylabel('erro')
    plt.show()

    
    contcerto = 0
    cont = 0
    for nmn in range(vsai):
        nmn = rd.randint(0,9)
        for ind in range(150 - ampdigitos):
            nomeTeste = str(nmn)+'_'+str(ampdigitos+ind)+'.txt'
            xteste = f(np.loadtxt(nomeTeste))

            for m2 in range(vsai):
                for n2 in range(neur):
                    zin[0][n2] = np.dot(xteste,vanterior[:,n2])+v0anterior[0][n2]
                z = np.tanh(zin)
                yin = np.dot(z,wanterior)+w0anterior
                y = np.tanh(yin)

            for j in range(vsai):
                if y[0][j]>=limiar:
                    y[0][j] = 1
                else:
                    y[0][j] = -1
            soma = np.sum(y[0]-target)
            if(soma==0):
                contcerto=contcerto+1
            cont = cont+1

    taxa = contcerto/cont
    print('\n\ntaxa de acerto: \t',taxa*100,'%')
    print('random seed',randomSeed)

    os.chdir(currentDir+'\\ValoresPesos')
    np.savetxt('va.txt',   vanterior, delimiter =', ')   
    np.savetxt('v0a.txt', v0anterior, delimiter =', ') 
    np.savetxt('wa.txt',   wanterior, delimiter =', ')    
    np.savetxt('w0a.txt', w0anterior, delimiter =', ') 


def MLP_teste(path: str, vsai: int,neur: int,limiar: int):
    currentDir = os.getcwd()
    os.chdir(currentDir+'\\ImgTeste')
    def f(x):
        return 2*x-1
    nomeTeste = path+'.txt'
    xteste = f(np.loadtxt(nomeTeste))

    def f(x):
        return 2*x-1
    
    os.chdir(currentDir+'\\ValoresPesos')

    zin         = np.zeros((1,neur))

    vanterior   = np.loadtxt('va.txt',delimiter=",")
    v0anterior  = np.loadtxt('v0a.txt',delimiter=",")
    wanterior   = np.loadtxt('wa.txt',delimiter=",")
    w0anterior  = np.loadtxt('w0a.txt',delimiter=",")

    for m2 in range(vsai):
        for n2 in range(neur):
            zin[0][n2] = np.dot(xteste,vanterior[:,n2])+v0anterior[0][n2]
        z = np.tanh(zin)
        yin = np.dot(z,wanterior)+w0anterior
        y = np.tanh(yin)

    for j in range(vsai):
        if y[0][j]>=limiar:
            y[0][j] = 1
        else:
            y[0][j] = -1
    
    return(y[0,:])