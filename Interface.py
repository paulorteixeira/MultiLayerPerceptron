import pygame
import numpy as np
from PIL import Image as im
from MultiLayerPerceptron import  MLP_teste
import os


def graf(currentDir:str):

    
    w = 500
    h = 500

    pygame.init()

    # Set up the drawing window

    screen = pygame.display.set_mode([w, h])

    # Run until the user asks to quit

    running = True

    font = pygame.font.Font('freesansbold.ttf', 50)
    text = font.render('fps', True, (0,0,0), (255,255,255))
    textRect = text.get_rect()
    textRect.center = (70,30)

    rectangle = pygame.rect.Rect(0, 0, 90, 90)
    desenhando = False

    resposta = []

    while running:


        # Did the user click the window close button?
        
        for event in pygame.event.get():
            
            if event.type == pygame.QUIT:

                running = False
            
            if event.type == pygame.KEYDOWN :
                if event.key == pygame.K_UP:
                    print('listening')

                if event.key == pygame.K_DOWN:
                    
                    #running = False
                    desenhando = False
                    resposta = guess(data=pygame.surfarray.array3d(screen),currentDir=currentDir)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:    
                    
                    if rectangle.collidepoint(event.pos):
                        print(event.button)
                        desenhando = not desenhando
                        mouse_x, mouse_y = event.pos
                        offset_x = rectangle.x - mouse_x
                        offset_y = rectangle.y - mouse_y

            if event.type == pygame.MOUSEMOTION:                    
                mouse_x, mouse_y = event.pos
                rectangle.x = mouse_x -rectangle.width/2
                rectangle.y = mouse_y -rectangle.height/2

        # Fill the background with white
        if(desenhando==False):
            screen.fill((0, 0, 0))

  
        pygame.draw.rect(screen, (255,0,0), rectangle)
        # Flip the display
        if(desenhando==False):
            if(len(resposta)>0):
                resp = 0
                for i in range(len(resposta)):
                    if(resposta[i]==1):
                        resp = i
                text = font.render('Resposta: '+str(resp)+' ', True, (0,0,0), (255,255,255))
            else:
                text = font.render('Desenhe: ', True, (0,0,0), (255,255,255))
            screen.blit(text, textRect)
        pygame.display.flip()


    # Done! Time to quit.

    pygame.quit()

def resize(data,fator):
    mmm = np.zeros((len(data)//fator+1,len(data[0])//fator+1))
    i = 1
    j = 1
    k = 0
    l = 0

    while i in range(len(data)-1): 
        j = 0
        l =0 
        while j in range(len(data[0])-1):

            valor1 = data[i][j]

            if(i-1>=0 and j-1>=0):
                valor = data[i-1][j-1]
                if(valor>valor1):
                    valor1 = valor

            if(j-1>=0):
                valor = data[ i ][j-1]
                if(valor>valor1):
                    valor1 = valor

            if(i+1<=len(data) and  j-1>=0):
                valor = data[i+1][j-1]
                if(valor>valor1):
                    valor1 = valor

            if(i-1>=0):
                valor = data[i-1][j]
                if(valor>valor1):
                    valor1 = valor

            if(i+1<=len(data)):
                valor = data[i+1][j]
                if(valor>valor1):
                    valor1 = valor

            if(i-1>0 and j+1<len(data[0])):
                valor = data[i-1][j+1]
                if(valor>valor1):
                    valor1 = valor

            if(j+1<len(data[0])):
                valor = data[ i ][j+1]
                if(valor>valor1):
                    valor1 = valor

            if(i+1<len(data) and j+1<len(data[0])):
                valor = data[i+1][j+1]
                if(valor>valor1):
                    valor1 = valor
            if(k<=((len(data)//fator)-1) and l<=((len(data[0])//fator)-1 )):
                mmm[k][l] = valor1
            l = l +1
            j = j +fator
        k = k +1
        i = i +fator
    return mmm

def convol(datas):
    kernel = [[1,1,1],[1,-8,1],[1,1,1]]

    mmm = np.zeros((len(datas)//3+1,len(datas[0])//3+1))
    i = 1
    j = 1
    k = 0
    l = 0
    wid = len(datas)
    hei = len(datas[0])
    while i in range(wid):
        j = 1 
        l = 0
        while j in range(hei):

            valor = 0
            if(i-1>=0 and j-1>=0):
                #gray = 0.299red + 0.587green + 0.114blue
                valor = valor + kernel[0][0]*datas[i-1][j-1]
            if(j-1>=0):
                valor = valor + kernel[0][1]*datas[ i ][j-1]
            if(i+1<=len(datas)-1 and  j-1>=0):
                valor = valor + kernel[0][2]*datas[i+1][j-1]
            if(i-1>=0):
                valor = valor + kernel[1][0]*datas[i-1][j]

            valor = valor + kernel[1][1]*datas[ i ][j]
            
            if(i+1<=len(datas)-1):
                valor = valor + kernel[1][2]*datas[i+1][j]
            if(i-1>0 and j+1<len(datas[0])-1):
                valor = valor + kernel[2][0]*datas[i-1][j+1]
            if(j+1<len(datas[0])-1):
                valor = valor + kernel[2][1]*datas[ i ][j+1]
            if(i+1<len(datas[0])-1 and j+1<len(datas[0])-1):
                valor = valor + kernel[2][2]*datas[i+1][j+1]
            mmm[k][l] = valor
            l = l +1
            j = j +3
        k = k +1
        i = i +3
    print(mmm)
    return mmm

def toBP(datas):
    mmm = np.zeros((len(datas),len(datas[0])))
    i = 0
    j = 0
    while i in range(len(datas)): 
        j = 0
        while j in range(len(datas[0])):
            aux = datas[i][j]
            amop = (0.299*aux[0] + 0.587*aux[1] + 0.114*aux[2])
            mmm[i][j] = amop
            j = j +1
        i = i +1
    #print(mmm)
    return mmm

def plot (a):
    st = ''
    for i in range(len(a)):
        
        for j in range(len(a[0])):
            if(int(a[i][j])==0):
                st=st+' '
            else:
                st=st+'X'
            #st = st + str(a[i][j]) + ' '
        st = st +'\n'
    print(st)
    print(len(a),len(a[0]))
   
def guess(data,currentDir):
    dataAux = np.flip(data, (0))
    dataAux = np.rot90(dataAux, k=3, axes=(0,1))
    data = im.fromarray(dataAux)
    data.save('desenho.png')

    mm = toBP(dataAux)
    m = convol(mm)
    res = resize(mm,32)

    plot(res)
    res = res.astype('int')
    res_aux = res.flatten()

    res1 = np.append(res_aux,[0,0,0,0,0,0,0,0,0,0])
    res2 = np.zeros(len(res1))
    for i in range(len(res1)):
        if(res1[i]==0):
            res2[i] = 0
        else:
            res2[i] = 1

    os.chdir(currentDir+'\\ImgTeste')
    np.savetxt('teste.txt', res2, delimiter =', ')
    os.chdir(currentDir)
    return MLP_teste(path='teste',vsai=10,neur=250,limiar=0.0,currentDir=currentDir)

