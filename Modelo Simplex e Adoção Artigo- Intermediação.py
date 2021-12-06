
# coding: utf-8

# In[6]:


import random
import numpy as np
import networkx as nx
from scipy.optimize import linprog
from numpy.linalg import solve
import matplotlib.pyplot as plt
import math as mt

#Criando grafo
G=nx.Graph()
n = 100  # nodes
m = 1000  # edges

G = nx.gnm_random_graph(n, m)
#nx.draw(G, with_labels=True)
#plt.show()

#Inserindo equações



def preco_novo(preco, custo, gama, demanda, oferta):
    pn=preco*(1+gama*(demanda-oferta))
    return max(custo, pn)
    #return preco*(np.exp(gama*(demanda-oferta)))

def producao( k, renda, custo):
    return k*renda/custo

def prod_max(demanda_inicial, numero):
    return 2*demanda_inicial/numero

def lucro(preco, custo, procucao):
    return (preco-custo)*producao


culturas =4
gama= 0.0001
T=100  
precos=[65.0,32.0,41.0, 60.0]
custos=[7.0,8.0, 6.0, 10.0]
demandas=[4000,2000,6000, 9000]
coef_pr_i=0.3
l_bet,l_deg,l_clo,l_eig=[],[],[],[]
inovadores=[1,2,4,6,8,12,15,20,25,30]


for i in range (n):
    tb=(None, None)
    td=(None, None)
    tc=(None, None)
    te=(None, None)
    nx.betweenness_centrality(G)[i]
    nx.degree_centrality(G)[i]
    tb=(i, nx.betweenness_centrality(G)[i])
    td=(i, nx.degree_centrality(G)[i])
    tc=(i, nx.closeness_centrality(G)[i])
    te=(i, nx.eigenvector_centrality(G)[i])
    l_bet.append(tb)
    l_deg.append(td)
    l_clo.append(tc)
    l_eig.append(tc)
    
l_bet=sorted(l_bet, key=lambda tup: tup[1], reverse=True)
l_deg=sorted(l_deg, key=lambda tup: tup[1], reverse=True)
l_clo=sorted(l_clo, key=lambda tup: tup[1], reverse=True)
l_eig=sorted(l_eig, key=lambda tup: tup[1], reverse=True)
lb=(list([i[0] for i in l_bet]))
ld=(list([i[0] for i in l_deg]))
lc=(list([i[0] for i in l_clo]))
le=(list([i[0] for i in l_eig]))
precos_zero=[35.0,22.0,41.0, 50.0]


    

lad=[]

#Iniciando loop de inovação
for p in (inovadores):
    lb1=lb[0:p]
    print (p, lb1)
    
    
    #Iniciando listas
    lfalencias, ladot=[], []    
    lt,prod, lprecos=[],[],[]
    
    #Calculando ks    
    for i in range (n):    
        r=list(np.random.randint(1,100, size=culturas-1))
        r.append(0)
        r/=sum(r)
        G.node[i]['k_culturas']=r

    #Definindo renda inicial
    renda_inicial=list(np.random.randint(0,200, size=n))
    
    
    #Definindo producoes iniciais    
    for i in range (n):        
        G.node[i]['renda']=renda_inicial[i]
        G.node[i]['producoes']=[0]*culturas
        for j in range (culturas):
            G.node[i]['producoes'][j]=G.node[i]['k_culturas'][j]*G.node[i]['renda']/custos[j]
        G.node[i]['renda']=0
        G.node[i]['adotante']=0
        
    #Iniciando loop de tempo
    for t in range (T):    
        ofertas=[0]*culturas
        falencias=0
    
        #Iniciando Listas
        lrenda=[]
        ladotantes=[]
    
        #Loop de agentes
        for i in range (n):            
            G.node[i]['lucros']=[0]*culturas
            
            #Inserindo dados sobre adotantes
            if  G.node[i]['adotante']==1:
                ladotantes.append(i)
        
            #Calculando lucro
            for j in range (culturas):                
                ofertas[j]+=G.node[i]['producoes'][j]
                G.node[i]['lucros'][j]=(precos[j]-custos[j])*G.node[i]['producoes'][j]
            
            #Calculando renda nova     
            G.node[i]['renda_nova']=G.node[i]['renda']        
            for j in range (culturas):
                if ofertas[j]<=demandas[j]:                
                    G.node[i]['renda_nova']+=(G.node[i]['producoes'][j])*(precos[j])
                else:
                    exd=ofertas[j]-demandas[j]
                    G.node[i]['renda_nova']+=(G.node[i]['producoes'][j]-(exd/n))*(precos[j])
        
            #Definindo falências
            if G.node[i]['renda_nova']<=0:
                falencias+=1                 
                G.node[i]['renda_nova']=0
                G.node[i]['renda']=0
                G.node[i]['producoes']=[0]*culturas
        
        ladot.append(len(ladotantes))    
        #print(len(ladotantes),ladotantes, t)
    
        #Inserindo Valores na lista de produção
        prod.append(ofertas)
    
        #Recalculando preços
        precos_novos=[0]*culturas        
        for j in range (culturas):
            r= random.uniform(0, 1)
            f=0.667
            precos_novos[j]=0.05*mt.cos(mt.pi*2*(f+0.05*r)*t/40)+(2*custos[j])
    
        #Inserindo valores na lista de preços
        lprecos.append(precos_novos)

        # inserindo simplex 
        for i in range(n):
            c, A, b=[],[],[]
            e=[0]*(culturas)
            for j in range (culturas): 
                if ofertas[j]<demandas[j]:
                    e[j]=(demandas[j]-ofertas[j])/n            
                c.append(-1.0*(precos[j]-custos[j]))
                A.append(custos[j])
            b.append(G.node[i]['renda_nova'])
            
            if  G.node[i]['adotante']==0:
                x3_bnds = (0, 0)
            else:
                x3_bnds = (0, (demandas[3]/n)+2*e[3])
            
            x0_bnds = (0, (demandas[0]/n)+2*e[0])
            x1_bnds = (0,(demandas[1]/n)+2*e[1])
            x2_bnds = (0,(demandas[2]/n)+2*e[2])
        
        
            res = linprog(c, A, b, bounds=(x0_bnds, x1_bnds, x2_bnds, x3_bnds))
            #print(res.x,t)
            G.node[i]['producoes']=res.x
            
        #Inserindo inovação    
        if t==50:
            for k in range (len(lb1)):            
                inov=lb1[k]
                #print(inov)
                p_inov=0
                for j in range (culturas-1):
                    G.node[inov]['producoes'][j]*=(1-coef_pr_i)/custos[j]
                    pinov=G.node[inov]['producoes'][j]*custos[j]                
                G.node[inov]['producoes'][culturas-1]=pinov/(coef_pr_i*custos[culturas-1])            
                G.node[inov]['adotante']=1
     
        #Atualizando a variável de renda
        for i in range(n):
            for j in range (culturas):
                if G.node[i]['renda_nova'] !=0:
                    sub=0
                    sub=G.node[i]['renda_nova']-G.node[i]['producoes'][j]*custos[j]
                    G.node[i]['renda_nova']=sub       
            G.node[i]['renda']=G.node[i]['renda_nova']

        #print(falencias, t)
        lfalencias.append(falencias)
        #print(precos_novos)
        precos=precos_novos
        lt.append(t)
    
        # Inserindo dispersão de inovação: 
        for i in range (n):
            alea=np.random.random()
            for v in range(len(ladotantes)):
                inova=ladotantes[v]
                if  G.node[i]['adotante']==0 and G.has_edge(i,inova)==True and sum(G.node[i]['lucros'])<sum(G.node[inova]['lucros']) and alea>0.98:
                    #print (i, alea,'sim', 'tempo', t)
                    for j in range (culturas-1):
                        G.node[i]['producoes'][j]*=(1-coef_pr_i)/custos[j]
                        pinov=G.node[i]['producoes'][j]*custos[j]                
                    G.node[i]['producoes'][culturas-1]=pinov/(coef_pr_i*custos[culturas-1])
                    G.node[i]['adotante']=1
                

    lad.append(ladot)                
    prod_cult, demanda_cult, precos_cult=[],[],[]
    for j in range (culturas):    
        prod_cult.append(list([i[j] for i in prod]))
        precos_cult.append(list([i[j] for i in lprecos]))
        d=[demandas[j]]*T
        demanda_cult.append(d)   
        
     
    
    #Definindo as cores para cada grafico de cultura
    cores=['green', 'blue', 'red', 'purple']
    cores_demanda=['green', 'blue', 'red', 'purple']
    
    #Plotando gráfico

    #for j in range(culturas):
        #plt.plot(lt,prod_cult[j], color=cores[j],label='a{}'.format(j+1))
        #plt.plot(lt,demanda_cult[j],'k--', color=cores_demanda[j], label='demanda {}'.format(j+1))
    #plt.legend(loc='lower right')
    #plt.xlabel('Tempo') 
    #plt.ylabel('Produção') 
    #plt.title("Exemplo1")
    #plt.show()


    #plt.plot(lt,lfalencias,'k--', color='black', label='Falências')
    #plt.legend(loc='lower right')
    #plt.xlabel('Tempo') 
    #plt.ylabel('Falências') 
    #plt.title("Exemplo2")
    #plt.show()


    #plt.plot(lt,ladot, color='purple', label= 'Adotantes')
    #plt.legend(loc='lower right')
    #plt.xlabel('Tempo') 
    #plt.ylabel('Adotantes') 
    #plt.title("Exemplo3")
    #plt.show()


    #print(lprecos)
    #for j in range(culturas):
        #plt.plot(lt,precos_cult[j], color=cores[j], label='preco {}'.format(j+1))
    #plt.legend(loc='lower right')
    #plt.xlabel('Tempo')
    #plt.ylabel('Preços') 
    #plt.title("Exemplo4")
    #plt.show()
    
    
    
for j in range(len(lad)):
    if j==0:
        plt.plot(lt, lad[j], label='{} adotante'.format(inovadores[j]))
    else:
        plt.plot(lt, lad[j], label='{} adotantes'.format(inovadores[j]))
plt.legend(loc='lower left')
plt.xlabel('Tempo') 
plt.ylabel('Produção') 
plt.title("Intermediação")
plt.show();


# In[7]:


def preco_novo(preco, custo, gama, demanda, oferta):
    pn=preco*(1+gama*(demanda-oferta))
    return max(custo, pn)
    #return preco*(np.exp(gama*(demanda-oferta)))

def producao( k, renda, custo):
    return k*renda/custo

def prod_max(demanda_inicial, numero):
    return 2*demanda_inicial/numero

def lucro(preco, custo, procucao):
    return (preco-custo)*producao


culturas =4
gama= 0.0001
T=100  
precos=[65.0,32.0,41.0, 60.0]
custos=[7.0,8.0, 6.0, 10.0]
demandas=[4000,2000,6000, 9000]
coef_pr_i=0.3

inovadores=[1,2,4,6,8,12,15,20,25,30]
nos=list(range(0, n))


lad=[]

#Iniciando loop de inovação
for p in (inovadores):
    la=random.sample(nos,p)
    print (p,la)
    
    
    #Iniciando listas
    lfalencias, ladot=[], []    
    lt,prod, lprecos=[],[],[]
    
    #Calculando ks    
    for i in range (n):    
        r=list(np.random.randint(1,100, size=culturas-1))
        r.append(0)
        r/=sum(r)
        G.node[i]['k_culturas']=r

    #Definindo renda inicial
    renda_inicial=list(np.random.randint(0,200, size=n))
    
    
    #Definindo producoes iniciais    
    for i in range (n):        
        G.node[i]['renda']=renda_inicial[i]
        G.node[i]['producoes']=[0]*culturas
        for j in range (culturas):
            G.node[i]['producoes'][j]=G.node[i]['k_culturas'][j]*G.node[i]['renda']/custos[j]
        G.node[i]['renda']=0
        G.node[i]['adotante']=0
        
    #Iniciando loop de tempo
    for t in range (T):    
        ofertas=[0]*culturas
        falencias=0
    
        #Iniciando Listas
        lrenda=[]
        ladotantes=[]
    
        #Loop de agentes
        for i in range (n):            
            G.node[i]['lucros']=[0]*culturas
            
            #Inserindo dados sobre adotantes
            if  G.node[i]['adotante']==1:
                ladotantes.append(i)
        
            #Calculando lucro
            for j in range (culturas):                
                ofertas[j]+=G.node[i]['producoes'][j]
                G.node[i]['lucros'][j]=(precos[j]-custos[j])*G.node[i]['producoes'][j]
            
            #Calculando renda nova     
            G.node[i]['renda_nova']=G.node[i]['renda']        
            for j in range (culturas):
                if ofertas[j]<=demandas[j]:                
                    G.node[i]['renda_nova']+=(G.node[i]['producoes'][j])*(precos[j])
                else:
                    exd=ofertas[j]-demandas[j]
                    G.node[i]['renda_nova']+=(G.node[i]['producoes'][j]-(exd/n))*(precos[j])
        
            #Definindo falências
            if G.node[i]['renda_nova']<=0:
                falencias+=1                 
                G.node[i]['renda_nova']=0
                G.node[i]['renda']=0
                G.node[i]['producoes']=[0]*culturas
        
        ladot.append(len(ladotantes))    
        #print(len(ladotantes),ladotantes, t)
    
        #Inserindo Valores na lista de produção
        prod.append(ofertas)
    
        #Recalculando preços
        precos_novos=[0]*culturas        
        for j in range (culturas):
            r= random.uniform(0, 1)
            f=0.667
            precos_novos[j]=0.05*mt.cos(mt.pi*2*(f+0.05*r)*t/40)+(2*custos[j])
    
        #Inserindo valores na lista de preços
        lprecos.append(precos_novos)

        # inserindo simplex 
        for i in range(n):
            c, A, b=[],[],[]
            e=[0]*(culturas)
            for j in range (culturas): 
                if ofertas[j]<demandas[j]:
                    e[j]=(demandas[j]-ofertas[j])/n            
                c.append(-1.0*(precos[j]-custos[j]))
                A.append(custos[j])
            b.append(G.node[i]['renda_nova'])
            
            if  G.node[i]['adotante']==0:
                x3_bnds = (0, 0)
            else:
                x3_bnds = (0, (demandas[3]/n)+2*e[3])
            
            x0_bnds = (0, (demandas[0]/n)+2*e[0])
            x1_bnds = (0,(demandas[1]/n)+2*e[1])
            x2_bnds = (0,(demandas[2]/n)+2*e[2])
        
        
            res = linprog(c, A, b, bounds=(x0_bnds, x1_bnds, x2_bnds, x3_bnds))
            #print(res.x,t)
            G.node[i]['producoes']=res.x
            
        #Inserindo inovação    
        if t==50:
            for k in range (len(la)):            
                inov=la[k]
                #print(inov)
                p_inov=0
                for j in range (culturas-1):
                    G.node[inov]['producoes'][j]*=(1-coef_pr_i)/custos[j]
                    pinov=G.node[inov]['producoes'][j]*custos[j]                
                G.node[inov]['producoes'][culturas-1]=pinov/(coef_pr_i*custos[culturas-1])            
                G.node[inov]['adotante']=1
     
        #Atualizando a variável de renda
        for i in range(n):
            for j in range (culturas):
                if G.node[i]['renda_nova'] !=0:
                    sub=0
                    sub=G.node[i]['renda_nova']-G.node[i]['producoes'][j]*custos[j]
                    G.node[i]['renda_nova']=sub       
            G.node[i]['renda']=G.node[i]['renda_nova']

        #print(falencias, t)
        lfalencias.append(falencias)
        #print(precos_novos)
        precos=precos_novos
        lt.append(t)
    
        # Inserindo dispersão de inovação: 
        for i in range (n):
            alea=np.random.random()
            for v in range(len(ladotantes)):
                inova=ladotantes[v]
                if  G.node[i]['adotante']==0 and G.has_edge(i,inova)==True and sum(G.node[i]['lucros'])<sum(G.node[inova]['lucros']) and alea>0.98:
                    #print (i, alea,'sim', 'tempo', t)
                    for j in range (culturas-1):
                        G.node[i]['producoes'][j]*=(1-coef_pr_i)/custos[j]
                        pinov=G.node[i]['producoes'][j]*custos[j]                
                    G.node[i]['producoes'][culturas-1]=pinov/(coef_pr_i*custos[culturas-1])
                    G.node[i]['adotante']=1
                

    lad.append(ladot)                
    prod_cult, demanda_cult, precos_cult=[],[],[]
    for j in range (culturas):    
        prod_cult.append(list([i[j] for i in prod]))
        precos_cult.append(list([i[j] for i in lprecos]))
        d=[demandas[j]]*T
        demanda_cult.append(d)   
        
     
    
    #Definindo as cores para cada grafico de cultura
    cores=['green', 'blue', 'red', 'purple']
    cores_demanda=['green', 'blue', 'red', 'purple']
    
    #Plotando gráfico

    #for j in range(culturas):
        #plt.plot(lt,prod_cult[j], color=cores[j],label='a{}'.format(j+1))
        #plt.plot(lt,demanda_cult[j],'k--', color=cores_demanda[j], label='demanda {}'.format(j+1))
    #plt.legend(loc='lower right')
    #plt.xlabel('Tempo') 
    #plt.ylabel('Produção') 
    #plt.title("Exemplo1")
    #plt.show()


    #plt.plot(lt,lfalencias,'k--', color='black', label='Falências')
    #plt.legend(loc='lower right')
    #plt.xlabel('Tempo') 
    #plt.ylabel('Falências') 
    #plt.title("Exemplo2")
    #plt.show()


    #plt.plot(lt,ladot, color='purple', label= 'Adotantes')
    #plt.legend(loc='lower right')
    #plt.xlabel('Tempo') 
    #plt.ylabel('Adotantes') 
    #plt.title("Exemplo3")
    #plt.show()


    #print(lprecos)
    #for j in range(culturas):
        #plt.plot(lt,precos_cult[j], color=cores[j], label='preco {}'.format(j+1))
    #plt.legend(loc='lower right')
    #plt.xlabel('Tempo')
    #plt.ylabel('Preços') 
    #plt.title("Exemplo4")
    #plt.show()
    
    
    
for j in range(len(lad)):
    if j==0:
        plt.plot(lt, lad[j], label='{} adotante'.format(inovadores[j]))
    else:
        plt.plot(lt, lad[j], label='{} adotantes'.format(inovadores[j]))
plt.legend(loc='lower left')
plt.xlabel('Tempo') 
plt.ylabel('Produção') 
plt.title(" Escolha Aleatória")
plt.show();

