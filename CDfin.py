import scipy.stats as sp
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as pex
import plotly.graph_objects as go
import statsmodels.api as sm
import numpy as np

#Conectemos con Google Drive
#from google.colab import drive
#import pandas as pd

#Leamos el archivo
#drive.mount('/content/drive')
#datos= pd.read_csv("/content/drive/MyDrive/CDIII/archivos/salary.csv")

#@title Funciones importantes

def follow(binary):
            '''
            Recibe:
                - binary(list,tuple): coeficientes binarios de un número
            Devuelve:
                - binary + 1
            Sutilezas:
                - NO puede aumentar el tamaño de binary, por lo que (1,1)
                  no devuelve nada
            '''
            # A) binary lista
            if type(binary) == list:
                copia = binary.copy()
                if copia[-1] == 0:
                    copia[-1] = 1
                    return copia
                else:
                    return follow(copia[:-1]) + [0]
            # B) binary tupla
            if type(binary) == tuple:
                copia = binary  # es lo mismo pues es tupla
                output = ()
                if copia == ():
                    return ()
                if copia[-1] == 0:  # si el último elemento es 0
                    return copia[:-1] + (1,)  # devolvemos la tupla con un 1 en ese lugar
                else:
                    return follow(copia[:-1]) + (0,)  # si el último elemento es 1, volvemos a iterar con la tupla y un 0 en el lugar del 1

#@title DistCont
class DistCont(sp.rv_continuous):
    def __init__(self, momtype=1, a=None, b=None, xtol=1e-14, badvalue=None, name=None, longname=None, shapes=None, seed=None):
        super().__init__(momtype=momtype, a=a, b=b, xtol=xtol, badvalue=badvalue, name=name, longname=longname, shapes=shapes, seed=seed)
        self.__stats_diccionario = self.stats
        pass
    #=========================================#
    #         Métodos aritméticos             #
    #=========================================#
    def __sum__(self,other):
        pass
    def __mul__(self,other):
        pass
    def __sub__(self,other):
        pass
    def __truediv__(self,other):
        pass
    def __pow__(self,other):
        pass
    #=========================================#
    #       Métodos esenciales                #
    #=========================================#

    #=========================================#
    #       Métodos de alteración             #
    #=========================================#
    def set_pdf(self,funcion):
        import types
        assert type(funcion) == types.FunctionType, TypeError('debes ingresar un objeto de tipo function')
        self._pdf = funcion

    def set_cdf(self,funcion):
        import types
        assert type(funcion) == types.FunctionType, TypeError('debes ingresar un objeto de tipo function')
        self._cdf = funcion
    #=========================================#
    #       Métodos de usuario                #
    #=========================================#
    def generar_datos(self,size):
        assert type(size) == int, TypeError('debes ingresar un tamaño de tipo int')
        return self.rvs(size)

    def percentil(self,prob,rightTail=False):
        '''
        ===================================
        Devuelve el valor cuya probabilidad
        de que algún otro valor sea menor
        o igual a él es prob
        ===================================
        '''
        assert type(prob) == float, TypeError('debes ingresar una probabilidad de tipo float')
        assert type(rightTail) == bool, TypeError('debes ingresar un argumento rightTail de tipo bool')

        return self.ppf(prob) if not rightTail else self.ppf(1-prob)

    def unif_eval(self,start,end,size,valores=False):
        '''
        ===================================
        Evalúa la función de densidad en
        un conjunto uniformemente distribuido
        de puntos
        ===================================
        Recibe:
            - start(float):  inicio de los datos
            - end(float): fin de los datos
            - size(int): cantidad de datos en [start, end]
            - valores(bool):  si True, la función devuelve también el vector de los argumentos
                              (valores, evaluaciones)
        '''
        assert start < end, 'el inicio debe ser menor que el final'
        assert type(size) == int, 'debes ingresar un tamaño de tipo int'
        assert type(valores) == bool, 'debes ingresar un argumento valores de tipo bool'

        import numpy as np
        x_val = np.linspace(start,end,size)
        if valores:
            return x_val,self.pdf(x_val)
        return self.pdf(x_val).values()

    def test(self,x,condition='equal'):
        '''
        ======================================
        Realiza un test de hipótesis para el
        valor x con una cierta condición de
        igualdad o de desiguladad
        ======================================
        Recibe:
            - x:  valor que se desea testear
            - condition:
                - equal
                - greater
                - less
        Devuelve:
            - el pvalor de x
        '''
        assert condition == 'equal' or condition == 'greater' or condition == 'less', f'{condition} no es una condición válida'
        # A)  Condición de igualdad
        if condition == 'equal':
            # 1)  vemos en qué cola se encuentra el valor
            if self.cdf(x) <= 0.5:
                pval = 2*self.cdf(x)
            else:
                pval = 2*(1-self.cdf(x))
        # B)  Condición de mayor
        if condition == 'greater':
            pval = 1-self.cdf(x)
        # C)  Condición de menor
        if condition == 'less':
            pval = self.cdf(x)
        return float(pval)

    def conf(self,x,confidence):  ### IMPLEMENTAR
        '''
        =========================================
        Genera un intervalo de confianza para
        un valor medido x
        =========================================
        '''
    #=======================================#
    #     Métodos gráficos                  #
    #=======================================#
    def graficar(self,a,b,show=False,add=False):
        '''
        ====================================================
        Grafica la función de densidad en un intervalo (a,b)
        ====================================================
        Recibe:
            - a,b (float): inicio y final del intervalo respectivamente
            - show(bool): si True, muestra el plot
            - add(bool):  si se van a plotear varias cosas una encima de la otra, una línea abajo de la otra
        Devuelve:
            - devuelve un objeto tipo lines.Line2D de la librería matplotlib
            - Genera un plot hecho con pyplot
        Recomendaciones:
            - add
        '''
        assert a < b, 'el inicio debe ser menor que el final'
        assert type(show) == bool, 'debes ingresar un argumento show de tipo bool'

        import matplotlib.pyplot as plt
        if not add:
          plt.ioff()
        x,y = self.unif_eval(a,b,1000,valores=True)
        line, = plt.plot(x,y)
        if show:
            plt.show()
        return line

#@title Kernels

def kernel_gaussiano(x):
    import numpy as np
    # Kernel gaussiano estándar
    x = np.asarray(x, dtype=np.float64)
    return (1/np.sqrt(2*np.pi))*np.exp(-0.5*x**2)

def kernel_uniforme(x):
    # Kernel uniforme
    x = np.asarray(x, dtype=np.float64)
    if -1/2<x<1/2:
      return 1
    else:
      return 0
#@title Otros Kernels

def epanechnikov(x):
    x = np.asarray(x, dtype=np.float64)
    def I(x):
        if -1<=x<=1:
            return 1
        return 0
    return (3/4)*(1-x**2)*I(x)

def triangular(x):
    x = np.asarray(x, dtype=np.float64)
    def I1(x):
        if -1<=x<=0:
            return 1
        return 0

    def I2(x):
        if 0<=x<=1:
            return 1
        return 0

    return (1+x)*I1(x) + (1-x)*I2(x)

#@title subclases de Densidad

class Normal(DistCont,sp.rv_continuous):
    def __init__(self,mean,sd,momtype=1, a=None, b=None, xtol=1e-14, badvalue=None, name=None, longname=None, shapes=None, seed=None):
        super().__init__(momtype=momtype, a=a, b=b, xtol=xtol, badvalue=badvalue, name=name, longname=longname, shapes=shapes, seed=seed)
        assert type(sd) == float or type(sd) == int, 'debes ingresar un número como desviación estándar'
        assert type(mean) == float or type(mean) == int, 'debes ingresar un número como desviación estándar'

        self.set_pdf(lambda x : (1/(sd*np.sqrt(2*np.pi))) * np.exp(-(x-mean)**2/(2*sd**2)))


class BS(DistCont):
    def __init__(self,mean,sd,momtype=1, a=None, b=None, xtol=1e-14, badvalue=None, name=None, longname=None, shapes=None, seed=None):
        import numpy as np
        super().__init__(momtype=momtype, a=a, b=b, xtol=xtol, badvalue=badvalue, name=name, longname=longname, shapes=shapes, seed=seed)
        self.__m = mean
        self.__s = sd
        self.set_pdf(lambda x : np.vectorize(self.__f)(x))

    def __f(self,x):
        import scipy as sp
        normal_0_1 = sp.stats.norm(self.__m,self.__s)
        sumando_1 = (1/2)*normal_0_1.pdf(x)
        sumando_2 = 0
        for j in range(5):
            normal_for = sp.stats.norm(j/2-1,1/10)
            sumando_2 += normal_for.pdf(x)
        sumando_2 *= 1/10
        return sumando_1 + sumando_2

class Uniforme(DistCont):
    def __init__(self,start,end,momtype=1, a=None, b=None, xtol=1e-14, badvalue=None, name=None, longname=None, shapes=None, seed=None):
        import numpy as np
        super().__init__(momtype=momtype, a=a, b=b, xtol=xtol, badvalue=badvalue, name=name, longname=longname, shapes=shapes, seed=seed)
        self.__a = start
        self.__b = end
        self.set_pdf(lambda x : np.vectorize(self.__f)(x))

    def __f(self,x):
        if self.__a<=x<=self.__b:
            return 1/(self.__b-self.__a)
        else:
            return 0

class tStudent(DistCont):
    def __init__(self,libertades,momtype=1, a=None, b=None, xtol=1e-14, badvalue=None, name=None, longname=None, shapes=None, seed=None):
        import numpy as np
        super().__init__(momtype=momtype, a=a, b=b, xtol=xtol, badvalue=badvalue, name=name, longname=longname, shapes=shapes, seed=seed)
        self.__lib = libertades
        self.set_pdf(lambda x : np.vectorize(self.__f)(x))

    def __f(self,x):
        import math
        v = self.__lib
        G = math.gamma
        return ((G((v+1)/(2)))/((v*math.pi)**(1/2) * G(v/2))) * (1+((x**2)/(v)))**((v+1)/(-2))

class exponencial(DistCont):
    def __init__(self,lamb,momtype=1, a=None, b=None, xtol=1e-14, badvalue=None, name=None, longname=None, shapes=None, seed=None):
        import numpy as np
        super().__init__(momtype=momtype, a=a, b=b, xtol=xtol, badvalue=badvalue, name=name, longname=longname, shapes=shapes, seed=seed)
        self.__lamb = lamb
        self.set_pdf(lambda x : np.vectorize(self.__f)(x))

    def __f(self,x):
        import math
        L = self.__lamb
        return L*math.exp(-L*x)

#@title Datos
class Datos(np.ndarray):
    def __init__(self,shape,dtype=object,buffer=None,offset=0,strides=None,order=None):
        '''
        ==================================================
        array de datos con algunas cosas extra
        ==================================================
        Recomendaciones:
            - declarar dtype=object
        '''
        super().__init__()
    #=======================================#
    #         Métodos entre datos           #
    #=======================================#
    def __mod__(self,otro):
        '''
        =======================================
        Calcula la covarianza entre dos arrays
        de datos (X%Y)
        ======================================
        '''
        assert np.ndarray in self.__class__.mro() and np.ndarray in self.__class__.mro(), TypeError('Los objetos no son ambos de tipo ndarray')
        for element in self:
            assert type(element) != str and type(element) != np.str_, TypeError('No se permiten strings')
        for element in otro:
            assert type(element) != str and type(element) != np.str_, TypeError('No se permiten strings')
        assert len(self) == len(otro), IndexError('Los array son de distinta longitud')

        X = self
        Y = otro
        n = len(X)
        m_X = X.mean()
        m_Y = Y.mean()
        cov_sample = 0

        for i in range(n):
            cov_sample += (X[i]-m_X)*(Y[i]-m_Y)
        cov_sample *= 1/(n-1)
        return cov_sample

    #=======================================#
    #         Métodos de usuario            #
    #=======================================#
    def sample(self,size,replace=False):
        copia = self
        import random as rand
        tipo = type(self)
        output = tipo((size,1),dtype=object)

        rng = np.random.default_rng()
        muestra = rng.choice(copia,size=size,replace=replace)
        muestra = muestra.reshape(len(muestra),)
        output[:,0] = muestra

        return output

    def resumen_numerico(self):
        import numpy as np
        data = np.array(self[:, 0], dtype=np.float64)
        res_num = {
        'Media': data.mean(),
        'Mediana': np.median(data),
        'Desvio': data.std(),
        'Cuartiles': (float(np.percentile(data,25)), float(np.percentile(data,50)), float(np.percentile(data,75))),
        'Mínimo': min(data),
        'Máximo': max(data)
        }
        for estad, valor in res_num.items():
            print(f"{estad}: {np.round(valor,3)}")
        return res_num

    def arr(self):
        output = np.array(self[:,0],dtype=object)
        return output

#@title Cualitative and Quantitative
#@title Cualitative and Quantitative
class CualDatos(Datos):
    '''
    Datos cualitativos
    '''
    def __init__(self,shape,dtype=object,buffer=None,offset=0,strides=None,order=None):
        super().__init__(shape=shape,dtype=dtype,buffer=buffer,offset=offset,strides=strides,order=order)
        #=================#
        # creados cuando se codifica
        self.__code = None
        self.__clase = None
        #=================
    #==================================#
    #     Métodos de construcción      #
    #==================================#
    def __numerizar(self,**codigos):
        '''
        =================================
        devuelve un vector numérico
        con las clases transformadas
        en números
        =================================
        Recibe:
            - codigos:  valores que tomará cada clase
                        clase = valor
        '''
        hashable = [] # permite que se pueda usar set
        for el in self[:,0]:
            hashable.append(el)
        clases = list(set(hashable))
        clases_num = codigos

        if clases_num == {}:
            for i in range(len(clases)):
                clases_num.update({clases[i]:i})
        else:
            for clase in codigos:
                assert type(codigos[clase]) != str and type(codigos[clase]) != np.str_, 'No se puede codificar con valores string'
                assert clase in clases, f'La clase {clase} no existe'
                assert len(set(codigos.keys())) == len(set(codigos.values())), 'Hay códigos repetidos'
            for clase in clases:
                assert clase in codigos, f'No se proveyó código para la clase {clase}'

        datos_num = [clases_num[dato] for dato in self[:,0]]
        Clases_Coded = CualDatos((len(self),1),dtype=object)
        Clases_Coded[:,0] = datos_num

        Clases_Coded.set_code({clase:clases_num[clase] for clase in clases})  # clase -> codigo
        Clases_Coded.set_clase({clases_num[clase]:clase for clase in clases})   # codigo -> clase
        return Clases_Coded

    def __binarizar(self,**codigos):
        '''
        ======================================
        devuelve un vector de tuplas, donde
        cada tupla es un código binario que
        identifica unívocamente a cada clase
        ======================================
        '''
        hashable = [] # permite que se pueda usar set
        for el in self[:,0]:
            hashable.append(el)
        clases = list(set(hashable))
        clases_num = codigos

        # A)  Si no se proveen códigos, se los generará
        if clases_num == {}:
            import math as m
            cant_coef = m.ceil(m.log(len(clases),2))  # necesitamos ceil(log_2(clases)) dígitos
            code = (0,)*cant_coef
            for i in range(len(clases)):
                clases_num.update({clases[i]:code})
                code = follow(code)
        else:
            for clase in codigos:
                assert type(codigos[clase]) != str and type(codigos[clase]) != np.str_ and type(codigos[clase]) != float and type(codigos[clase]) != np.float64 and type(codigos[clase]) != int and type(codigos[clase]) != np.int_, 'Se debe codificar con tuplas'
                assert clase in clases, f'La clase {clase} no existe'
                #assert len(set(codigos.keys())) == len(set(codigos.values())), 'Hay códigos repetidos' ####### GUARDA REPETITION
                assert len(set(codigos[clase])) <= 2 and (0 in set(codigos[clase]) or 1 in set(codigos[clase])), f'La clase {clase} no tiene un código binario'
            for clase in clases:
                assert clase in codigos, f'No se proveyó código para la clase {clase}'

        datos_num = [clases_num[dato] for dato in self[:,0]]
        Clases_Coded = CualDatos((len(self),1),dtype=object)
        Clases_Coded[:,0] = datos_num

        Clases_Coded.set_code({clase:clases_num[clase] for clase in clases})  # clase -> codigo
        Clases_Coded.set_clase({clases_num[clase]:clase for clase in clases})   # codigo -> clase
        return Clases_Coded

    def __bin_df(self,nombre,**codigos):
        '''
        =======================================
        Transforma el vector binarizado en un
        DataFrame
        =======================================
        Recibe:
            - nombre(str): Nombre que recibirá cada columna enumerada
            - codigos: códigos de las clases
        '''
        binary = self.biny(**codigos)
        output_df = pd.DataFrame()
        col_num = len(binary[0,0])
        print(f'{codigos=}')      ###
        if col_num == 1:  
            new_col = np.empty((0,1))
            for j in range(len(binary)):
                new_col = np.vstack([new_col,binary[j,0][0]])
            new_col = new_col.reshape(-1)
            output_df.insert(0,nombre,new_col)  # si se agrega sólo una columna, entonces no se le pone números al nombre
            return output_df
        for i in range(col_num):
            # Necesitamos crear la nueva columna
            new_col = np.empty((0,1))
            for j in range(len(binary)):
                new_col = np.vstack([new_col,binary[j,0][i]])
            new_col = new_col.reshape(-1)
            output_df.insert(i,nombre+str(i),new_col)
        return output_df
    #=========================================#
    #   Métodos de alteración                 #
    #=========================================#
    def set_code(self,codigos):
        self.__code = codigos
    def set_clase(self,clase):
        self.__clase = clase
    #==========================================#
    #   Métodos de usuario                     #
    #==========================================#
    def num(self,**codigos):
        '''
        ======================================
        Devuelve un array de datos codificado
        ======================================
        '''
        return self.__numerizar(**codigos)

    def biny(self,**codigos):
        '''
        ======================================
        Devuelve un array de datos codificado
        de forma binaria
        ======================================
        '''
        return self.__binarizar(**codigos)

    def biny_df(self,nombre,**codigos):
        '''
        ======================================
        Devuelve un pd.DataFrame de datos
        codificados de forma binaria
        ======================================
        '''
        return self.__bin_df(nombre,**codigos)

    def clases(self):
        '''
        ================================
        devuelve una lista de las clases
        de la categoría
        ================================
        '''
        datos = pd.Series(self[:,0])
        clases = set(datos)
        return list(clases)

class QuantDatos(Datos):
    '''
    Datos cuantitativos
    '''
    def __init__(self,shape,dtype=object,buffer=None,offset=0,strides=None,order=None):
        super().__init__(shape=shape,dtype=dtype,buffer=buffer,offset=offset,strides=strides,order=order)

    def __genera_histograma(self,h):
      '''
      aproxima la distribución de los datos a través de un
      histograma de intervalos regulares 'h'.

      declara el atributo self.densidades, que contendrá un diccionario
      con los cortes del histograma y las ALTURAS de dichos cortes (irónicamente, no las densidades).
      '''
      import numpy as np
      datos = self.arr()
      intervalos = np.arange(min(datos),max(datos)+1,h)
      densidades = dict()
      len_datos = len(datos)
      for corte in intervalos:
          corteDensidad = 0
          for dato in datos:
              if corte <= dato < corte + h:
                  corteDensidad += 1
          densidades.update({corte:corteDensidad/(len_datos*h)})
      self.__densidades = densidades
      print(self.__densidades)    ###
      return densidades

    def evalua_histograma(self,h,x):
        self.__genera_histograma(h)
        densidades = self.__densidades
        x = x.arr()
        if type(x) == int or type(x) == float:
            for corte in densidades:
                if corte <= x < corte + h:
                    return densidades[corte]
        evaluaciones = dict()
        print(f'x: {type(x)}')
        for valor in x:
            for corte in densidades:
                if corte <= valor < corte + h:
                    evaluaciones.update({valor:densidades[corte]})
        return evaluaciones

    def evalua_kernel(self,x,h,kernel):
        kernelDiccionarios = {  'gaussiano':kernel_gaussiano,\
                                  'uniforme':kernel_uniforme,\
                                  'epa':epanechnikov,\
                                  'triangular':triangular}
        len_data = len(self)
        kernelElegido = kernelDiccionarios[kernel]
        density = dict()
        datos = self.arr()
        x = x.arr()
        for core in x:
            coreValor = 0
            for valor in datos:
                coreValor += kernelElegido((valor-core)/h)
            density.update({core:coreValor/(len_data*h)})
        return density

    def graficar_histograma(self,h,show=False,add=False):
        '''
        ===================================
        Grafica histograma de los datos
        ===================================
        Recibe:
            - show(bool): mostrar el ploteo
            - add(bool):  si se van a plotear varias cosas una atrás de la otra
        '''
        import numpy as np
        import matplotlib.pyplot as plt
        try:
          densidades = self.__densidades
        except:
          self.__genera_histograma(h)
          densidades = self.__densidades

        a = min(self)
        b = max(self)
        x = np.linspace(a,b,1000)
        y = self.evalua_histograma(h,x).values()
        if not add:
            plt.ioff()
        line, = plt.plot(x,y)
        if show:
            plt.show()
        return line

    def graficar_kernel(self,h,kernel,show=False,add=False):
        '''
        Grafica una estimación de la densidad de los datos
        utilizano un kernel

        Recibe:
            h(float): longitud de los intervalos
        '''
        import numpy as np
        import matplotlib.pyplot as plt
        a = min(self)
        b = max(self)
        x = np.linspace(a,b,1000)
        densidad = self.evalua_kernel(x,h,kernel)
        if not add:
            plt.ioff()
        line, = plt.plot(x,densidad.values())
        if show:
            plt.show()
        return line

    def qqPlot(self,distribucion,*param,cuantiles=None,show=False,add=False):
        '''
        #########################################################
        Plotea cuantil vs. cuantil con una distribución
        arbitraria
        #########################################################
        Recibe:
          - distribucion(str):  la densidad con la que se desea
                                comparar los cuantiles
          - cuantiles(list):  una lista de los cuantiles que se
                              desean comparar
          - param(list):  una lista de parámetros necesarios para
                          determinar corréctamente la función de
                          densidad
        param:
          - Normal:
              - param[0] = media
              - param[1] = desvío estándar
          - t-student:
              - param[0] = grados de libertad
          - Uniforme:
              - param[0] = inicio del intervalo
              - param[1] = fin del intervalo
          - exponencial:
              - param[0] = lambda
        '''
        import scipy.stats as sp
        import numpy as np
        distribuciones = {'Normal':Normal,
                          't-student':tStudent,
                          'Uniforme':Uniforme,
                          'Exponencial':exponencial}
        if cuantiles == None:
            n = len(self)
            cuantiles = list((i/(n+1) for i in range(n)))

        dist = distribuciones[distribucion](*param)
        x_ord = self.arr() # copiamos los datos para no sobreescribirlos
        x_ord.sort()  # ordenamos la copia
        cuantiles_teoricos = []

        for cuantil in cuantiles: # obtenemos los cuantiles teóricos
            cuantiles_teoricos.append(dist.percentil(cuantil))
        mu = x_ord.mean()
        sd = x_ord.std()
        x_ord_s = []

        for x in x_ord: # estandarizamos los datos
            x_ord_s.append((x-mu)/sd)
        cuantiles_muestrales = []
        x_ord_s = np.array(x_ord_s,dtype=object)

        for cuantil in cuantiles:
            cuantiles_muestrales.append(np.percentile(x_ord_s,cuantil*100))

        import matplotlib.pyplot as plt
        if not add:
            plt.ioff()
        scatter = plt.plot(cuantiles_muestrales,cuantiles_teoricos,'o',label='muestral')
        print(f'{min(cuantiles_teoricos[1:])=}')    ###
        print(f'{max(cuantiles_teoricos)=}')        ###
        identity, = plt.plot([min(cuantiles_teoricos[1:]),max(cuantiles_teoricos)],[min(cuantiles_teoricos[1:]),max(cuantiles_teoricos)],'-',color='orange',label='teórico')
        if show:
            plt.xlabel('Cuantiles teóricos')
            plt.ylabel('Cuantiles muestrales')
            plt.legend()
        return scatter, identity
    
#@title Dataframe

class Dataframe(pd.DataFrame):
    def __init__(self,data=None, index=None, columns=None, dtype=None, copy=False):
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
    #===================================#
    #     Métodos de usuario            #
    #===================================#
    def num(self,**codigos):
        '''
        =============================================
        Devuelve un pd.DataFrame con todas las clases
        codificadas en números
        =============================================
        Recibe:
            - codigos:  los códigos de cada columna
                        columna = {clase1:code1, clase2:code2,...}
        '''
        output = pd.DataFrame(self)
        for columna in codigos:
            cualitative = CualDatos((self.shape[0],1),dtype=object)
            cualitative[:,0] = self[columna]
            output[str(columna)] = cualitative.num(**codigos[columna]).arr()

        return output

    def biny(self,**codigos):
        '''
        =============================================
        Devuelve un pd.DataFrame con todas las clases
        especificadas codificadas de forma binaria
        =============================================
        Recibe:
            - codigos:  columna = {clase1:code1,clase2:code2}
                        o
                        columna = None

        '''
        output = pd.DataFrame(self)
        for columna in codigos:
            # chequeamos que están los códigos de la columna
            #print(f'{codigos[columna]=}') ###
            if codigos[columna] == None:
                clases = list(set(output[columna]))
                col_code = {}
                import math as m
                cant_coef = m.ceil(m.log(len(clases),2))  # necesitamos ceil(log_2(clases)) dígitos
                code = (0,)*cant_coef
                for i in range(len(clases)):
                    col_code.update({clases[i]:code})
                    code = follow(code)

            cualitative = CualDatos((self.shape[0],1),dtype=object)
            cualitative[:,0] = output[columna]
            cual_biny_df = cualitative.biny_df(nombre=columna,**codigos[columna])  # dataframe binarizado tipo DataFrame
            #print(f'{cual_biny_df=}')   ###
            # necesitamos desdoblar esta codificación en varias columnas
            col_num = cual_biny_df.shape[1] # cantidad de columnas
            col_index = output.columns.get_loc(columna) # índice de la columna, donde insertaremos el dataframe binarizado
            output = output.drop(columna,axis=1) # eliminamos la columna que vamos a reemplazar
            for i in range(col_num):
                col_name = cual_biny_df.columns[i]
                output.insert(loc=col_index + i,column=col_name,value=cual_biny_df[col_name])
        return output

    #################################################
    #             Métodos de usuario                #
    #################################################

    def sample(self,size,axis=0,replace=False):
        '''
        Devuelve un objeto Dataframe sampleado
        en el eje axis
        '''
        if size<=1 and size>= 0:
            sampled = self.sample(frec=size,axis=axis,replace=replace)
        else:
            sampled = self.sample(n=size,axis=axis,replace=replace)
        output = Dataframe(sampled,columns=sampled.columns)
        return output

    def grafico2D(self,X,Y,kind='scatter',show=False,add=False):
        '''
        Recibe:
            - X: columna del eje X
            - Y: columna del eje Y
            - col: columna de coloreo
            - kind: tipo de ploteo
        '''
        import matplotlib.pyplot as plt
        if not add:
            plt.ioff()
        # Scatter:
        if kind == 'scatter':
            scatter = self.plot(kind=kind,x=X,y=Y)
            if show:
                plt.show()
            return scatter
        elif kind == 'line':
            line, = self.plot(kind=kind,x=X,y=Y)
            if show:
                plt.show()
            return line
        elif kind == 'boxplot':
            self.boxplot(by=X,column=Y)
            if show:
              plt.show()
        else:
            self.plot(kind=kind,x=X,y=Y)
            if show:
              plt.show()

    def prob_joint(self,X,Y,freq=False):
        '''
        ================================================
        Muestra las probabilidades conjuntas de cada una
        de las clases de las categorías X e Y.
        ================================================
        Ej:   X:  {male, female}
              Y:  {smoker,nonsmoker}
        P(male y smoker), P(female y nonsmoker), etc.
                  male  | female
        smoker    0.25  | 0.10
        nonsmoker 0.60  | 0.05
        ------------------------------------------------
        Asume:
            - X e Y son categóricas
        '''
        import pandas as pd
        joint_freq = pd.crosstab(self[X], self[Y])
        clases_X = list(set(self[X]))
        joint_freq = joint_freq.reindex(index=[0,1],columns=[0,1],fill_value=0)
        if freq:
            print('----------------------------------')
            print(f'    {X} and {Y}    ')
            print(joint_freq)
            return joint_freq
        joint_prob = joint_freq / joint_freq.to_numpy().sum()
        print('----------------------------------')
        print(f'    P({X} and {Y})    ')
        print(joint_prob)
        return joint_prob

    def prob(self,category,freq=False):
        '''
        ===============================================
        Muestra las probabilidades marginales de una
        categoría
        ===============================================
        Ej: category:  {male, female}
        -----------------------------------------------
        Asume:
            - X e Y son categóricas
        '''
        import pandas as pd
        joint_freq = pd.crosstab(self[category],columns='count')
        clases = list(set(self[category]))
        joint_freq = joint_freq.reindex(index=clases,fill_value=0)
        if freq:
            print('--------------------------------')
            print(f'    {category}   ')
            print(joint_freq)
            return joint_freq

        total = joint_freq.to_numpy().sum()
        joint_prob = joint_freq / total

        print('--------------------------------')
        print(f'    P({category})   ')
        print(joint_prob)
        return joint_prob

    def prob_cond(self,X,Y):
        '''
        =============================================
        Realiza la probabilidad condicional de X
        dado Y
        =============================================
        Ej: X:  {smoker, nonsmoker}
            Y:  {male, female}
        P(smoker|male), P(smoker|female), etc.
        '''
        import pandas as pd
        joint_freq = pd.crosstab(self[Y], self[X])
        clases_Y = list(set(self[Y]))
        joint_freq = joint_freq.reindex(index=[0,1],columns=[0,1],fill_value=0)
        joint_prob = joint_freq / joint_freq.to_numpy().sum()
        cond_prob = joint_freq.div(joint_freq.sum(axis=1), axis=0)
        print('-------------------------------')
        print(f'    P({X}|{Y}):   ')
        print(cond_prob)
        return cond_prob

#@title Modelo
class Modelo:
    '''
    ##############################################
    Modela un conjunto de datos de la forma
    predictor-respuesta
    ##############################################
        - predictor, respuesta: pueden ser cualquier
                                tipo de Datos; las subclases
                                manejarán cada uno
                                como sea adecuado
    '''
    def __init__(self,df,predictor,respuesta):
        '''

        Recibe:
            - df(Dataframe):  dataframe de los datos en su totalidad (predictores y respuestas)
            - predictores:  etiquetas de los datos predictores en df
            - respuestas: etiquetas de los datos respuesta en df
        '''
        self.__df = df
        #print(f'Model df: { type(df)}') ###
        #print(f'Model self.__df: {type(self.__df)}')  ###

        self.predictores = predictor  # labels predictores
        self.respuestas = respuesta   # labels respuesta

        self.predictores_df = Dataframe(self.__df[self.predictores])
        self.respuestas_df = Dataframe(self.__df[self.respuestas])

        self.funcion = None # la función de predicción
        self.__parametros = {}  # {'param':{'VAL','DIST','MEAN','SE'}}
        self.longitud = self.__df.shape[0]
    #============================================#
    #     Métodos de alteración                  #
    #============================================#
    def set_param(self,param,charact,value):
        try:
            self.__parametros[param][charact]
        except:
            try:
                self.__parametros[param]
            except:
                self.__parametros.update({param:{charact:value}})
            else:
                self.__parametros[param].update({charact:value})
        else:
            self.__parametros[param][charact] = value
    #============================================#
    #     Métodos de usuario                     #
    #============================================#
    def predict(self,dato,multiple=False):
        '''
        ##########################################
        Predice un valor de respuesta en base al
        dato proveído
        ##########################################
        '''
        return self.funcion(dato,multiple) # se especifica en el modelo específico

    def param(self,param,charact):
        output = self.__parametros[param][charact]
        return output

    def param_list(self):
        output = self.__parametros.keys()
        return output

    def resumen(self):
        print(self.resultado.summary())
        return self.resultado.summary()
    #===========================================#
    #       Relación con el dataframe           #
    #===========================================#
    def matriz(self,cond=None,datos=None):
        '''
        ====================================
        Devuelve una submatriz del dataframe
        ====================================
        Recibe:
            - datos:  lista de nombres de columnas
            - cond(str):  condiciones en formato estándar de DataFrame
        '''
        if datos != None:
            if cond != None:
                return self.__df[datos].query(cond)
            else:
                return self.__df[datos]
        else:
            if cond != None:
                return self.__df.query(cond)
            else:
                return self.__df

    def df(self):
        output = self.__df
        return output

    #====================================================#
    #       Métodos gráficos                             #
    #====================================================#
    def grafico2D(self,X,Y,kind='scatter',show=False,add=False):
        #print(f'Se ejecuta grafico2D dentro del modelo')
        #print(f'{X=}')
        #print(f'{Y=}')
        self.__df.grafico2D(X=X,Y=Y,kind=kind,show=show,add=add)

    def grafico_modelo(self,show=False,add=False):
        '''
        '''
        import numpy as np
        codigos_pred = {}
        codigos_res = {}  # (?)

        #######################################
        # excepciones
        try:
            self.codigos_pred
        except:
            pass
        else:
            codigos_pred = self.codigos_pred

        try:
            self.codigos_res
        except:
            pass
        else:
            codigos_res = self.codigos_res
        ########################################

        pred = self.biny_pred(**codigos_pred)
        #print(pred) ###
        #print(pred.iloc[0,0]) ###
        #print(type(pred.iloc[0,0])) ###
        assert len(pred.columns) <= 2, f'Demasiadas variables predictoras: {len(pred.columns)}'
        resp = self.respuestas_df

        # Caso 2D:
        if len(pred.columns) == 1:
            import matplotlib.pyplot as plt
            if not add:
                plt.ioff()
            scatter = self.grafico2D(X=pred.columns[0],Y=resp.columns[0],show=False,add=True)
            if type(pred.iloc[0,0]) != str and type(pred.iloc[0,0]) != np.str_:
                a = min(pred.iloc[:,0])
                b = max(pred.iloc[:,0])
                #print(a)  ###
                #print(b)  ###
                x = np.linspace(a,b,1000)
                #print(x)  ###
                #print(self.predict(x,True))  ###
            else:
                x = list(set(self.predictores_df.iloc[:,0]))
            print(f'x: {x}')  ###
            line, = plt.plot(x,self.predict(x,True),'-',color='orange')
            if show:
                plt.show()
            return scatter,line

        # Caso 3D:
        if len(pred.columns) == 2:
            pass

#@title subclases de Modelos según Quant o Cual

class RQmodel(Modelo):
    '''
    ################################################
    respuesta:  cuantitativa
    ################################################
    Por ahora sólo respuestas unidimensionales
    '''
    def __init__(self,df,predictor,respuesta):
        # los predictores pueden ser vectores
        super().__init__(df=df,predictor=predictor,respuesta=respuesta)

    def __calc_residuos(self,**codigos): ### test vector de predictores y de respuestas
        '''
        ###############################
        Calcula los residuos del modelo
        ###############################
        Por ahora, sólo acepta respuestas unidimensionales
        '''
        import numpy as np
        residuos = []
        predictores = self.biny_pred(**codigos) #  matriz de predictores binarizada
        respuestas = self.respuestas_df # matriz de respuestas
        for i in range(len(predictores)):
            X = predictores.iloc[i]
            Y = respuestas.iloc[i]
            prediccion = self.predict(X)
            residuo = Y - prediccion
            residuos.append(residuo)
        self.__residuos = residuos

    def __calc_SE_residuos(self,**codigos):
        residuos = self.residuos(**codigos)
        n = self.longitud
        s = 0
        for valor in residuos:
            s += valor**2
        s *= 1/(n-2)
        s **= 1/2
        self.__SE_residuos = s

    def residuos(self):
        return self.resultado.resid

    def SE_residuos(self):
        return self.resultado.mse_resid**0.5

    def shapiro_residuos(self):
        '''
        Testea para
        H0: residuos normales
        H1: residuos no normales
        '''
        from scipy.stats import shapiro
        stat,p_valor1 = shapiro(self.residuos())
        return p_valor1

    def homoced_residuos(self):
        '''
        Testea para
        H0: varianza constante
        H1: varianza no constante
        '''
        residuos = self.residuos()
        X = sm.add_constant(self.biny_pred(**self.codigos),has_constant='add')
        from statsmodels.stats.diagnostic import het_breuschpagan, het_white
        bp_test = het_breuschpagan(residuos, X)
        bp_value = bp_test[1]
        return bp_value
    #=========================================#
    #     Métodos gráficos                    #
    #=========================================#

    def graficar_residuos(self,show=False,add=False): ### habría que reutilizar el graficar de Dataframe generalizado (hay que generalizar Dataframe.graficar())
                                            ### O sea, hacer que los residuos y los predictores se metan en un dataframe que va a graficarlos
        '''
        ############################################
        Grafica:
            - Número de observación, vs residuo
            - qqPlot Normal de los residuos
        ############################################
        '''
        residuos = self.residuos()
        X = self.predictores_df
        Residuos = QuantDatos((len(residuos),1),dtype=object)
        Residuos[:,0] = residuos

        import matplotlib.pyplot as plt
        if not add:
            plt.ioff()
        numeros = [x+1 for x in range(len(Residuos))]
        plt.figure(figsize=(5,10))
        plt.subplot(2,1,1)
        scatter = plt.scatter(numeros,residuos)
        plt.subplot(2,1,2)
        qqplot = Residuos.qqPlot('Normal',0,1,show)
        if show:
            plt.show()
        return scatter,qqplot

class PQmodel(Modelo):
    '''
    ################################################
    predictor: cuantitativo
    ################################################
    '''
    def __init__(self,df,predictor,respuesta):
        super().__init__(df=df,predictor=predictor,respuesta=respuesta)

class RCmodel(Modelo):
    '''
    ################################################
    respuesta: cualitativa
    ################################################
    '''
    def __init__(self,df,predictor,respuesta,**codigos):
        self.Rcodigos = codigos
        super().__init__(df=df,predictor=predictor,respuesta=respuesta)

    def biny_res(self,**codigos):
        '''
        Devuelve matriz binaria de categorias de las respuestas
        '''
        return self.respuestas_df.biny(**codigos)
    # Agregar binarización de las respuestas acá

class PCmodel(Modelo):
    '''
    ################################################
    predictor: cualitativo
    ################################################
    '''
    def __init__(self,df,predictor,respuesta,**codigos):
        # los predictores pueden ser vectores
        self.Pcodigos = codigos
        #print(f'PCmodel: {type(df)}') ###
        super().__init__(df=df,predictor=predictor,respuesta=respuesta)

    def biny_pred(self,**codigos):
        '''
        Devuelve matriz binaria de categorias de las predictoras
        '''
        return self.predictores_df.biny(**codigos)

    # Agregar binarización de las predictoras acá

#@title Modelos específicos

class RL(RQmodel,PQmodel,PCmodel):
    def __init__(self,df,predictor,respuesta,**codigos):
        '''
        Códigos de las variables cualitativas
        '''
        self.codigos = codigos
        self.__df = df
        self.codigos_pred = {codigo:codigos[codigo] for codigo in codigos if codigo in predictor}
        PCmodel.__init__(self,df=df,predictor=predictor,respuesta=respuesta,**self.codigos_pred)
        PQmodel.__init__(self,df=df,predictor=predictor,respuesta=respuesta)
        self.codigos = codigos
        self.__model()

    def __model(self):
        '''
        codigos:  escribe todas las columnas cualitativas, aunque no se
                  quiera ingresar un código específico.
            - Si se ingresa código: columna={clase_i:code_i}
            - Si no se ingresa código: columna=None
        ----------------------------------------------------------------
        Genera:
            - self.resultado
            - self.modelo
        '''
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        import numpy as np
        import pandas as pd
        if self.predictores != []:
            X = self.biny_pred(**self.codigos_pred)
            self.pred_bin = X
        else:
            n = len(self.respuestas_df.iloc[:,0])
            X = pd.DataFrame(np.zeros((n,1)),columns=['empty'])
        # si no metemos predictoras, queremos ser capaces de tener una Intercept
        #print(X)  ###
        # Generamos la fórmula de regresión, puesto que sino nos perdemos el atributo design_info
        formula = self.respuestas[0]+'~'
        columnas = X.columns
        #print(columnas) ###
        formula += columnas[0]
        for columna in columnas[1:]:
            formula += '+'+columna

        y = self.respuestas_df
        DF = pd.concat([y, X],axis=1)
        #print(DF)   ###
        modelo = smf.ols(formula,data=DF)
        resultado = modelo.fit()
        self.resultado = resultado
        self.modelo = modelo
        parametros = resultado.params
        param_names = parametros.index.tolist()
        param_values = parametros.values
        param_SE = resultado.bse
        param_PVAL = resultado.pvalues
        R_squared = resultado.rsquared
        R_squared_adj = resultado.rsquared_adj
        self.set_param('rsquared','VAL',R_squared)  # no sirve para distinta cantidad de regresoras
        self.set_param('rsquared_adj','VAL',R_squared_adj)  # sirve para distinta cantidad de regresoras
        # Guardamos los datos de los parámetros
        for i in range(len(param_names)):
            self.set_param(param_names[i],'VAL',param_values[i])
            self.set_param(param_names[i],'SE',param_SE[i])
            self.set_param(param_names[i],'PVAL',param_PVAL[i])

        def f(value,multiple=False):
            import numpy as np
            import pandas as pd
            if multiple:
                output = []
                for valor in value:
                    valor = [valor]
                    valor = pd.DataFrame(valor,columns=self.pred_bin.columns)
                    pred = self.resultado.get_prediction(valor)
                    summary = pred.summary_frame()
                    output.append(summary['mean'].iloc[0])
                return output

            value = [value]
            value = pd.DataFrame(value,columns=self.pred_bin.columns)
            pred = self.resultado.get_prediction(value)
            summary = pred.summary_frame()
            return summary['mean'].iloc[0]

        self.funcion = f

    #=============================================#
    #     Métodos de usuario                      #
    #=============================================#
    def ttest(self,param,value,conf,condition='equal'):
        '''
        =======================================
        Realiza el test de hipótesis:
            - H0: param >=< value
            - H1: param <!=> value
        al conf*100% de confianza

        Utiliza el estadístico

        T = (valor - param)/SE(param)

        que tiene distribución t-student
        con n-p grados de libertad
            - n:  cantidad de observaciones
            - p:  cantidad de parámetros
        =======================================
        Recibe:
          - param:  el nombre del parámetro a testear
          - value:  el valor que queremos testear
          - conf:  con cuánta confianza lo vamos a testear
          - condition:
              - greater
              - less
              - equal
        '''
        import scipy.stats as sp
        import numpy as np
        SE = self.param(param,'SE')
        B = self.param(param,'VAL')
        T = (B-value)/SE
        p = len(self.param_list())
        n = self.longitud
        p_val = 0
        signif = 1-conf

        if condition == 'greater':
            p_val = sp.t.sf(T,n-p)
        elif condition == 'less':
            p_val = sp.t.cdf(T,n-p)
        else:
            p_val = 2 * sp.t.sf(np.abs(T),n-p)

        if p_val <= signif:
            print('==============================')
            print(f'Se rechaza la hipótesis nula')
            print(f'Para el parámetro {param}')
            print(f'Para la condición {condition}')
            print(f'Para el valor {value}')
            print(f'Con {signif} de significancia')
            print(f'Con un p-valor de {p_val}')
            print('===============================')

        if p_val > signif:
            print('==========================================')
            print(f'No es posible rechazar la hipótesis nula')
            print(f'Para el parámetro {param}')
            print(f'Para la condición {condition}')
            print(f'Para el valor {value}')
            print(f'Con {signif} de significancia')
            print(f'Con un p-valor de {p_val}')
            print('===========================================')

    def conf_int(self,param,conf):
        alfa = 1-conf
        conf_df = self.resultado.conf_int(alpha=alfa)
        return conf_df.loc[param]

    def conf_pred(self,value,conf):
        import statsmodels.api as sm
        alfa = 1-conf
        #print(f'{value=}')  ###
        #print('----')
        if np.isscalar(value):
            value = [value]
        #print(f'{value=}')  ###
        #print('----')
        value = pd.DataFrame([value],columns=self.pred_bin.columns)
        #print(f'{value.shape=}')
        #print(f'value:\n{value}')
        pred = self.resultado.get_prediction(value)
        summary = pred.summary_frame(alpha=alfa)
        low = summary['obs_ci_lower'].iloc[0]
        up = summary['obs_ci_upper'].iloc[0]
        return (low,up)

    def SE_pred(self,value):
        import statsmodels.api as sm
        import numpy as np
        if np.isscalar(value):
            value = [value]
        value = pd.DataFrame([value],columns=self.pred_bin.columns)
        pred = self.resultado.get_prediction(value)
        summary = pred.summary_frame()
        mean_se = self.SE_Mpred(value)
        residual_se = self.SE_residuos()
        obs_SE = np.sqrt(mean_se**2 + residual_se**2)
        return obs_SE

    def conf_Mpred(self,value,conf):
        import statsmodels.api as sm
        alfa = 1-conf
        if np.isscalar(value):
            value = [value]
        value = pd.DataFrame([value],columns=self.pred_bin.columns)
        pred = self.resultado.get_prediction(value)
        summary = pred.summary_frame(alpha=alfa)
        low = summary['mean_ci_lower'].iloc[0]
        up = summary['mean_ci_upper'].iloc[0]
        return (low,up)

    def SE_Mpred(self,value):
        import statsmodels.api as sm
        import numpy as np
        if np.isscalar(value):
            value = [value]
        value = pd.DataFrame([value],columns=self.pred_bin.columns)
        pred = self.resultado.get_prediction(value)
        summary = pred.summary_frame()
        SE = summary['mean_se'].iloc[0]
        return SE

    def anova(self,categoria):
        '''
        ===============================================
        Realiza ANOVA sobre las clases de una categoría
        predictora especificada
        ===============================================
        '''
        import statsmodels.stats.anova as sma
        resultadoH1 = self.resultado
        predictores_new = self.predictores.copy()
        predictores_new.pop(predictores_new.index(categoria))
        print(predictores_new)
        modeloH0 = RL(df=self.__df,predictor=predictores_new,respuesta=self.respuestas)
        resultadoH0 = modeloH0.resultado

        return sma.anova_lm(resultadoH0,resultadoH1) # poner primero la H0 sino el p_val da mal

class Log(RCmodel,PQmodel,PCmodel):
    def __init__(self,df,predictor,respuesta,**codigos):
        self.codigos = codigos
        self.__df = df
        self.codigos_res = {codigo:codigos[codigo] for codigo in codigos if codigo in respuesta}
        self.codigos_pred = {codigo:codigos[codigo] for codigo in codigos if codigo in predictor}
        RCmodel.__init__(self,df=df,predictor=predictor,respuesta=respuesta,**self.codigos_res)
        PCmodel.__init__(self,df=df,predictor=predictor,respuesta=respuesta,**self.codigos_pred)
        PQmodel.__init__(self,df=df,predictor=predictor,respuesta=respuesta)
        self.codigos = codigos
        self.__model()

    def __model(self):
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        import pandas as pd
        X = self.biny_pred(**self.codigos_pred)
        self.pred_bin = X

        # Generamos la fórmula de regresión, puesto que sino nos perdemos el atributo design_info
        formula = self.respuestas[0]+'~'
        columnas = X.columns
        formula += columnas[0]
        for columna in columnas[1:]:
            formula += '+'+columna

        y = self.biny_res(**self.codigos_res)
        DF = pd.concat([y, X],axis=1)

        modelo = smf.logit(formula,data=DF) # ajuste Log
        resultado = modelo.fit()
        self.resultado = resultado
        self.modelo = modelo
        parametros = resultado.params
        param_names = parametros.index.tolist()
        param_values = parametros.values
        param_SE = resultado.bse
        param_PVAL = resultado.pvalues

        for i in range(len(param_names)):
            self.set_param(param_names[i],'VAL',param_values[i])
            self.set_param(param_names[i],'SE',param_SE[i])
            self.set_param(param_names[i],'PVAL',param_PVAL[i])

        def f(value,multiple=False):
            import numpy as np
            import pandas as pd
            if multiple:
                output = []
                for valor in value:
                    valor = [valor]
                    valor = pd.DataFrame(valor,columns=self.pred_bin.columns)
                    pred = self.resultado.get_prediction(valor)
                    summary = pred.summary_frame()
                    output.append(summary['predicted'].iloc[0])
                return output

            value = [value]
            #print(value)    ###
            value = pd.DataFrame(value,columns=self.pred_bin.columns)
            #print(value)    ###
            pred = self.resultado.get_prediction(value)
            summary = pred.summary_frame()
            return summary['predicted'].iloc[0]

        self.funcion = f
    #########################################
    #       Métodos de usuario              #
    #########################################
    def conf_int(self,param,conf):
        alfa = 1-conf
        conf_df = self.resultado.conf_int(alpha=alfa)
        return conf_df.loc[param]

    def conf_pred(self,value,conf):
        import statsmodels.api as sm
        alfa = 1-conf
        #print(f'{value=}')  ###
        #print('----')
        if np.isscalar(value):
            value = [value]
        #print(f'{value=}')  ###
        #print('----')
        value = pd.DataFrame([value],columns=self.pred_bin.columns)
        #print(f'{value.shape=}')
        #print(f'value:\n{value}')
        pred = self.resultado.get_prediction(value)
        summary = pred.summary_frame(alpha=alfa)
        low = summary['ci_lower'].iloc[0]
        up = summary['ci_upper'].iloc[0]
        return (low,up)

    def SE_pred(self,value):
        import statsmodels.api as sm
        import numpy as np
        if np.isscalar(value):
            value = [value]
        value = pd.DataFrame([value],columns=self.pred_bin.columns)
        pred = self.resultado.get_prediction(value)
        summary = pred.summary_frame()
        SE = summary['se'].iloc[0]
        return SE

    def test_model(self,train=0.8,cut=0.5): # QUIZAS implementar selección arbitraria de filas
        '''
        ======================================
        Realiza un testeo por cross validation
        ======================================
        Recibe:
            - train:  porcentaje de datos de entrenamiento
            - cut:  a partir de qué porcentaje son considerados
                    positivos
        '''
        import random
        cache = True
        try:
            self.cache
        except:
            cache = False

        if not cache or self.cache == {}:
            n = len(self.__df.iloc[:,0])

            #print(n)
            n_train = int(n*train)
            n_test = n-n_train
            #print(n_train)
            indices_train = random.sample(range(n),n_train)
            indices_train.sort()
            indices_test = [i for i in range(n) if i not in indices_train]

            df_train = self.__df.iloc[indices_train,:].reset_index(drop=True)  ### MUY IMPORTANTE
            DF_train = Dataframe(df_train,columns=df_train.columns)
            #print(DF_train) ###
            #print(self.predictores) ###
            #print(self.respuestas)  ###
            #print(self.codigos_res) ###
            # 1)  modelo que queremos testear
            modelo_train = Log(df=DF_train,predictor=self.predictores,respuesta=self.respuestas,**self.codigos_res)

            # 2)  tomamos los datos de testeo

            datos_test = self.predictores_df
            datos_test = datos_test.iloc[indices_test]
            datos_test = np.array(datos_test,dtype=object)

            # 3)  Generamos los conjuntos a comparar
            prob_pred = modelo_train.predict(datos_test,multiple=True)
            prob_test = [1 if x=='Yes' else 0 for x in self.respuestas_df.iloc[indices_test,0]]
            DF_table = Dataframe()
            DF_table['test'] = prob_test

            # 4)  Guardamos en el cache:
            self.cache = {}
            self.cache.update({'n_test':n_test,'indices_test':indices_test,'indices_train':indices_train,'prob_pred':prob_pred,'DF_table':DF_table})

        y_pred = [1 if x>=cut else 0 for x in self.cache['prob_pred']]

        #print(y_pred)     ###
        #print(prob_test)  ###
        #print(len(y_pred))  ###
        #print(len(prob_test)) ###
        # 4) Tabla de comparaciones

        DF_table = self.cache['DF_table']
        DF_table['pred'] = y_pred
        n_test = self.cache['n_test']
        print('-----------------------------------------')
        print('     Predicciones vs Realidad (freq)    ')
        table = DF_table.prob_joint('pred','test',freq=True)
        print('---------------------------------')
        print('     Predicciones vs Realidad    ')
        DF_table.prob_joint('pred','test')
        print('---------------------------------')
        marginal_error = (table.iloc[0,1] + table.iloc[1,0])/n_test
        sens=table.iloc[1,1]/(table.iloc[1,1]+table.iloc[0,1])
        spec=table.iloc[0,0]/(table.iloc[0,0]+table.iloc[1,0])
        PYY =table.iloc[1,1]/(table.iloc[1,1]+table.iloc[1,0])  # P(real: si | ajuste: si)
        PNN =table.iloc[0,0]/(table.iloc[0,0]+table.iloc[0,1])  # P(real: no | ajuste: no)
        PYN = 1-PNN                                             # P(real: si | ajuste: no)
        PNY = 1-PYY                                             # P(real: no | ajuste: si)
        print(f'     Error marginal:  {marginal_error}  ')
        print('---------------------------------')
        print(f'Falsos positivos:\n')
        print(f'marginal: {table.iloc[1,0]/n_test}')
        print(f'P(ajuste: si | real: no):{1-spec}')
        print(f'--------------------------')
        print(f'Falsos negativos:\n')
        print(f'marginal: {table.iloc[0,1]/n_test}')
        print(f'P(ajuste: no | real: si): {1-sens}')
        print(f'--------------------------')
        print(f'Sensibilidad: {sens}')
        print(f'Especificidad: {spec}')
        print(f'--------------------------')
        print(f'P(real: si | ajuste: si) {PYY}')
        print(f'P(real: no | ajuste: no) {PNN}')
        print(f'P(real: si | ajuste: no) {PYN}')
        print(f'P(real: no | ajuste: si) {PNY}')
        return {'err':marginal_error,'sens':sens,'spec':spec,'PYY':PYY,'PNN':PNN,'PYN':PYN,'PNY':PNY,'indices_train':indices_train,'indices_test':indices_test}
    
# esta branch es test