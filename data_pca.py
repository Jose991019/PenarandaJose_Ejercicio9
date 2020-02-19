import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datos = pd.read_csv('USArrests.csv')
columnas = ['Murder', 'Assault', 'UrbanPop','Rape']
datos2 = datos[columnas]
datos2 = (datos2 - datos2.mean())/datos2.std()

cov_matrix = np.cov(datos2.T)

values, vectors = np.linalg.eig(cov_matrix)

vector_A = vectors[:,0]
vector_B = vectors[:,1]

plt.figure(figsize=(12,12))
datos = np.array(datos)

for i in range(len(datos)):
    ciudades = datos[:,0][i]
    v = np.array(datos2.iloc[i])
    x = np.dot(vector_A, v) 
    y = np.dot(vector_B, v) 
    plt.text(x,y, ciudades, fontsize=10, color='blue')
    plt.scatter(x, -y, s=0.001)

for j in range(len(columnas)):
    plt.arrow(0.0, 0.0, 2.5*vector_A[j], 2.5*vector_B[j], color='red', head_width=0.1)
    plt.text(2.75*vector_A[j], 2.75*vector_B[j], columnas[j], color='red')
plt.ylim(-3,3)
plt.xlabel('Primera Componente Principal')
plt.ylabel('Segunda Componente Principal')
plt.savefig('arrestos.png', bbox_inches='tight')

values2 = values*25
y = [values2[0]]
for i in values2[1:]:
    y.append(y[-1]+i)
x = range(1,5)
plt.figure()
plt.plot(x,y)
plt.scatter(x,y)
plt.grid()
plt.ylim(0,100)
plt.xlabel('Número de autovalores')
plt.ylabel('Porcentaje de varianza explicada')
plt.savefig('varianza_arrestos.png')

datos = pd.read_csv('Cars93.csv')
columnas = ['MPG.city', 'MPG.highway', 'EngineSize', 'Horsepower', 'RPM', 'Rev.per.mile', 
          'Fuel.tank.capacity', 'Length', 'Width', 'Turn.circle', 'Weight']
datos2 = datos[columnas]
datos2 = (datos2 - datos2.mean())/datos2.std()


cov_matrix = np.cov(datos2.T)
values, vectors = np.linalg.eig(cov_matrix)

vector_A = vectors[:,0]
vector_B = vectors[:,1]

plt.figure(figsize=(12,12))
datos = np.array(datos)

for i in range(len(datos)):
    model = datos[:,2][i]
    v = np.array(datos2.iloc[i])
    x = np.dot(vector_A, v) 
    y = np.dot(vector_B, v) 
    plt.text(x,-y, model, fontsize=10, color='blue')
    plt.scatter(x, y, s=0.001)

for j in range(len(columnas)):
    plt.arrow(0.0, 0.0, 2.5*vector_A[j], -2.5*vector_B[j], color='red', head_width=0.1)
    plt.text(2.75*vector_A[j], -2.75*vector_B[j], columnas[j], color='red')
plt.ylim(-3,3)
plt.xlabel('Primera Componente Principal')
plt.ylabel('Segunda Componente Principal')
plt.savefig('cars.png', bbox_inches='tight')

values2 = values*9.09
y = [values2[0]]
for i in values2[1:]:
    y.append(y[-1]+i)
x = range(1,12)
plt.figure()
plt.plot(x,y)
plt.scatter(x,y)
plt.grid()
plt.ylim(0,100)
plt.xlabel('Número de autovalores')
plt.ylabel('Porcentaje de varianza explicada')
plt.savefig('varianza_cars.png')

