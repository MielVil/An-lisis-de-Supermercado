
# Importamos librerías y módulos a utilizar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from scipy.stats import chi2_contingency

#Cargamos el dataset
datos = pd.read_csv("supermarket_sales.csv")
print("Number of datapoints:", len(datos))
datos[:]

#Características generales de los datos:
datos.describe()

print(datos.info(), '\n')

NA = datos.isna().sum()
print(NA[NA != 0])

# Observar variables que podrían ser categóricas:
for j in [x for x in datos.columns]:
    print(pd.concat([datos[j].value_counts(),
                     datos[j].value_counts(normalize = True)],
                     axis = 1), '\n')

"""Variables categóricas:

En esta parte del código, además de identificar las variables de tipo categórico, hacemos un recorrido por cada una de ellas para explorar las diversas clasificaciones y contabilizar la cantidad de registros por cada categoría, entre ellas se encuentran: Branch, City, Custumer type, Gender, Product Line y Payment.

# **Limpieza de Datos**
"""

#Construimos Data Frame
df = pd.DataFrame(datos)
df

#Verificamos nuevamente que no existan datos faltantes:
df.isnull().sum()

"""Vemos que ninguna variable tiene datos faltantes."""

#Verificamos si es de utilidad la variable de 'gross margin percentage' generando una extracción de sus valores únicos pues se observó previamente que su desviación estándar es cero; en escencia debería tratarse de una constante:
df['gross margin percentage'].unique()

"""Vemos que esta variable se mantiene constante para todos los registros y que no tiene un valor significativo, pues puede tratarse de una regla de negocio. Por ello podemos eliminarla."""

df = df.drop(['gross margin percentage'], axis = 1)
df.head(3)

"""# **Análsis Exploratorio de Datos**

## Exploración y comparación a detalle entre variables

Variables cuantitativas:


*   Unit Price
*   Quantity
*   Tax 5%
*   Total
"""

#Variable 'Gross Income': Precio de cada producto

#Info de la variable
print(df['gross income'].describe())
#Histograma
plt.figure(figsize=(10, 6))
sns.histplot(df['gross income'], bins=20, kde=True, color='skyblue')
plt.title('Distribución de las Ganancias')
plt.xlabel('Gross Income')
plt.ylabel('Frecuencia')
plt.show()

"""La variable ingreso bruto brinda información de las ganancias brutas obtenidas en cada transacción de venta. El ingreso bruto promedio por transacción es de aproximadamente 15.38 unidades monetarias; esto representa una ganancia promedio por venta."""

#Variable 'Unit Price': Precio de cada producto

#Info de la variable con max y min:
min_precio = df['Unit price'].min()
max_precio = df['Unit price'].max()

print(f"El precio mínimo es: {min_precio}")
print(f"El precio máximo es: {max_precio}")

# Tomamos columna de precio
plt.figure(figsize=(10, 6))
plt.hist(df['Unit price'], bins=30, edgecolor='k', alpha=0.7)
plt.title('Distribución de Precios de Productos')
plt.xlabel('Unit price')  # Precio unitario
plt.ylabel('Frecuencia')  # Frec

# Marcamos los el mínimo y máximo precio en el histograma
plt.axvline(min_precio, color='r', linestyle='dashed', linewidth=1, label=f'Mínimo: {min_precio}')
plt.axvline(max_precio, color='g', linestyle='dashed', linewidth=1, label=f'Máximo: {max_precio}')

plt.legend()
plt.show()

# Diagrama de caja (Boxplot)
plt.figure(figsize=(4, 5))
bx = df.boxplot(column=['Unit price'], patch_artist=True)

for patch in bx.patches:
    patch.set_facecolor('lightblue')
plt.title('Boxplot de Precio x Unidad')
plt.show()

"""Los precios se distribuyen con media de 55.67 y desviación estándar de 26.5.
El precio mínimo que se maneja es de 10.08 y el más alto es de 99.96.

"""

#Gráfico de Unit Price por linea de producto
sns.set(style="whitegrid")

# Crear el gráfico de barras
plt.figure(figsize=(12, 6))
ax = sns.barplot(data=df, x='Product line', y='Unit price', palette='viridis')

# Agregar números en la parte superior de cada barra
for p in ax.patches:
    ax.annotate('{:.2f}'.format(p.get_height()),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.title('Precio unitario por línea de producto')
plt.xticks(rotation=45, ha='right')  # Rotar etiquetas del eje x para mayor legibilidad
plt.xlabel('Línea de producto')
plt.ylabel('Precio unitario')
plt.tight_layout()
plt.show()

"""Observamos el precio unitario promedio por cada línea de producto, vemos que la línea con precio promedio mas alto es la de Fashion accesories."""

#Variable 'Quantity': Número de productos adquiridos por cliente.

#Info de la variable
print(df['Quantity'].describe())
#Histograma
plt.figure(figsize=(10, 6))
sns.histplot(df['Quantity'], bins=10, kde=True, color='skyblue')
plt.title('Distribución de Quantity')
plt.xlabel('Quantity')
plt.ylabel('Frecuencia')
plt.show()

"""Observamos que en su mayoría los clientes suelen comprar entre 9 y 10 artículos."""

# Diagrama de caja (Boxplot)
plt.figure(figsize=(10, 6))
bx = df.boxplot(column=['Quantity'], patch_artist=True)

for patch in bx.patches:
    patch.set_facecolor('lightblue')
plt.title('Boxplot de N°. Productos x Cliente')
plt.show()

#Variable 'Tax 5%': Tasa de impuesto a la compra

#Info de la variable
print(df['Tax 5%'].describe())
#Histograma
plt.figure(figsize=(10, 6))
sns.histplot(df['Tax 5%'], bins=10, kde=True, color='skyblue')
plt.title('Distribución de Tax 5%')
plt.xlabel('Tax 5%')
plt.ylabel('Frecuencia')
plt.show()

# Diagrama de caja (Boxplot)
plt.figure(figsize=(10, 6))
bx = df.boxplot(column=['Tax 5%'], patch_artist=True)

for patch in bx.patches:
    patch.set_facecolor('lightblue')
plt.title('Boxplot de Tax')
plt.show()

#Variable 'Total': Precio total con impuesto incluido

#Info de la variable
print(df['Total'].describe())
#Histograma
plt.figure(figsize=(10, 6))
sns.histplot(df['Total'], bins=10, kde=True, color='skyblue')
plt.title('Distribución de Total')
plt.xlabel('Total')
plt.ylabel('Frecuencia')
plt.show()

"""Observamos la distribución de las ventas totales, las cuales son en promedio de 322.96, ahora bien, para visualizar a qué ciudades percenecen estas ventas:"""

# Agrupar y sumamos el Total de ventas por ciudades en el dataframe:
ventas_por_ciudad = df.groupby('City')['Total'].sum().sort_values(ascending=False)
# Y calculamos su porcentaje
porcentaje_ventas = (ventas_por_ciudad / ventas_por_ciudad.sum()) * 100

# Ahora bien, visualizamos las ventas agrupadas por ciudad:
plt.figure(figsize=(10, 6))
ventas_por_ciudad.plot(kind='bar', color='skyblue', edgecolor='black')

# Pegar etiquetas al gráfico
for i, v in enumerate(ventas_por_ciudad):
    plt.text(i, v + 1, f"{porcentaje_ventas.iloc[i]:.2f}%", ha='center')
plt.title('Porcentaje de Ventas por Ciudad')
plt.xlabel('Ciudad')
plt.ylabel('Ventas Totales')
plt.xticks(rotation=45)
plt.show()

# Diagrama de caja (Boxplot)
plt.figure(figsize=(10, 6))  # Configura el tamaño del gráfico
bx = df.boxplot(column=['Total'], patch_artist=True)
# Personaliza el color de fondo de las cajas
for patch in bx.patches:
    patch.set_facecolor('lightblue')
plt.title('Boxplot de Total')
plt.show()

"""Vemos que las ventas son muy similares en las tres ciudades, sin embargo la mayor concentración de ventas se tiene en Naypytaw con un 34.24%

*Segmentación de clientes según tipo y ciudad.*
"""

#Agrupar y sumar los clientes de acuerdo al tipo y a la ciudad a la que pertenecen
cont_clientes = df.groupby(['City', 'Customer type']).size().unstack(fill_value=0)

#Calculo de porcentaje
pc_clientes=cont_clientes.div(cont_clientes.sum(axis=1), axis=0) * 100
pc_clientes.plot(kind='bar', stacked=True, figsize=(8, 5))

#Gráfica de barras de la información
plt.title('Tipo de clientes por Ciudad')
plt.xlabel('Ciudad')
plt.ylabel('Clientes')
plt.legend(title='Tipo de Cliente', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()

"""*Segmentación de clientes por género y ciudad*"""

#Conteo de clientes según su género y la ciudad donde se efectúo la compra
cont_clientes2 = df.groupby(['City', 'Gender']).size().unstack(fill_value=0)

#Cálculo de porcentaje
pc_clientes2 = cont_clientes.div(cont_clientes.sum(axis=1), axis=0) * 100
pc_clientes2.plot(kind='bar', stacked=True, figsize=(8, 5))

#Gráfica de barras de la información
plt.title('Clientes por género por Ciudad')
plt.xlabel('Ciudad')
plt.ylabel('Clientes')
plt.legend(title='Tipo de Cliente', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()

"""Se observa que la segmentación de clientes dadas estas caracteristicas no presenta una variación importante.
Los clientes de las tres sucursales con ubicación en cada ciudad se conforman por el 50% hombres y 50% mujeres, donde estos a su vez, el 50% son miembros y el otro 50% no lo es.

*Segmentación de clientes por valor de compra*
"""

#Definimos rangos de compra
bins = [0, 200, 600, float('inf')]
labels = ['Bajo', 'Medio', 'Alto']

#Asignación de clientes según el monto de compra
df['Grupo de compra'] = pd.cut(df['Total'], bins=bins, labels=labels)

#Conteo
compra_counts = df['Grupo de compra'].value_counts().sort_index()

#Gráfica
compra_counts.plot(kind='bar', figsize=(8, 5))

plt.title('Distribución de Clientes por valor de Compra')
plt.xlabel('Grupo de Compra')
plt.ylabel('Número de Clientes')

plt.show()

"""Se observa que la mayoría de los clientes se concentran en un rango medio seguido del rango bajo.

Por otra parte, sabemos que la mayoría de los clientes compra entre 9 y 10 productos, por lo que podemos inferir que el supermercado es de "gama baja - media".

Variables cualitativas:

*   City
*   Customer type
*   Gender
*   Payment
*   Product line
"""

#Variable 'City': Sucursales

#Info de la variable
print(df['City'].describe())

#Obtención de porcentaje que representa cada ciudad
porcentajes = df['City'].value_counts(normalize=True)*100

#Pie Chart
plt.figure(figsize=(10, 6))
plt.pie(porcentajes, labels=porcentajes.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Distribución de la presencia del supermercado en las ciudades')
plt.show()

"""Observamos que el mercado esta distribuido en cantidades generales en las tres ciudades, sin embargo el más grande es Yangon y el menor es Naypyitaw en el cual observamos que concentra la mayor cantidad de ventas."""

plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='gross income', hue='City', multiple='stack')

# Gráfico
plt.title('Distribución del Ingreso Bruto por Ciudad')
plt.xlabel('Ingreso Bruto')
plt.ylabel('Densidad')
plt.show()

"""Observamos que los ingresos brutos parecen seguir una distribución positivamente asimétrica, con una media de 15.37. En cuanto a la ciudad en la cual se representa el mayor numero de ingresos vemos que es Yangon.

Vemos que la mayor cantidad de clientes se concentra en Yangon
"""

#Variable 'Customer type': Clientes que hicieron uso de membresía "Members" y los que no, "Normal".

#Info de la variable
print(df['Customer type'].describe())

#Obtención de porcentaje que representa cada ciudad
porcentajes = df['Customer type'].value_counts(normalize=True)*100

#Pie Chart
plt.figure(figsize=(10, 6))
plt.pie(porcentajes, labels=porcentajes.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Distribución de clientes miembros y normales')
plt.show()

#Variable 'Gender': Composición de la población por género, masculino y femenino.

#Info de la variable
print(df['Gender'].describe())

#Obtención de porcentaje que representa cada género
porcentajes = df['Gender'].value_counts(normalize=True)*100

#Pie Chart
plt.figure(figsize=(10, 6))
plt.pie(porcentajes, labels=porcentajes.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Distribución de clientes por género')
plt.show()

#Variable 'Payment': Método de pago utilizado por los clientes; efectivo, tarjeta de crédito o e-wallet.

#Info de la variable
print(df['Payment'].describe())

#Obtención de porcentaje que representa cada método de pago
porcentajes = df['Payment'].value_counts(normalize=True)*100

#Pie Chart
plt.figure(figsize=(10, 6))
plt.pie(porcentajes, labels=porcentajes.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Distribución por método de pago')
plt.show()

#Variable 'Product line': Categoría de productos; electrónicos, moda, alimentos, bebidas, salud y belleza, casa y lifestyle y deportes y viaje.

#Info de la variable
print(df['Product line'].describe())

#Obtención de porcentaje que representa cada categoría de producto
porcentajes = df['Product line'].value_counts(normalize=True)*100

#Pie Chart
plt.figure(figsize=(10, 6))
plt.pie(porcentajes, labels=porcentajes.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Tipo de producto')
plt.show()

#Comparación género - productos

#Tabla de contingencia
t_contingencia = pd.crosstab(df['Gender'], df['Product line'])

#Heatmap para la visualización
plt.figure(figsize=(10, 6))
sns.heatmap(t_contingencia, annot=True, cmap='RdBu', fmt='g')
plt.title('Relación entre género y productos comprados de acuerdo al departamento')
plt.xlabel('Departamento')
plt.ylabel('Género')
plt.show()

#Comparamos las variables Gender y Product line para ver los comportamientos actuales de los consumidores por género en cada categoría de linea de poducto.
plt.figure(figsize=(10, 6))
#Construimos el gráfico:
sns.countplot(x='Product line', hue='Gender', data=df)
plt.title('Distribución de Género por Línea de Producto')
plt.xlabel('Línea de Producto')
plt.ylabel('Cantidad')
plt.xticks(rotation=45)
plt.legend(title='Género')

plt.show()

"""De acuerdo al gráfico, las mujeres compran mas productos de "accesorios de moda", mientras que los hombres compran mayormente productos de "salud y bbelleza".

:### Comparación entre lineas de productos y ventas totales
"""

#Analizamos las ventas totales por en cada línea de producto

plt.figure(figsize = (12,6))
barplot = sns.barplot(x = df['Total'], y = df['Product line'], palette='viridis')
# Anotaciones sobre valores exactos
for p in barplot.patches:
    width = p.get_width()
    plt.text(p.get_width(), p.get_y() + p.get_height() / 2. + 0.2, '{:1.2f}'.format(width),
             ha="left", va="center")
plt.xlabel('Ventas Totales')
plt.ylabel('Línea de Producto')
plt.title('Ventas Totales por Línea de Producto')
plt.show()

"""Vemos que el departamento con mayor cantidad de ventas es el de Home and lifestyle."""

# Ventas totales por línea de productos en  vertical y con datos numéricos
ventas_por_producto = df.groupby('Product line')['Total'].sum().reset_index()
sns.barplot(x='Product line', y='Total', data=ventas_por_producto)
plt.title('Ventas totales por línea de productos')
plt.xticks(rotation=45)
plt.show()

# Calcular la venta total por línea de productos
ventas_por_producto = df.groupby('Product line')['Total'].sum().reset_index()

# Calcular la cantidad de productos vendidos por línea de productos
cantidad_por_producto = df.groupby('Product line')['Quantity'].sum().reset_index()

print(ventas_por_producto)

print(cantidad_por_producto)

"""1. Comparación de Ventas Totales por Línea de Productos:

La línea "Food and beverages" tiene las ventas totales más altas ($56144.84), lo que indica que es la categoría más rentable en términos de ingresos.

2. Comparación de Cantidades Vendidas por Línea de Productos:

"Electronic accessories" tiene la mayor cantidad de productos vendidos (971 unidades), lo que puede indicar una alta demanda para esta categoría.

3. Segmentación de Estrategias de Marketing:

Puede considerarse enfocar estrategias de marketing principalmente a "Health and beauty" para incrementar sus ventas totales, ya que tiene el menor total de ventas ($49193.74).
"""

# Calcular la venta total por línea de productos
ventas_por_producto = df.groupby('Product line')['Total'].sum().reset_index()

# Calcular la cantidad de productos vendidos por línea de productos
cantidad_por_producto = df.groupby('Product line')['Quantity'].sum().reset_index()

print(ventas_por_producto)
print(cantidad_por_producto)

"""### Comparación entre sucursales y su rendimiento"""

# Calcular la venta total y ganancias por sucursal
ventas_por_sucursal = df.groupby('Branch')['Total'].sum().reset_index(name='Ventas Totales')
ganancias_por_sucursal = df.groupby('Branch')['gross income'].sum().reset_index(name='Ganancias Totales')

# Calcular la cantidad de ventas por sucursal
cantidad_ventas_por_sucursal = df.groupby('Branch')['Invoice ID'].count().reset_index(name='Cantidad de Ventas')

# Mostrar juntos los resultados
resultados_sucursal = pd.merge(ventas_por_sucursal, ganancias_por_sucursal, on='Branch')
resultados_sucursal = pd.merge(resultados_sucursal, cantidad_ventas_por_sucursal, on='Branch')

# Mostrar los resultados
print(resultados_sucursal)

# Calcular el ingreso promedio por venta
resultados_sucursal['Ingreso Promedio por Venta'] = resultados_sucursal['Ventas Totales'] / resultados_sucursal['Cantidad de Ventas']

# Calcular la ganancia promedio por venta
resultados_sucursal['Ganancia Promedio por Venta'] = resultados_sucursal['Ganancias Totales'] / resultados_sucursal['Cantidad de Ventas']

# Mostrar los resultados
print(resultados_sucursal)

# Analizamos ahora el ingreso y ganancia promedio por venta en cada sucursal
fig, ax1 = plt.subplots(figsize=(10, 6))

# Gráfico de Ingreso Promedio por Venta
color = 'tab:blue'
ax1.set_xlabel('Sucursal')
ax1.set_ylabel('Ingreso Promedio por Venta', color=color)
ax1.bar(resultados_sucursal['Branch'], resultados_sucursal['Ingreso Promedio por Venta'], color=color, alpha=0.6, label='Ingreso Promedio por Venta')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

# Crear un segundo eje para la Ganancia Promedio por Venta
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Ganancia Promedio por Venta', color=color)
ax2.plot(resultados_sucursal['Branch'], resultados_sucursal['Ganancia Promedio por Venta'], color=color, marker='o', label='Ganancia Promedio por Venta')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

# Título
plt.title('Ingreso y Ganancia Promedio por Venta por Sucursal')
fig.tight_layout()
plt.show()

"""
El análisis muestra que las ventas totales son similares en las sucursales A, B y C, alrededor de $106,000 cada una. Pero notamos que las ganancias totales varían ligeramente, al ser más altas en la sucursal C, seguida por la A y luego la B. La sucursal A registra la mayor cantidad de ventas, mientras que la C tiene un ingreso y ganancia promedio por venta ligeramente más alto, sugiriendo una posible eficiencia operativa o estrategia de precios más efectiva en esa sucursal.

Si contaramos con mayor información sobre el tipo de productos en cada línea, promociones u otras estrategias de venta implementadas en la sucursal C podríamos encontrar su valor agregado y replicar estas técnicas en las otras dos tiendas."""

# Análisis del Mix de Productos
# Calcular la suma de ventas por línea de productos para cada sucursal
sales_by_product_line = df.groupby(['Branch', 'Product line'])['Total'].sum().reset_index()

# Visualizar los resultados utilizando un gráfico de barras
plt.figure(figsize=(10, 6))
sns.barplot(x='Product line', y='Total', hue='Branch', data=sales_by_product_line)
plt.title('Ventas por Línea de Productos y Sucursal')
plt.xlabel('Línea de Producto')
plt.ylabel('Ventas Totales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Comparación por Método de Pago
# Calcular la cantidad de ventas por método de pago para cada sucursal
sales_by_payment_method = df.groupby(['Branch', 'Payment'])['Invoice ID'].count().reset_index()

# Visualizar los resultados utilizando un gráfico de barras apiladas
plt.figure(figsize=(10, 6))
sns.barplot(x='Branch', y='Invoice ID', hue='Payment', data=sales_by_payment_method)
plt.title('Cantidad de Ventas por Método de Pago y Sucursal')
plt.xlabel('Sucursal')
plt.ylabel('Cantidad de Ventas')
plt.tight_layout()
plt.show()

"""1. Análisis de desempeño relativo

La sucursal C tiene el ingreso promedio por venta más alto, lo que indica que, en promedio, cada venta genera más ingresos en la sucursal C que en las sucursales A y B. La sucursal C también tiene la ganancia promedio por venta más alta, lo que indica que cada venta es más rentable en la sucursal C en comparación con las sucursales A y B.

Se puede notar que la sucursal C tiene las ventas más altas en "Food and beverages" y "Fashion accessories", mientras que la sucursal A tiene las ventas más altas en "Home and lifestyle" y "Sports and travel".

En cuanto a las preferencias de pago observamos que ue la mayoría de los clientes en la sucursal A prefieren pagar con Ewallet, mientras que en la sucursal C, la mayoría elige pagar en efectivo.
"""

#Analizamos el Gross income por cada línea de producto
# Calcular el ingreso bruto total por línea de producto
ingresos_por_producto = df.groupby('Product line')['gross income'].sum().sort_values(ascending=False)

# Configurar el estilo de seaborn
sns.set(style="whitegrid")

# Crear el gráfico de barras
plt.figure(figsize=(12, 6))
sns.barplot(x=ingresos_por_producto.index, y=ingresos_por_producto.values, palette='viridis')
plt.title('Ingreso bruto por línea de producto')
plt.xlabel('Línea de producto')
plt.ylabel('Ingreso bruto')
plt.xticks(rotation=45, ha='right')  # Rotar etiquetas del eje x para mayor legibilidad
plt.tight_layout()
plt.show()

"""Vemos nuevamente que los rendimientos mas altos son de la línea de Home and Lifestyle como la mayor cantidad de ventas.
Observamos que es necesario implementar estrategias de marketing y ventas para aumentar el rendimiento en la línea Health and beauty.
"""

#Podemos también analizar el desempeño de cada línea de producto revisando la variable Rating:
# Configuración de estilo para seaborn
sns.set(style="whitegrid")

# Crear el gráfico de caja
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Product line', y='Rating', palette='inferno')
plt.title('Rating por línea de producto')
plt.xlabel('Línea de producto')
plt.ylabel('Rating')
plt.xticks(rotation=45, ha='right')  # Rotar etiquetas del eje x para mayor legibilidad
plt.ylim(1, 10)  # Establecer límites del eje y de 1 a 10
plt.tight_layout()
plt.show()

"""De entrada vemos que la media de satisfacción está en 7 para todas las líneas de producto. Sin embargo la calificación promedio mas alta en satisfacción la tiene la línea de Food and beverage con 7.2 en rating. Mientras que la línea con menor rating promedio es Home and Lifestyle, lo cual es curioso dado que es la línea que representa mayor nivel de ganancias al supermercado. Es preciso que el supermercado mejore su estrategia de ventas o atención y servisio al cliente en este departamento para  mantener este nivel de rendimiento de ventas o incluso incrementarlo."""

#Comparación género - método de pago

t_contingencia2 = pd.crosstab(df['Gender'], df['Payment'])

#Heatmap para la visualización
plt.figure(figsize=(10, 6))
sns.heatmap(t_contingencia2, annot=True, cmap='RdBu', fmt='g')
plt.title('Relación entre género y método de pago')
plt.xlabel('Método de pago')
plt.ylabel('Género')
plt.show()

#Comparamos las variables Gender y método de pago para ver los comportamientos actuales de los consumidores por género
plt.figure(figsize=(10, 6))
#Construimos el gráfico:
sns.countplot(x='Payment', hue='Gender', data=df)
plt.title('Distribución de Género por Forma de pago ')
plt.xlabel('Payment')
plt.ylabel('Cantidad')
plt.xticks(rotation=45)
plt.legend(title='Género')

plt.show()

"""En cuento al método de pago, las mujeres pagan mayormente en efectivo mientras que los hombres en monedero electrónico."""

#Relación género - tipo de cliente

t_contingencia3 = pd.crosstab(df['Gender'], df['Customer type'])

#Heatmap para la visualización
plt.figure(figsize=(10, 6))
sns.heatmap(t_contingencia3, annot=True, cmap='RdBu', fmt='g')
plt.title('Relación entre género y tipo de cliente')
plt.xlabel('Tipo de cliente')
plt.ylabel('Género')
plt.show()

#Realizamos una histograma del género y el tipo de cliente:
plt.figure(figsize=(10, 6))
#Construimos el gráfico:
sns.countplot(x='Customer type', hue='Gender', data=df)
plt.title('Distribución de Género por Tipo de cliente')
plt.xlabel('Customer type')
plt.ylabel('Cantidad')
plt.xticks(rotation=45)
plt.legend(title='Género')

plt.show()

"""La mayoría de la población que es clasificada como miembro es mujer.

### Comparación entre tipos de clientes
"""

# Comparación entre Tipos de Clientes: Ventas y Ganancias
# Calcular la suma de ventas y ganancias por tipo de cliente
sales_by_customer_type = df.groupby('Customer type')[['Total', 'gross income']].sum().reset_index()

# Visualizar los resultados
plt.figure(figsize=(10, 6))
sns.barplot(x='Customer type', y='Total', data=sales_by_customer_type, palette='muted', ci=None)
plt.title('Ventas por Tipo de Cliente')
plt.xlabel('Tipo de Cliente')
plt.ylabel('Ventas Totales')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Customer type', y='gross income', data=sales_by_customer_type, palette='muted', ci=None)
plt.title('Ganancias por Tipo de Cliente')
plt.xlabel('Tipo de Cliente')
plt.ylabel('Ganancias Totales')
plt.tight_layout()
plt.show()

# Evaluación de la Satisfacción (Rating) entre Tipos de Clientes
# Calcular el promedio de rating por tipo de cliente
rating_by_customer_type = df.groupby('Customer type')['Rating'].mean().reset_index()

# Visualizar los resultados
plt.figure(figsize=(8, 5))
sns.barplot(x='Customer type', y='Rating', data=rating_by_customer_type, palette='muted', ci=None)
plt.title('Satisfacción Promedio por Tipo de Cliente')
plt.xlabel('Tipo de Cliente')
plt.ylabel('Satisfacción Promedio (Rating)')
plt.ylim(0, 10)  # Establecer el rango del eje y
plt.tight_layout()
plt.show()

# Segmentación por ciudad y género
# Análisis de ventas y satisfacción promedio por ciudad y género
sales_rating_by_city_gender = df.groupby(['City', 'Gender']).agg({'Total': 'sum', 'Rating': 'mean'}).reset_index()

# Visualización de los resultados
plt.figure(figsize=(12, 6))
sns.barplot(x='City', y='Total', hue='Gender', data=sales_rating_by_city_gender, palette='muted')
plt.title('Ventas por Ciudad y Género')
plt.xlabel('Ciudad')
plt.ylabel('Ventas Totales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='City', y='Rating', hue='Gender', data=sales_rating_by_city_gender, palette='muted')
plt.title('Satisfacción Promedio por Ciudad y Género')
plt.xlabel('Ciudad')
plt.ylabel('Satisfacción Promedio (Rating)')
plt.ylim(0, 10)  # Establecer el rango del eje y
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

"""## Ánalisis de correlaciones

Para el análisis de correlación eliminaremos algunas variables que no serán de utilidad para nuestros fines y ver si entre nuestras variables de interés existe alguna correlación que podamos aprovechar para la construcción de un modelo predictivo.

Creamos una copia del dataset para eliminar las variables libremente y trabajar con las numéricas:
"""

df_num = df.copy()

#Generamos variables dummy para variable género:
df_num['Male'] = df_num['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
df_num['Female'] = df_num['Gender'].apply(lambda x: 1 if x == 'Female' else 0)

df_num = df_num.drop(['Invoice ID', 'Branch','City','Customer type', 'Gender', 'Product line', 'Date', 'Time','Payment'], axis = 1)

df_num.head()

"""Realizamos un mapa de calor para ver inicialmente si hay alguna correlación"""

sns.heatmap(df_num.corr())

"""Observamos la presencia de multicolinealidad entre las variables relacionadas a las ventas, impuesto, cogs etc. Además, no detectamos alguna correlación valiosa con respecto a la variable Rating para hacer un modelo de predicción de la calificación del cliente.

"""

# Creamos un dataframe para hacer la correlación con variables categóricas y variables numéricas:
df_cor = df.copy()

df_cor = df_cor.drop(['Invoice ID', 'Branch', 'Date', 'Time'], axis = 1)

#Realizamos el análisis mixto de correlación usando Cramér's V y correlación de pearson:
# Convertir las variables categóricas en variables dummy sin eliminar la primera categoría
df_dummies = pd.get_dummies(df_cor, drop_first=False)

# Función para calcular Cramér's V
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1) * (r-1)) / (n-1))
    rcorr = r - ((r-1)**2) / (n-1)
    kcorr = k - ((k-1)**2) / (n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# Seleccionar las columnas numéricas y categóricas
num_cols = df_dummies.select_dtypes(include=['float64', 'int64']).columns
cat_cols = df_cor.select_dtypes(include=['object']).columns

# Inicializar la matriz de correlación
corr_matrix = pd.DataFrame(index=num_cols.append(cat_cols), columns=num_cols.append(cat_cols))

# Calcular la correlación de Pearson para las variables numéricas
for col1 in num_cols:
    for col2 in num_cols:
        corr_matrix.loc[col1, col2] = df_dummies[col1].corr(df_dummies[col2])

# Calcular la correlación de Cramér's V para las variables categóricas
for col1 in cat_cols:
    for col2 in cat_cols:
        corr_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])

# Calcular la correlación entre variables categóricas y numéricas
for col1 in cat_cols:
    for col2 in num_cols:
        corr_matrix.loc[col1, col2] = cramers_v(df[col1], pd.cut(df_dummies[col2], bins=3, labels=False))

for col1 in num_cols:
    for col2 in cat_cols:
        corr_matrix.loc[col1, col2] = cramers_v(df[col2], pd.cut(df_dummies[col1], bins=3, labels=False))

# Convertir los valores de la matriz de correlación a float
corr_matrix = corr_matrix.astype(float)

# Mostrar la matriz de correlación
print(corr_matrix)

# Crear un mapa de calor de la matriz de correlación
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Mapa de Correlación de Variables Mixtas')
plt.show()

"""La correlación entre las variables categóricas "Customer type" y "Gender" se destaca ligeramente con (0.021051), lo que sugiere una asociación moderada entre el tipo de cliente y el género en las ventas del supermercado.
Observamos una ligera correlación entre algunas de las variables, pero son débiles.

###PCA

Al contar con multiples variables, se aplicará un análisis de componentes principales como técnica de reducción de componentes, así eliminaremos la posible redundancia existente y podremos detectar tendencias y comportamientos.

Se llevará a cabo un Análisis de Componentes Principales (PCA), para obtener así una reducción en la dimensión del dataframe con el que se está trabajando y que actualmente presenta n características (Unit Price, Quantity, Tax 5%, Total, COGS, Gross Income y Rating). Esto nos permitirá conservar información relevante de los datos.
"""

#Asignación de valor numérico a variables categoricas City, Customer type,
#Gender, Product line y Payment

#City
citydata = df['City'] #Datos categoricos de ciudades
label_encoder = LabelEncoder()
df['numcity'] = label_encoder.fit_transform(citydata)

#Customer Type
mapping = {'Member': 0, 'Normal':1} #Asignación binaria al tipo de cliente
df['numcust'] = df['Customer type'].map(mapping)

#Gender
mapping_gen = {'Female': 0, 'Male': 1} #Asignación binaria al género
df['numgen'] = df['Gender'].map(mapping_gen)

#Product Line
pline = df['Product line'] #Datos categoricos de la linea del producto
df['numpline']= label_encoder.fit_transform(pline)

#Payment
pay = df['Payment'] #Datos categoricos del método de pago
df['numpayment']= label_encoder.fit_transform(pay)

print(df)

#Eliminaremos variables categoricas para así unicamente trabajar con las variables numericas
#Eliminaremos también variables que presentaron redundancia en la información como Tax o cogs
newdata=df.drop(columns=['Invoice ID', 'Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Payment', 'Date', 'Time', 'cogs', 'Tax 5%'])
print(newdata.head(6))

#newdata=df.drop(columns=['Grupo de compra'])

print(newdata.head(6))

#Se usará la transformación Yeo-Johnson debido a los valores 0 presentados en las columnas
from sklearn.preprocessing import PowerTransformer

#Transformación Yeo-Johnson
pt = PowerTransformer(method='yeo-johnson')
data_transformed = pt.fit_transform(newdata.select_dtypes(include=[np.number]))
data_transformed = pd.DataFrame(data_transformed, columns=newdata.select_dtypes(include=[np.number]).columns)

#Se presenta la tabla de correlacion entre las variables
data_transformed.corr()

#Determinación del número de componentes con gráfica de codo
from sklearn.decomposition import PCA

# Aplicación de PCA
pca = PCA().fit(data_transformed)

# Gráfica de codo con puntos marcados
plt.figure(figsize=(8, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')  # Añade marker='o' para poner puntos
plt.xlabel('Número de componentes')
plt.ylabel('Varianza acumulativa explicada')
plt.title('Gráfica de codo para PCA')
plt.grid(True)  # Opcional, añade una rejilla para mejor visualización
plt.show()

"""Bajo este análisis tendríamos que aplicar 6 componentes principales."""

#Número de componentes según Regla de Kaiser

num_components_kaiser = (pca.explained_variance_ > 1).sum()
print("Número de componentes según el criterio de Kaiser:", num_components_kaiser)

"""Consideraremos el resultado bajo la Regla de Kaiser, pues puede ser matemáticamente más preciso que determinarlo gráficamente"""

# Aplicación de PCA con el número óptimo de componentes
pca_opt = PCA(n_components=5)
data_pca = pca_opt.fit_transform(data_transformed)

# Cargar las cargas (loadings) en un DataFrame de pandas para mejor visualización
loadings = pd.DataFrame(pca_opt.components_.T, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'], index=data_transformed.columns)

# Mostrar las cargas i.e, coeficientes de los componentes principales
print(loadings)

"""Los valores más relevantes en PC1 son de valores **negativos**. Siendo estos el total de las **compras** y el **ingreso bruto**, seguido de la **cantidad de productos adquiridos** y el **precio unitario**.
El valor positivo, aunque no muy significativo pero más relevante es el presentado en la variable de **género**.

Los hombres están positivamente asociados con el PC1.

Para el PC2, observamos que esta positivamente afectado por la cantidad de **productos comprados** y en un contraste (al mismo nivel), se encuentra la ciudad de las **ubicaciones de los supermercados**.
Un posible analisis de esta relación podría decirnos que el supermercado de la ciudad Yangon tiende a vender una menor cantidad de productos, en comparación con Mandalay.

PC3 está fuertemente determinado por el **rating** de los clientes, seguido de las **ubicaciones de los supermercados**. Podríamos concluir que los clientes en la ciudad Yangon tienen mejores ratings.

Por otro lado se observa un contraste entre el tipo de cliente y el tipo de pago. Esto nos podría indicar que los clientes sin membresía suelen pagar con efectivo, mientras que los miembros optan por una e-wallet o tarjetas de crédito.

También está se observa un valor de -0.5274 en la línea de los productos, y que este valor contrasta del de rating, lo cual podría expresarnos que los clientes con mejor rating consumen en departamentos como electronicos, moda y posiblemente comida y bebidas.

PC5 está principalmente afectado por el **precio** y en un mismo nivel contrasta la **forma de pago**. Esto podría indicar que los productos más costosos, suelen ser pagados en efectivo y por ende, productos menos costosos son pagados con tarjetas de crédito y/o e-wallet.

Por otro lado, los clientes miembros están mayormente asociados a este componente.

### Clustering

Se realizará el análisis de clústeres a partir de los resultados obtenidos en el análisis PCA. Esto nos permitirá la identificación de grupos o patrones que podríamos no observar con la base de datos original.

Además, los clústeres pueden ser de utilidad para confirmar la segmentación de clientes analizada previamente.
"""

from sklearn.preprocessing import StandardScaler
#Número de clústeres por método de codo
#Normalización de los datos
scaler = StandardScaler()
data_scaled = scaler.fit_transform(newdata)

#Implementación del método
from sklearn.cluster import KMeans
wcss = []
k_values = range(1, 11)

for i in k_values:
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

# Imprimir los valores de WCSS para cada número de clústers
for i in range(len(wcss)):
    print("WCSS for", i+1, "clusters:", wcss[i])

# Graficamos WCSS para cada número de clústers
plt.plot(k_values, wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#Determinar el número óptimo de clústeres usando análisis de silueta
from sklearn.metrics import silhouette_score

silhouette_scores = []
k_values2 = range(2, 11)

for n_clusters in k_values2:
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(data_scaled)

    silhouette_avg = silhouette_score(data_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"Para {n_clusters} clústeres, la puntuación de silueta promedio es {silhouette_avg}")

# Graficar las puntuaciones de silueta
plt.plot(k_values2, silhouette_scores, marker='o')
plt.xlabel('Número de clústeres')
plt.ylabel('Puntuación de silueta promedio')
plt.show()

# Escoger el número oṕtimo de clústeres
optimal_clusters = k_values2[np.argmax(silhouette_scores)]
print(f"El número óptimo de clústeres es {optimal_clusters}")

"""Ambos metodos nos indican que el número óptimo de clústeres es de 2.
Al desear la visualización de los clústeres en función de todas las características.
"""

#Clusters
import pandas as pd
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0)
clusters = kmeans.fit_predict(data_pca)
labels = kmeans.labels_

# Gráfico de dispersión de los clusters identificados por KMeans
plt.figure(figsize=(8, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.5)
plt.title('Visualización de Clusters de Supermercado')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.colorbar(label='Cluster')
plt.show()

"""PC1 y PC2 son los componentes con mayor relevancia, por lo que se analisa el cluster entre estos dos.
Es fácil notar que no es del todo clara la separación entre estos, más allá de que están divididos justo por la mitad, esto puede ser causado por la redundancia y similitud presentada en la base de datos.

# **ANALISIS DE SERIE TEMPORAL**

Cargando los datos
"""

#Cargamos el dataset
datos = pd.read_csv("sales_market_fecha_unificado.csv")
print("Number of datapoints:", len(datos))
datos[:]

print(datos.dtypes['Date'])
df = datos[['Date','Suma_Gross_Income']]
print(df)

df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

#Agrupar por Date y sumar valores de gross income
df_agrupado = df.groupby('Date')['Suma_Gross_Income'].sum().reset_index()

print(df_agrupado.columns.tolist())

print(df_agrupado)

# Convertir la columna 'Date' a formato datetime
df_agrupado['Date'] = pd.to_datetime(df_agrupado['Date'], format='%Y-%m-%d')

# Establecer 'Date' como índice
df_agrupado.set_index('Date', inplace=True)

print(df_agrupado.head())

plt.plot(df_agrupado.index, df_agrupado['Suma_Gross_Income'])
plt.xlabel('Date')
plt.ylabel('Suma_Gross_Income')
plt.xticks(rotation = 45)
plt.title('SERIE DE TIEMPO GANANCIAS')
plt.show()

"""# Se proceden a realizar pruebas de **Stationarity**


Graficamente no se observa tendencia alguna.
"""

print('Media:', statistics.mean(df_agrupado['Suma_Gross_Income']))
print('Varianza:', statistics.variance(df_agrupado['Suma_Gross_Income']))
print('DS:', statistics.stdev(df_agrupado['Suma_Gross_Income']))

resumen_estadistico = df_agrupado['Suma_Gross_Income'].describe()
print(resumen_estadistico)

print('Rango: ', resumen_estadistico[7]-resumen_estadistico[3])
prueba_adf = adfuller(df_agrupado)

print('ADF Statistic:', prueba_adf[0])
print('p-value:', prueba_adf[1])

"""Dado que el p-valor es menor que la significancia  se rechaza H0, la serie es estacionaria

# **SEASONALITY**
"""

#Se procede a hacer uso de ACF , PACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Assuming 'data' is the time series data
plot_acf(df_agrupado)
plot_pacf(df_agrupado)
plt.show()

from statsmodels.stats.diagnostic import acorr_ljungbox

test_result = acorr_ljungbox(df_agrupado, lags=[20], return_df=True)

print(test_result)

"""1) La serie es estacionaria
2) No hay existencia de correlación
3) No pasa Dicky fuller prueba
4) Ljung Box: No se rechaza H0: No Autocorrelación

Por lo cual es ruido blanco y por lo cual las ganancias no predecidas


A continuación probaremos agrupar las ganacias de manera semanal en lugar de que sea diario.
"""

# Supongamos que ya tienes el dataframe `df`
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Configura la columna 'Date' como índice
df.set_index('Date', inplace=True)

# Agrupar por semanas y sumar valores de gross income
df_agrupado_semanal = df['Suma_Gross_Income'].resample('W').sum().reset_index()

# Mostrar el dataframe resultante
print(df_agrupado_semanal)

# Convertir la columna 'Date' a formato datetime
df_agrupado_semanal['Date'] = pd.to_datetime(df_agrupado_semanal['Date'], format='%Y-%m-%d')


# Establecer 'Date' como índice
df_agrupado_semanal.set_index('Date', inplace=True)

print(df_agrupado_semanal.head())

plt.plot(df_agrupado_semanal.index, df_agrupado_semanal['Suma_Gross_Income'])
plt.xlabel('Date')
plt.ylabel('Suma_Gross_Income')
plt.xticks(rotation = 45)
plt.title('SERIE DE TIEMPO GANANCIAS SEMANAL')
plt.show()

from statsmodels.tsa.stattools import adfuller
import statistics
print('Media:', statistics.mean(df_agrupado_semanal['Suma_Gross_Income']))
print('Varianza:', statistics.variance(df_agrupado_semanal['Suma_Gross_Income']))
print('DS:', statistics.stdev(df_agrupado_semanal['Suma_Gross_Income']))

resumen_estadistico = df_agrupado_semanal['Suma_Gross_Income'].describe()
print(resumen_estadistico)

print('Rango: ', resumen_estadistico[7]-resumen_estadistico[3])
prueba_adf = adfuller(df_agrupado_semanal)

#Probar si la serie es estacionaria
print('ADF Statistic:', prueba_adf[0])
print('p-value:', prueba_adf[1])

"""Se rechaza H0:  No es estacionaria.

Por lo cual la serie temporal es estacionaria
"""

#Se procede a hacer uso de ACF , PACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Assuming 'data' is the time series data
plot_acf(df_agrupado_semanal)
plot_pacf(df_agrupado_semanal)
plt.show()

# El parámetro lags determina el número de retardos a considerar
test_result = acorr_ljungbox(df_agrupado_semanal, lags=[6], return_df=True)

print(test_result)

"""Sigue siendo ruido Blanco, no pasa pruebas y no existe una correlación significativa.

Se necesita más información, pues el registro historico de la tienda no es suficiente, es decir necesita pasar más tiempo.

# **MODELOS PREDICTIVOS Y DE CLASIFICACIÓN**
"""

df.drop(columns=["Invoice ID"], inplace=True)


# Crear características adicionales (binarias) ya que son variables cualitativas
columns_to_encode = ['Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Payment', 'Date', 'Time']
df_encoded = pd.get_dummies(df, columns=columns_to_encode, prefix=columns_to_encode)

print (df_encoded)

# Dividimos los datos en conjuntos de entrenamiento y prueba
X = df_encoded.drop(columns=["Total"])
y = df_encoded['Total']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo de Árbol de Decisión
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Métricas
print("Árbol de Decisión:")
print("Error cuadrático medio (MSE):", mean_squared_error(y_test, y_pred_dt))
print("Coeficiente de determinación (R2):", r2_score(y_test, y_pred_dt))

# Modelo Random Forest
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Métricas
print("\nRandom Forest:")
print("Error cuadrático medio (MSE):", mean_squared_error(y_test, y_pred_rf))
print("Coeficiente de determinación (R2):", r2_score(y_test, y_pred_rf))

# Validación cruzada
cv_scores_dt = cross_val_score(dt_model, X, y, cv=5)
cv_scores_rf = cross_val_score(rf_model, X, y, cv=5)
print("\nValidación cruzada (Árbol de Decisión):", cv_scores_dt.mean())
print("Validación cruzada (Random Forest):", cv_scores_rf.mean())


# Visualizamos el árbol de decisión
plt.figure(figsize=(12, 8))
plot_tree(dt_model, feature_names=X.columns, filled=True)
plt.show()

# Extraemos los árboles individuales del bosque aleatorio para poder visualizarlos
individual_trees = rf_model.estimators_

# Visualizamos los primeros 5 árboles
for i, tree in enumerate(individual_trees[:5]):
    plt.figure(figsize=(12, 8))
    plot_tree(tree, feature_names=X.columns, filled=True)
    plt.title(f"Árbol {i+1}")
    plt.show()

"""1. Arbol de decisión:
El Error cuadrático medio mide la diferencia promedio entre los valores reales y las predicciones del modelo. En este caso, el valor 3.88 indica que el modelo tiene un buen ajuste a los datos.
El Coeficiente de determinación mide la varianza en la variable objetivo. En nuestro modelo, el 0.99 indica que se explica casi completamente la variabilidad de los datos.

2. Random forest:
El Error cuadrático medio de 1.65 sugiere un buen ajuste a los datos, mucho mejor que el Árbol de decisión.
El coeficiente de determinación, también explica casi toda la variabilidad de los datos.

3. Validación cruzada:
Proporciona una idea de cómo se desempeña el modelo para otros datos. Para el árbol de decisión y el random forest, los valores altos sugieren que ambos modelos se desempeñan bien.
Por tanto, ambos son modelos efectivos para los datos.

# **SIMULACIÓN DE MONTE CARLO**
"""

# Visualización rápida de los datos
df['Total'].plot(figsize=(10, 5))
plt.title("Total")
plt.show()

# Dividimos los datos en conjuntos de entrenamiento y prueba
X = df_encoded.drop(columns=["Total"])
y = df_encoded['Total']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalización de los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definir los modelos base
modelos = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "LinearRegression": LinearRegression(),
    "SVR": SVR(),
    "KNeighbors": KNeighborsRegressor()
}

# Entrenar los modelos base
for name, model in modelos.items():
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    print(f"{name} - RMSE: {rmse:.4f}, R^2: {r2:.4f}")

# Crear el meta-modelo para el stacking
estimators = [(name, model) for name, model in modelos.items()]
final_estimator = GradientBoostingRegressor(random_state=42)
stacking_model = StackingRegressor(estimators=estimators, final_estimator=final_estimator)

# Entrenar el modelo de stacking
stacking_model.fit(X_train_scaled, y_train)

# Predicción y evaluación del modelo de stacking
pred_stacking = stacking_model.predict(X_test_scaled)
rmse_stacking = np.sqrt(mean_squared_error(y_test, pred_stacking))
r2_stacking = r2_score(y_test, pred_stacking)
print(f"Stacking Regressor - RMSE: {rmse_stacking:.4f}, R^2: {r2_stacking:.4f}")

# Función para realizar simulaciones de Monte Carlo
def monte_carlo_simulation_adjusted(model, initial_features, n_simulations=1000, n_days=30):
    simulations = []
    for _ in range(n_simulations):
        preds = []
        # Introducir un pequeño ruido a las características iniciales para cada simulación
        features = initial_features + np.random.normal(0, 1.0, initial_features.shape)
        for i in range(n_days):
            pred = model.predict(features.reshape(1, -1))[0]
            preds.append(pred)
            # Incrementar el nivel de ruido para simular mayor variabilidad del mercado
            noise = np.random.normal(0, 1.0)
            # Actualizar las características para la siguiente predicción
            features = np.roll(features, -1)
            features[-1] = pred + noise
        simulations.append(preds)
    return np.array(simulations)

# Realizar simulaciones de Monte Carlo con las características ajustadas
simulations_adjusted = monte_carlo_simulation_adjusted(stacking_model, X_test_scaled[-1], n_simulations=200, n_days=30)

# Mostrar las primeras 20 simulaciones
print(simulations_adjusted[:20, :])


# Visualizar las simulaciones
plt.figure(figsize=(10, 5))
for sim in simulations_adjusted:
    plt.plot(sim, color='grey', alpha=0.1)
plt.title("Simulación de Monte Carlo del Precio Total")
plt.show()

# Calcular estadísticas de las simulaciones
mean_sim = np.mean(simulations_adjusted, axis=0)
std_sim = np.std(simulations_adjusted, axis=0)

print(f"Media de las simulaciones: {mean_sim[-1]:.2f}")
print(f"Desviación estándar de las simulaciones: {std_sim[-1]:.5f}")

"""1. **Random Forest**:
   - Error Cuadrático Medio: El 1.2706 significa que, en promedio, las predicciones del modelo Random Forest difieren en aproximadamente 1.27 unidades de los valores reales. Indica un mejor ajuste del modelo.
   - Coeficiente de Determinación (R²): El valor de 1.0000 sugiere que el 100% de la variabilidad en la variable de respuesta se explica por el modelo de Random Forest. En otras palabras, el modelo se ajusta perfectamente a los datos.

2. **Gradient Boosting**:
   - Error Cuadrático Medio: El 2.2975 indica una buena precisión en las predicciones.
   - Coeficiente de Determinación (R²): El 0.9999. Esto significa que el 99.99% de la variabilidad también se explica por el modelo de Gradient Boosting.

3. **Regresión Lineal**:
   - Error Cuadrático Medio: El 4.2222 sugiere que la regresión lineal tiene una precisión inferior en comparación a los anteriores.
   - Coeficiente de Determinación (R²): El 0.9997 no es tan perfecto como los modelos anteriores.

4. **SVR (Support Vector Regression)**:
   - Error Cuadrático Medio: El 267.1046 es significativamente alto en comparación con los otros modelos.
   - Coeficiente de Determinación (R²): El -0.0966 sugiere que el modelo SVR no se ajusta bien a los datos y tiene un rendimiento pobre.

5. **KNeighbors (K-Vecinos más Cercanos)**:
   - Error Cuadrático Medio: El 242.0503 también es alto.
   - Coeficiente de Determinación (R²): El 0.0995 indica que el modelo tiene una capacidad limitada para explicar la variabilidad.

6. **Stacking Regressor**:
   - Error Cuadrático Medio: El 2.8354 es similar al Gradient Boosting.
   - Coeficiente de Determinación (R²): El 0.999 es un buen resultado.

  **Conclusión:**
- **Random Forest** parece ser el mejor modelo en términos de precisión y ajuste.
- **SVR** y **KNeighbors** tienen un rendimiento deficiente.
- **Gradient Boosting** y **Stacking Regressor** son buenas opciones.


  **Monte Carlo:**
- La media de las simulaciones es 305.80 La desviación estándar (109.80972) muestra la variabilidad entre las simulaciones.

Resultados de análisis exploratorio y relaciones a detalle:
  



1.   El ingreso bruto promedio por transacción es de
aproximadamente 15.38 unidades monetarias.
2.   Los precios se distribuyen con media de 55.67 y desviación estándar de 26.5. El precio mínimo que se maneja es de 10.08 y el más alto es de 99.96.
3.  La línea con precio promedio mas alto es la de Fashion accesories con 56.01.

4. Encantidad de productos los clientes suelen comprar entre 9 y 10 artículos.
  
5. La mayor concentración de ventas se tiene en Naypytaw con un 34.24%

6. Las mujeres compran mas productos de "accesorios de moda", mientras que los hombres compran mayormente productos de "salud y bbelleza".
7. El departamento con mayor cantidad de ventas es el de Home and lifestyle.
"""
