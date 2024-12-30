# Análisis de Supermercado
Análisis EDA, modelado, clustering, correlación, Monte Carlo de datos historicos de una tienda de supermercado.

# **Universidad Autónoma del Estado de México**
## **Facultad de Economía**
### *Licenciatura en Actuaría*
#### *Modelos de Simulación*
*Proyecto de Evaluación: Ventas en un Supermercado*

### Elaborado por:
*   Espinosa Arias Lizbeth
*   Gallegos Díaz Ximena
*   Tapia González Carmen
*   Vilchis Sotelo María Miel

# **INTRODUCCIÓN**
En el mundo actual, la toma de decisiones informada y estratégica es esencial para el éxito. Para lograrlo, las organizaciones confían en herramientas avanzadas que les permitan anticipar tendencias, comprender patrones y optimizar recursos.
En este proyecto exploraremos un conjunto de datos históricos de ventas en supermercados (Supermarket Ratings). El dataset contiene registros de ventas de la empresa de supermercados, en tres sucursales diferentes, durante tres meses.
Del mismo modo, evaluaremos diferentes modelos para comprender mejor los datos y tener una mejor perspectiva para la toma de decisiones.

# **Supermarket Dataset**

Este conjunto de datos consta de las ventas históricas de la empresa de supermercados que se ha registrado en 3 sucursales diferentes durante 3 meses. Las variables son:

*   Invoice ID: número de identificación de factura de comprobante de venta generado por computadora.

* Branch: Sucursal del supermercado (se encuentran disponibles 3 sucursales identificadas por A, B y C).

* City: Ubicación de los supermercados.

* Customer type: Tipo de clientes, registrados por Membresia para clientes con tarjeta de miembro y Normal para sin membresia.

* Gender: Tipo de género del cliente.

* Product line: Grupos generales de categorización de artículos: accesorios electrónicos, accesorios de moda, alimentos y bebidas, salud y belleza, hogar y estilo de vida, deportes y viajes.

* Unit price: Precio de cada producto en $.

* Quantity: Número de productos adquiridos por el cliente.

* Tax: 5% de impuesto para la compra del cliente.

* Total: Precio total con impuestos incluidos.

* Date: Fecha de compra (Registro disponible desde enero de 2019 a marzo de 2019).

* Time: Hora de compra (de 10 a 21 horas).

* Payment: Forma de pago utilizado por el cliente para la compra (hay 3 métodos disponibles: efectivo, tarjeta de crédito y e-wallet).

* COGS: Costo de los bienes vendidos.

* Gross margin percentage: Porcentaje de margen bruto.

* Gross income: Ingresos brutos.

* Rating: Calificación de estratificación del cliente sobre su experiencia de compra general (en una escala del 1 al 10).

**OBJETIVOS**
1. *Optimización de Ingresos:*
Utilizaremos técnicas de modelado para predecir las ventas futuras y analizaremos patrones de compra, segmentos de clientes y precios de productos para maximizar los ingresos. Esto podría incluir promociones, descuentos,  estrategias de fidelización, cambios en horarios de apertura.

2. *Mejorar de la Experiencia del Cliente:*
Analizaremos la satisfacción del cliente para proponer mejoras que ayuden a brindar una mejor experiencia a los clientes.

**METODOLOGÍA**
1.	Análisis Exploratorio de Datos (EDA):
Es un proceso en el que se investigan conjuntos de datos para descubrir patrones, detectar anomalías, probar hipótesis y verificar suposiciones. Ayuda a comprender las variables del conjunto de datos y las relaciones entre ellas.
En este proyecto, exploraremos la distribución de los datos, identificaremos patrones, relaciones y tendencias. Además, evaluaremos la calidad de los datos y realizaremos una limpieza de datos no significativos.

2.	Evaluar modelos:
Los modelos son representaciones matemáticas de sistemas o procesos en los que se simulan eventos a lo largo del tiempo. Estos modelos se basan en datos históricos y parámetros que describen el comportamiento del sistema. Permiten proyectar cómo podría evolucionar el sistema en el futuro. La simulación se utiliza para ensayar alternativas y elegir la mejor estrategia antes de implementar cambios en el sistema real.
De este modo, construiremos modelos como árboles de decisión y bosques aleatorios, y compararemos su desempeño con métricas importantes; con el objetivo de seleccionar el modelo más adecuado.

3. Clustering:
Las técnicas de clustering son importantes para agrupar datos y comprender las características de los grupos. Esto será de ayuda para identificar patrones de comportamiento a través de métodos como el del codo o el coeficiente de silueta.

4. Correlación:
Consideraremos las relaciones entre variables para obtener una visión más amplia de cómo estas pueden afectar las decisiones comerciales.
