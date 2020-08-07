# Evaluación actividad Iris en R
# Alumno: Daniel García Loyola
# Profesor: Israel Naranjo Retamal
# Ramo: Minería de datos (MDY7101)
# Sección: 002D
#source("D:/r/eva03_DanielGarcia.R")

# Cargando dataset de iris
data = read.csv('D:/r/iris.csv')
object.size(data)


# Cambiando nombres de columnas y visualizando resultados
names(data) = c('ID', 'Largo_Sepal', 'Ancho_Sepal',
  'Largo_Petalo', 'Ancho_Petalo', 'Especies')
names(data)


# Explorando datos
head(data, 5)
tail(data, 5)


# Verificando contenido del dataframe
str(data)
summary(data)


#Contando valores nulos
sapply(data, function(x) sum(is.na(x)))

# Tratamiento de lementos nulos
# Si hay nulos en los valores numericos de iris
# le asignara el promedio para no generar problemas
# en los modelos
data$Largo_Petalo = ifelse(is.na(data$Largo_Petalo),
  ave(data$Largo_Petalo, FUN = function(x) mean(x, na.rm = TRUE)),
  data$Largo_Petalo)

data$Largo_Sepal = ifelse(is.na(data$Largo_Sepal),
  ave(data$Largo_Sepal, FUN = function(x) mean(x, na.rm = TRUE)),
  data$Largo_Sepal)

data$Ancho_Petalo = ifelse(is.na(data$Ancho_Petalo),
  ave(data$Largo_Petalo, FUN = function(x) mean(x, na.rm = TRUE)),
  data$Ancho_Petalo)

data$Ancho_Sepal = ifelse(is.na(data$Ancho_Sepal),
  ave(data$Ancho_Sepal, FUN = function(x) mean(x, na.rm = TRUE)),
  data$Ancho_Sepal)


# Visualizando  las variables
x11()
pairs(data[,1:4], main='Distribucion de las variables del dataframe Iris')


# Visualizando las especies en boxplot
par(mar=c(7,5,1,1))
boxplot(data[2:5],las=2)


# Visualizando grafico de distribucion y densidad de iris
# install.packages('beanplot')
library(beanplot)
beanplot(data[2:5], main ="Tipos Iris", col=c('#ff8080','#0000FF','#0000FF','#FF00FF'), border='#000000')


# Visualizando diferencias entre especies y anchos de sepalo y petalos
par(mfrow=c(1,2))
boxplot(data$Ancho_Sepal ~ data$Especies, col = "gray", main = "Especies de iris\nsegún la anchura del sépalo")
boxplot(data$Ancho_Petalo ~ data$Especies, col = "gray", main = "Especies de iris\nsegún la anchura del petalo")


# Verificando correlacion de variables
par()
corr <- cor(data[,2:5])
round(corr,3)


# Comparando variables con graficos de lineas paralelas
# install.packages('')
library(MASS)
par(xpd=TRUE)
parcoord(data[,2:5], col=c(1,2,3),var.label=TRUE, oma=c(4,4,6,12))
legend(0.85,0.6, as.vector(unique(iris$Species)), fill=c(1,2,3))




#=======================================
# Preparando datos para el modelo
#

# Procesando nombre de especie como variable categorica
data$Especies <- factor(data$Especies)
str(data$Especies)


# ***********************************
# Modelo: Desicion tree

# instalar packages:
# install.packages('e1071')
# install.packages('rpart.plot')
library('e1071')
library(rpart)
library(rpart.plot)
library(caTools)

# obteniendo el conjunto de entrenamiento y de test
set.seed(12021)
split_dt <- sample.split(data$Especies, SplitRatio = 0.80)
X_train_dt <- subset(data, split_dt == TRUE)
X_test_dt <- subset(data, split_dt == FALSE)

# Preparando el modelo
modelo_dt <- rpart(formula = Especies ~ ., data = X_train_dt)

# Realizando prediccion con el conjunto de test, obteniendo matriz de confusion y mostrando resultados
y_predict_dt <- predict(modelo_dt, newdata = X_test_dt, type = "class")
m_confusion_dt <- table(X_test_dt[, 5], y_predict_dt)
summary(m_confusion_dt)

# Visualizando las probabilidades
table(y_predict_dt, X_test_dt[, 5])

# Verificando la presicion del modelo
presicion_dt <- sum(diag(m_confusion_dt))/sum(m_confusion_dt)
presicion_dt

# Graficando el modelo
plot(modelo_dt)
text(modelo_dt)
prp(modelo_dt, type = 3, extra = 2, split.font = 3, box.col = c("blue","green", 'red')[modelo_dt$frame$yval])
summary(modelo_dt)

# Probando el modelo, se obtiene un registro al azar y se comprueba la prediccion
set.seed(8475)
prueba_dt <- data[sample(nrow(data), 1), ]
prueba_dt
set.seed(12021)
y_predict_dt <- predict(modelo_dt, newdata = prueba_dt, type = "class")
y_predict_dt

# Visualizando las probabilidades despues de predecir con el registro al azar
table(y_predict_dt, prueba_dt[, 5])

# Verificando la presicion del modelo luego de la prediccion al azar
m_confusion_dt <- table(prueba_dt[, 5], y_predict_dt)
presicion_dt <- sum(diag(m_confusion_dt))/sum(m_confusion_dt)
presicion_dt




# ***********************************
# Modelo: Tree clasifier
# install.packages('tree')
library(tree)

# obteniendo el conjunto de entrenamiento y de test
set.seed(12321)
split_tc <- sample.split(data$Especies, SplitRatio = 0.80)
X_train_tc <- subset(data, split_tc == TRUE)
X_test_tc <- subset(data, split_tc == FALSE)

# Preparando el modelo y visualizando resultados
modelo_tc <- tree(Especies ~ ., data=X_train_tc)

# Realizando prediccion con datos de prueba, obteniendo matriz de confusion y mostrando resultados
y_predict_tc <- predict(modelo_tc, newdata=X_test_tc, type="class")
m_confusion_tc <- table(X_test_tc[, 5], y_predict_tc)
summary(m_confusion_tc)

# Visualizando las probabilidades
table(y_predict_tc, X_test_tc[, 5])

# Verificando la presicion del modelo
presicion_tc <- sum(diag(m_confusion_tc))/sum(m_confusion_tc)
presicion_tc

# Graficando el modelo
plot(modelo_tc)
text(modelo_tc)
summary(modelo_tc)

# Probando el modelo, se obtiene un registro al azar y se comprueba la prediccion
set.seed(8475)
prueba_tc <- data[sample(nrow(data), 1), ]
prueba_tc
set.seed(12321)
y_predict_tc <- predict(modelo_tc, newdata = prueba_tc, type = "class")
y_predict_tc

Visualizando las probabilidades despues del registro aleatorio
table(y_predict_tc, prueba_tc[, 5])

# Verificando la presicion del modelo posterior a las pruebas del registro aleatorio
m_confusion_tc <- table(prueba_tc[, 5], y_predict_tc)
presicion_tc <- sum(diag(m_confusion_tc))/sum(m_confusion_tc)
presicion_tc




# ***********************************
# Modelo: SVM
# install.packages('caTools')
library(caTools)

# Preparando el conjunto de datos de entrenamiento y de test
set.seed(32123)
split_svm <- sample.split(data$Especies, SplitRatio = 0.80)
X_train_svm <- subset(data, split_svm == TRUE)
X_test_svm <- subset(data, split_svm == FALSE)

# Escalando variables, Preparando el modelo y visualizando resultados
X_train_svm[-6] <- scale(X_train_svm[-6])
X_test_svm[-6] <- scale(X_test_svm[-6])
modelo_svm <- svm(Especies ~ ., data=X_train_svm)

# Realizando prediccion con datos de prueba, obteniendo matriz de confusion y mostrando resultados
y_predict_svm <- predict(modelo_svm, newdata = X_test_svm[-6])
m_confusion_svm <- table(X_test_svm[, 6], y_predict_svm)
summary(m_confusion_svm)

# Visualizando las probabilidades
table(y_predict_svm, X_test_svm[, 6])

# Verificando la presicion del modelo
presicion_svm <- sum(diag(m_confusion_svm))/sum(m_confusion_svm)
presicion_tc

# Mostrando resultados del modelo
summary(modelo_svm)

# Probando el modelo, se obtiene un registro al azar y se comprueba la prediccion
set.seed(8475)
prueba_svm <- data[sample(nrow(data), 1), ]
prueba_svm
set.seed(32123)
y_predict_svm <- predict(modelo_svm, newdata = prueba_svm, type = "class")
y_predict_svm

Visualizando las probabilidades despues de predecir con el registro aleatorio
table(y_predict_svm, prueba_svm[, 5])

# Verificando la presicion del modelo posterior a las pruebas con el registro al azar
m_confusion_svm <- table(prueba_svm[, 6], y_predict_svm)
presicion_svm <- sum(diag(m_confusion_svm))/sum(m_confusion_svm)
presicion_svm




# ***********************************
# Modelo:
# install.packages('ggplot2')
# install.packages('dplyr')
# install.packages('cluster')
# install.packages('tidyr')
library(tidyr)
library(ggplot2)
# library(dplyr)
library(cluster)
library(RColorBrewer)
theme_set(theme_bw(base_size=12)) 

# Data a trabajar con kMeans
str(data)
head(data, 10)
tail(data, 10)

# Aplicando el metodo del codo para buscar el numero optimo de nodos y muestro el resultado
X = data[, 2:5]
kmean_withinss <- function(k) {
    cluster <- kmeans(X, k)
    return (cluster$tot.withinss)
}
max_k <- 20
wss <- sapply(2:max_k, kmean_withinss)
elbow <-data.frame(2:max_k, wss)
ggplot(elbow, aes(x = X2.max_k, y = wss)) + geom_point() + geom_line() +
  ggtitle ('Metodo del codo para ubicar (k) nodos') +
  labs(x = 'Número de clusters (k)',y = 'WCSS(k)') +
  scale_x_continuous(breaks = seq(1, 20, by = 1))

# Aplicando el metodo kMeans con k optimo
num_k = 7
modelo_km <- kmeans(X, num_k, iter.max = 300, nstart = 10)

# Visualizando los cluster generados
clusplot(X, modelo_km$cluster, lines = 0, shade = TRUE,
    color = TRUE, labels = 2, plotchar = FALSE,
    span = TRUE,main = "Clustering de Iris",
    xlab = 'Caracteristicas', ylab = 'Caracteristicas')

# Mostrando los centroides
modelo_km$centers

# Mostrando el tamaño de los cluesters
modelo_km$size

cluster <- c(1: num_k)
centroides <- data.frame(cluster, modelo_km$centers)
centroides <- gather(centroides, features, values, 2:4)
paleta_colores <-colorRampPalette(rev(brewer.pal(10, 'RdYlGn')),space='Lab')
ggplot(data = centroides, aes(x = features, y = cluster, fill = values)) +
  ggtitle ('Mapa de calor del contenido de los clusters') +
  scale_y_continuous(breaks = seq(1, 7, by = 1)) +
  geom_tile() +coord_equal() +
  scale_fill_gradientn(colours = paleta_colores(90)) +
  theme_classic()

# Visualizando el modelo
str(centroides)


# Visualizando el cluester
str(modelo_km)

# Verificando el cluster
table(data$Especies, modelo_km$cluster)


