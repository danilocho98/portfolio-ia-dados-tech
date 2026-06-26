#Exercício 1
#cria um vetor com os anos extraídos da coluna de datas do dataframe D.
D <- read.csv("epData.csv")

criaVetorAno <- function(D) {
  vetorAno <- c()
  for(i in 1:nrow(D)){
    dataSeparada <- strsplit(as.character(D$date[i]), "-")
    vetorData <- unlist(dataSeparada)
    vetorAno <- c(vetorAno, as.integer(vetorData[1]))
  }
  return(vetorAno)
}
resultado <- criaVetorAno(D)
print(resultado)

#Exercício 2
#retorna o preço do Big Mac nos EUA para um determinado ano A.
#filtra os dados do dataframe D que correspondem aos EUA e ao ano A
#retorna o primeiro preço local encontrado para este ano
BigMacUSPrice <- function(D, A) {
  filtro <- D$isoA3 == "USA" & as.integer(substr(D$date, 1, 4)) == A
  dadosFiltrados <- D[filtro, ]
  return(dadosFiltrados$localPrice[1])
}

A <- as.integer(readline("Insira um ano: "))
valorBigMacUS <- BigMacUSPrice(D, A)
print(valorBigMacUS)

#Exercício 3
#calcula a média de uma determinada coluna C para um país específico P.
#filtra o dataframe D para incluir apenas as linhas correspondentes ao país P
#soma os valores da coluna C e divide pela contagem para obter a média
mediaColunaPais <- function(D, C, P) {
  dadosFiltrados <- D[D$isoA3 == P, ]
  soma <- 0
  contador <- 0
  for (i in 1:nrow(dadosFiltrados)) {
    soma <- soma + dadosFiltrados[i, C]
    contador <- contador + 1
  }
  media <- soma / contador
  return(media)
}

C <- as.integer(readline(prompt="Insira o número da coluna que deseja calcular a média: "))
P <- as.character(readline(prompt="Insira o código do país que deseja calcular a média: "))
media <- mediaColunaPais(D, C, P)
print(media)

#Exercício 4
#calcula a variância de uma determinada coluna C para um país específico P
#Mesma lógica da 3, só muda a fórmula

varColunaPais <- function(D, C, P) {
  dadosFiltrados <- D[D$isoA3 == P, ]
  soma <- 0
  somaQuadrados <- 0
  contador <- 0
  for (i in 1:nrow(dadosFiltrados)) {
    valor <- as.numeric(dadosFiltrados[i, C])
    soma <- soma + valor
    somaQuadrados <- somaQuadrados + valor^2
    contador <- contador + 1
  }
  media <- soma / contador
  variancia <- (somaQuadrados - contador * media^2) / (contador - 1)
  return(variancia)
}

C <- as.integer(readline(prompt="Insira o número da coluna que deseja calcular a variância: "))
P <- as.character(readline(prompt="Insira o código do país que deseja calcular a variância: "))
variancia <- varColunaPais(D, C, P)
print(variancia)


#Exercício 5
#cria um vetor com o índice Big Mac para todas as linhas do dataframe D
#para cada linha do dataframe, a função encontra o preço do Big Mac nos EUA para o mesmo ano e, em seguida, divide o preço local do Big Mac pelo preço do Big Mac nos EUA
#Se o preço do Big Mac nos EUA for NA ou zero, a função atribui NA ao índice Big Mac para essa linha
criaVetorBMI <- function(D) {
  
  BMI <- numeric(nrow(D))
  
  Ano <- as.integer(substr(D$date, 1, 4))
  
  precosBigMacUS <- unique(D[D$isoA3 == "USA", c("localPrice", "date")])
  precosBigMacUS$Ano <- as.integer(substr(precosBigMacUS$date, 1, 4))
  
  for (i in 1:nrow(D)) {
    
    precoBigMacUS <- precosBigMacUS$localPrice[precosBigMacUS$Ano == Ano[i]]
    
    if (!is.na(precoBigMacUS) && precoBigMacUS != 0) {
      BMI[i] <- D$localPrice[i] / precoBigMacUS
    } else {
      BMI[i] <- NA
    }
  }
  
  return(BMI)
}

vetorBMI <- criaVetorBMI(D)

#Exercício 6
#Carrega o arquivo CSV no dataframe BigMacData
#Usaas funções criadas anteriormente para adicionar duas novas colunas ao dataframe: 'Ano' e 'BMI
#coluna 'Ano' é criada com a função 'criaVetorAno', que extrai o ano da coluna 'date'
#coluna 'BMI' é criada com a função 'criaVetorBMI', que calcula o índice Big Mac para cada linha

BigMacData <- read.csv('epData.csv', sep = ',')
ls.str(BigMacData)
vetorAno <- criaVetorAno(BigMacData)
BigMacData <- cbind(BigMacData, Ano = vetorAno)
vetorBMI <- criaVetorBMI(BigMacData)
BigMacData <- cbind(BigMacData, BMI = vetorBMI)


# Parte 7
#criando listas de códigos de países para cada continente e hemisfério
Paises <- c("ARE", "ARG", "AUS", "AUT", "AZE", "BEL", "BHR", "BRA", "CAN", "CHE", 
            "CHL", "CHN", "COL", "CRI", "CZE", "DEU", "DNK", "EGY", "ESP", "EST", 
            "FIN", "FRA", "GBR", "GRC", "GTM", "HKG", "HND", "HRV", "HUN", "IDN", 
            "IND", "IRL", "ISR", "ITA", "JOR", "JPN", "KOR", "KWT", "LBN", "LKA", 
            "LTU", "LVA", "MDA", "MEX", "MYS", "NIC", "NLD", "NOR", "NZL", "OMN", 
            "PAK", "PER", "PHL", "POL", "PRT", "QAT", "ROU", "RUS", "SAU", "SGP", 
            "SVK", "SVN", "SWE", "THA", "TUR", "TWN", "UKR", "URY", "USA", "VNM", 
            "ZAF")

Africa <- c("ZAF", "EGY")
Asia <- c("ARE", "AZE", "BHR", "CHN", "HKG", "IND", "IDN", "ISR", "JOR", "JPN", "KOR", "KWT", "LBN", "MDA", "MYS", "OMN", "PAK", "PHL", "QAT", "SAU", "SGP", "THA", "TUR", "TWN", "VNM")
Europe <- c("AUT", "BEL", "CHE", "CZE", "DEU", "DNK", "ESP", "EST", "FI N", "FRA", "GBR", "GRC", "HRV", "HUN", "IRL", "ITA", "LTU", "LVA", "NLD", "NOR", "POL", "PRT", "ROU", "RUS", "SVK", "SVN", "SWE", "UKR")
Americas <- c("ARG", "BRA", "CAN", "CHL", "COL", "CRI", "GTM", "HND", "MEX", "NIC", "PER", "USA", "URY")
Oceania <- c("AUS", "NZL")


HemisferioNorte <- c("ARE", "AUT", "AZE", "BEL", "BHR", "CAN", "CHE", "CHN", "CRI", "CZE", "DEU", "DNK", "EGY", "ESP", "EST", "FIN", "FRA", "GBR", "GRC", "GTM", "HKG", "HND", "HRV", "HUN", "IDN", "IND", "IRL", "ISR", "ITA", "JOR", "JPN", "KOR", "KWT", "LBN", "LTU", "LVA", "MDA", "MEX", "MYS", "NIC", "NLD", "NOR", "OMN", "PAK", "PHL", "POL", "PRT", "QAT", "ROU", "RUS", "SAU", "SGP", "SVK", "SVN", "SWE", "THA", "TUR", "TWN", "UKR", "USA", "VNM")
HemisferioSul <- c("ARG", "AUS", "BRA", "CHL", "COL", "IDN", "IND", "NZL", "PER", "URY", "ZAF")





#Exercício 8
#aggregate é usada para calcular a média do índice Big Mac para cada país e para cada continente
#resultados são plotados em um gráfico de barras para facilitar a comparação entre os países e continentes.
media_BigMac <- aggregate(BigMacData$BMI, by=list(Category=BigMacData$isoA3), FUN=mean)
barplot(media_BigMac$x, names.arg=media_BigMac$Category, xlab="Países", ylab="Índice Big Mac Médio", main="Índice Big Mac Médio por País")

media_BigMac_continente <- lapply(list(Africa, Asia, Europe, Americas, Oceania), function(x) mean(BigMacData$BMI[BigMacData$isoA3 %in% x]))
barplot(unlist(media_BigMac_continente), names.arg=c("África", "Ásia", "Europa", "Américas", "Oceania"), xlab="Continentes", ylab="Índice Big Mac Médio", main="Índice Big Mac Médio por Continente")

media_BigMac_hemisferio <- lapply(list(HemisferioNorte, HemisferioSul), function(x) mean(BigMacData$BMI[BigMacData$isoA3 %in% x]))
barplot(unlist(media_BigMac_hemisferio), names.arg=c("Hemisfério Norte", "Hemisfério Sul"), xlab="Hemisférios", ylab="Índice Big Mac Médio", main="Índice Big Mac Médio por Hemisfério")

#Exercício 9

BigMacData$indice_big_mac <- vetorBMI

dados_agrupados <- split(BigMacData, BigMacData$isoA3)
variancias <- sapply(dados_agrupados, function(df) var(df$indice_big_mac, na.rm = TRUE))

barplot(variancias, xlab = "Países", ylab = "Variância do Índice Big Mac", main = "Variância do Índice Big Mac dos Países")

