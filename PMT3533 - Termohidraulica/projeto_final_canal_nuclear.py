# Importação das bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from iapws import IAPWS97


# Definição do número de pontos da malha
nos = int(input("Defina o número de pontos da malha: "))


# Parâmetros do Elemento Combustível (EC)
Erev = 0.00038        # Espessura do revestimento em metros
Ecomb = 0.00076       # Espessura do combustível em metros
Ecint = 0.00289       # Espessura interna do canal em metros
Ecext = 0.00452       # Espessura externa do canal em metros
l_placa = 0.0626      # Largura da placa em metros
h_placa = 0.6         # Altura da placa em metros
l_canal = 0.0671      # Largura do canal em metros
n_EC = 25             # Número de EC no reator
n_placa = 18          # Número de placas por EC


# Parâmetros do núcleo
fp = 1.0              # Fração de potência (para a Condição 2)
fv = 1.0              # Fração de vazão (para a Condição 2)
p_total = 5_000_000 * fp  # Potência total de operação [W]
fr = 1.914            # Fator radial de potência


# Propriedades do material
k_rev = 180           # Condutividade do revestimento [W/m.K]
Tin = 30.0 + 273.15   # Temperatura de entrada [K]
Pin = 0.16            # Pressão de entrada no canal [MPa]
Pout = 0.15           # Pressão de saída no canal [MPa]
vazao_ec = 22.8 * fv  # Vazão de fluido por elemento combustível [m³/h]


# Cálculo de parâmetros derivados
A_canal = l_canal * Ecint  # Área de escoamento do canal [m²]
A_troca = 2 * h_placa * l_placa  # Área de troca por placa [m²]
A_no = A_troca / nos  # Área de troca por nó [m²]
Dh = 4 * A_canal / (2 * (Ecint + l_canal))  # Diâmetro hidráulico [m]
vazao_canal = vazao_ec / (17 * 3600)  # Vazão por canal interno [m³/s]
p_placa = p_total / (n_EC * n_placa)  # Potência por placa [W]
q_med = p_placa / A_troca  # Fluxo médio por placa [W/m²]


# Pressão ao longo do canal
pressao = np.linspace(Pin, Pout, nos)  # Pressão ao longo do canal [MPa]


# Inicialização de vetores de temperatura
Twater = np.zeros(nos)      # Temperatura do fluido [K]
Tinter = np.zeros(nos)      # Temperatura na interface [K]
Trev = np.zeros(nos)        # Temperatura do revestimento [K]
Tcomb = np.zeros(nos)       # Temperatura do combustível [K]


# Posições dos nós
pos_in = -h_placa / 2 + h_placa / (2 * nos)
pos_out = pos_in + (nos - 1) * (h_placa / nos)
y = np.linspace(pos_in, pos_out, nos)  # Posições ao longo do canal [m]


# Distribuição do fluxo de calor ao longo da placa
def q_ponto(y):
    return fr * q_med * np.cos((np.pi / 2) * (y / (h_placa / 2)))


q_nos = q_ponto(y)  # Fluxo de calor em cada nó [W/m²]


# Função para calcular a condutividade térmica do combustível (unidades SI)
def calcula_kf(T):
    T_F = (T - 273.15) * 9/5 + 32  # Converter K para °F
    kf = (3978.1 / (692.61 + T_F)) + 6.023366e-12 * (T_F + 460) ** 3
    kf_SI = kf * 1.730735  # Converter BTU/h.ft.°F para W/m.K
    return kf_SI


# Função para obter as propriedades da água
def water_props(T, P):
    return IAPWS97(T=T, P=P)


# Cálculo da vazão mássica por canal [kg/s]
water_in = water_props(Tin, Pin)
rho_in = water_in.rho  # Densidade na entrada [kg/m³]
mass_flow = rho_in * vazao_canal  # Vazão mássica por canal [kg/s]


# Funções auxiliares para cálculos
def calc_velocity():
    return vazao_canal / A_canal  # Velocidade [m/s]


def calc_Re(rho, mu, vel):
    return rho * vel * Dh / mu


def calc_Pr(mu, cp, k):
    return mu * cp / k


def calc_h(k, Re, Pr):
    Nu = 0.023 * Re ** 0.8 * Pr ** 0.4
    return Nu * k / Dh


# Pré-cálculo de propriedades da água
rho_array = np.zeros(nos)
mu_array = np.zeros(nos)
cp_array = np.zeros(nos)
k_array = np.zeros(nos)
vel_array = np.zeros(nos)
Re_array = np.zeros(nos)
Pr_array = np.zeros(nos)
h_array = np.zeros(nos)


# Propriedades iniciais do fluido
water_node = water_props(Tin, Pin)
rho_array[0] = water_node.rho
mu_array[0] = water_node.mu
cp_array[0] = water_node.cp * 1000  # Converter para J/kg.K
k_array[0] = water_node.k
vel_array[0] = calc_velocity()
Re_array[0] = calc_Re(rho_array[0], mu_array[0], vel_array[0])
Pr_array[0] = calc_Pr(mu_array[0], cp_array[0], k_array[0])
h_array[0] = calc_h(k_array[0], Re_array[0], Pr_array[0])


# Cálculo das temperaturas do fluido
Twater[0] = Tin + q_nos[0] * A_no / (mass_flow * cp_array[0])


for i in range(1, nos):
    # Atualiza a temperatura do fluido
    Twater[i] = Twater[i - 1] + q_nos[i] * A_no / (mass_flow * cp_array[i - 1])


    # Atualiza as propriedades da água
    water_node = water_props(Twater[i], pressao[i])
    rho_array[i] = water_node.rho
    mu_array[i] = water_node.mu
    cp_array[i] = water_node.cp * 1000  # Converter para J/kg.K
    k_array[i] = water_node.k
    vel_array[i] = calc_velocity()
    Re_array[i] = calc_Re(rho_array[i], mu_array[i], vel_array[i])
    Pr_array[i] = calc_Pr(mu_array[i], cp_array[i], k_array[i])
    h_array[i] = calc_h(k_array[i], Re_array[i], Pr_array[i])


# Cálculo das temperaturas na interface
for i in range(nos):
    Tinter[i] = Twater[i] + q_nos[i] / h_array[i]


# Cálculo das temperaturas do revestimento
for i in range(nos):
    Trev[i] = Tinter[i] + q_nos[i] * Erev / k_rev


# Cálculo das temperaturas do combustível
# Estimativa inicial
Tcomb[0] = Trev[0] + q_nos[0] * Ecomb / (4 * calcula_kf(Trev[0]))
for i in range(1, nos):
    kf = calcula_kf(Tcomb[i - 1])
    Tcomb[i] = Trev[i] + q_nos[i] * Ecomb / (4 * kf)


# Converter temperaturas para Celsius
Twater_C = Twater - 273.15
Tinter_C = Tinter - 273.15
Trev_C = Trev - 273.15
Tcomb_C = Tcomb - 273.15


# Gráfico de distribuição das temperaturas
plt.figure(figsize=(12, 8))
plt.plot(y, Twater_C, label="Água (°C)", color='blue', marker='x')
plt.plot(y, Tinter_C, label="Interface (°C)", color='green', marker='^')
plt.plot(y, Trev_C, label="Revestimento (°C)", color='orange', marker='s')
plt.plot(y, Tcomb_C, label="Combustível (°C)", color='red', marker='o')
plt.title(f"Distribuição de temperaturas com {nos} nós para o canal mais quente ao longo da posição y")
plt.xlabel("Posição y [m]")
plt.ylabel("Temperatura [°C]")
plt.legend()
plt.grid(True)
plt.show()


# Gráfico do coeficiente h ao longo de y
plt.figure(figsize=(12, 8))
plt.plot(y, h_array, label="Coeficiente h [W/m².K]", color='black', marker='o')
plt.title(f"Distribuição do coeficiente de transferência com {nos} nós de calor convectivo (h) ao longo do canal")
plt.xlabel("Posição y [m]")
plt.ylabel("h [W/m².K]")
plt.legend()
plt.grid(True)
plt.show()


# Criação de um DataFrame com os dados
dados = {
    "y [m]": y,
    "Temperatura do combustível [°C]": Tcomb_C,
    "Temperatura do revestimento [°C]": Trev_C,
    "Temperatura da interface [°C]": Tinter_C,
    "Temperatura da água [°C]": Twater_C,
    "Coeficiente de convecção h [W/m².K]": h_array,
    "Pressão [MPa]": pressao
}


tabela = pd.DataFrame(dados)


# Salvar em um arquivo Excel
arquivo = f"tabela_{nos}_nos.xlsx"
tabela.to_excel(arquivo, index=False)
print(f"Tabela salva com sucesso no arquivo: {arquivo}")


# Exibir a tabela
print("Tabela de valores para cada ponto da malha:")
print(tabela)
