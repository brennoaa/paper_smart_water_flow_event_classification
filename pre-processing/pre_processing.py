import os
import re
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

def calcular_descida(pulsos,mediana):
    tempo_descida = 0          # acumulador que será retornado
    desceu = False             # flag que indica se já detectamos a descida
    tam = len(pulsos)         # tamanho da lista de pulsos/amostras
    zeros_consecutivos = 0

    # percorre da metade da lista até o final
    for i in range(int(tam / 2), tam):
        # compara o valor atual com o valor 40 amostras antes:
        # se pulsos[i] + 0.5 <= pulsos[i - 40] considera que "a vazão começou a cair"
        tempo_restante = tam - i
        # A descida completa nunca dura mais de 300 amostras.
        # Por isso, só consideramos o início da descida quando faltam menos de 300 pontos até o fim.
        if mediana<=1:
            if pulsos[i] + 0.1 <= pulsos[i - 10] and tempo_restante<900:
                # verifica os próximos valores (janela à frente) para garantir que
                # não exista um valor subsequente maior que o atual (i)
                subsequentes_menores = True
                for j in range(i + 1, min(i + 10, tam)):
                    if pulsos[j] > pulsos[i]:
                        subsequentes_menores = False
                        break  # se encontrar um subsequente maior, cancela a marcação
                    
                if pulsos[i] <= 0.1:
                    zeros_consecutivos += 1
                    if zeros_consecutivos >= 10: #Quando ocorre 10 vezes indica que acabou a descida
                        return tempo_descida                   
                else:
                    zeros_consecutivos = 0            
                # se todos os subsequentes dentro da janela forem menores ou iguais,
                # considera a "descida" válida
                if subsequentes_menores:
                    desceu = True
                else:
                    desceu = False
        else:    
            if pulsos[i] + 0.5 <= pulsos[i - 40] and tempo_restante<300:
                # verifica os próximos valores (janela à frente) para garantir que
                # não exista um valor subsequente maior que o atual (i)
                subsequentes_menores = True
                for j in range(i + 1, min(i + 40, tam)):
                    if pulsos[j] > pulsos[i]:
                        subsequentes_menores = False
                        break  # se encontrar um subsequente maior, cancela a marcação
                    
                if pulsos[i] == 0:
                    zeros_consecutivos += 1
                    if zeros_consecutivos >= 10: #Quando ocorre 10 vezes indica que acabou a descida
                        return tempo_descida                   
                else:
                    zeros_consecutivos = 0            
                # se todos os subsequentes dentro da janela forem menores ou iguais,
                # considera a "descida" válida
                if subsequentes_menores:
                    desceu = True
                else:
                    desceu = False
        
        # depois que desceu é detectada, conta 1 por cada iteração até o fim
        if desceu:
            tempo_descida += 1
    
    return tempo_descida


def calcular_repet(serie,mediana):
    """
    Conta quantas vezes o evento se repete na série temporal.
    Uma repetição ocorre quando o valor sai da mediana e encontra 10 zeros consecutivos depois volta para mediana.
    """
    if len(serie) == 0:
        return 0

    repeticoes = 0
    zeros_consecutivos = 0
    novo_evento = False
    estado = "mediana"  # estados possíveis: "mediana", "zeros"

    for valor in serie:
        if estado == "mediana":
            if (mediana - 0.1 <= valor <= mediana + 0.1) and novo_evento:
                estado = "zeros"
                repeticoes += 1
            elif valor == mediana:
                estado = "zeros"
        elif estado == "zeros":
            if valor == 0:
                estado = "mediana"
                novo_evento = True
    if repeticoes >= 1:
        return repeticoes + 1
    return repeticoes


def calcular_mediana(pulsos):
    # Filtra apenas valores maiores que 0.2
    valores = [valor for valor in pulsos if valor > 0.2]

    # Ordena os valores
    valores.sort()

    n = len(valores)
    meio = n // 2

    # Se a quantidade for ímpar
    if n % 2 == 1:
        return float(valores[meio])
    # Se for par
    else:
        return float((valores[meio - 1] + valores[meio]) / 2)

def medias_por_blocos(vetor, tamanho_bloco=10):
    vetor = np.array(vetor)
    n_blocos = len(vetor) // tamanho_bloco  # número de blocos completos
    medias = []
    
    for i in range(n_blocos):
        bloco = vetor[i*tamanho_bloco : (i+1)*tamanho_bloco]
        media = np.mean(bloco)
        medias.append(media)
        
    return np.array(medias)

def calcular_media(vetor):
    vetor_media = vetor.copy()
    #print(len(vetor))
    for i in range(0, len(vetor_media), 10): 
        media = np.mean(vetor_media[i:i+10]) 
        vetor_media[i:i+10] = media
    #for i in range(0,len(vetor)):
    #    print(f"{i}: {vetor[i]}")
    return vetor_media


def extrair_atributos(vetor, x):
    contaZeros = 0
    zerosConsecutivos = 0
    contaLinha = 0

    for Linha in vetor:
        contaLinha += 1
        if Linha == 0:
            zerosConsecutivos += 1
        if zerosConsecutivos >= 10:
            contaZeros += 1 # 1 segundo
            zerosConsecutivos = 0
        else:
            zerosConsecutivos = 0

    T = round((contaLinha / 10.0) - contaZeros, 2) # Tempo menos GAP
    pulsos_media = calcular_media(vetor)
    serie_tratada = medias_por_blocos(vetor)
    mediana = calcular_mediana(serie_tratada)
    repeticao = calcular_repet(serie_tratada, mediana)

    tempoDescida = 0  # valor padrão
    if repeticao == 0:
        tempoDescida = calcular_descida(pulsos_media,mediana)
    elif T<500:
        tempoDescida = 0
        T = T/repeticao
    
    a = [T, mediana, tempoDescida, repeticao, x]
    
    return a

# =========================================================
# CONFIGURAÇÃO
# =========================================================
LIMITE_MINUTOS = 6

# =========================================================
# EXTRAIR DATA/HORA DO NOME DO ARQUIVO
# =========================================================
def extrair_data_hora(arquivo_nome: str):
    nome = arquivo_nome.split(".")[0]
    partes = nome.split('-')

    ano = int(partes[0])
    mes = int(partes[1])
    dia = int(partes[2])

    hora_str = partes[3]

    m = re.match(r"(\d+)h(\d+)m(\d+)s(\d+)?", hora_str)
    if not m:
        raise ValueError(f"Não foi possível extrair hora de: {hora_str}")

    hora = int(m.group(1))
    minuto = int(m.group(2))
    segundo = int(m.group(3))

    return datetime(ano, mes, dia, hora, minuto, segundo)

# =========================================================
# ORDENAÇÃO NUMÉRICA
# =========================================================
def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# =========================================================
# LEITURA DO ARQUIVO .dat
# =========================================================
def ler_serie_dat(caminho_arquivo):
    serie = np.loadtxt(caminho_arquivo)
    return np.array(serie)

# =========================================================
# FUNÇÃO PRINCIPAL PARA JUNTAR EVENTOS
# =========================================================
def juntar_eventos_da_pasta(caminho_pasta):
    arquivos = sorted(
        [arq for arq in os.listdir(caminho_pasta) if arq.endswith(".dat")],
        key=numericalSort
    )

    eventos_unificados = []
    juntou = 0
    if len(arquivos) == 0:
        return eventos_unificados

    primeiro_arquivo = arquivos[0]
    caminho_primeiro = os.path.join(caminho_pasta, primeiro_arquivo)

    serie = ler_serie_dat(caminho_primeiro)
    horario_atual = extrair_data_hora(primeiro_arquivo)

    arquivos_do_evento = [primeiro_arquivo]

    # -----------------------------------------------------
    # Percorre os sucessores
    # -----------------------------------------------------
    for i in range(1, len(arquivos)):
        arquivo_sucessor = arquivos[i]
        caminho_sucessor = os.path.join(caminho_pasta, arquivo_sucessor)

        serie_sucessor = ler_serie_dat(caminho_sucessor)
        horario_sucessor = extrair_data_hora(arquivo_sucessor)

        diferenca_min = (horario_sucessor - horario_atual).total_seconds() / 60

        # -------------------------------------------------
        # Se diferença <= 6 min → junta no mesmo evento
        # -------------------------------------------------
        if diferenca_min <= LIMITE_MINUTOS:
            serie = np.concatenate([serie, serie_sucessor])
            horario_atual = horario_sucessor
            arquivos_do_evento.append(arquivo_sucessor)

        # -------------------------------------------------
        # Se diferença > 6 min → fecha evento atual
        # -------------------------------------------------
        else:
            tam = len(arquivos_do_evento)
            juntou = 1 if tam >= 2 else 0

            eventos_unificados.append({
                "horario_inicio": extrair_data_hora(arquivos_do_evento[0]),
                "horario_fim": horario_atual,
                "arquivos": arquivos_do_evento.copy(),
                "serie": serie.copy(),
                "juntou": juntou
            })

            # começa novo evento
            serie = serie_sucessor
            horario_atual = horario_sucessor
            arquivos_do_evento = [arquivo_sucessor]

    # -----------------------------------------------------
    # Processa o último evento acumulado
    # -----------------------------------------------------
    tam = len(arquivos_do_evento)
    juntou = 1 if tam >= 2 else 0
    eventos_unificados.append({
        "horario_inicio": extrair_data_hora(arquivos_do_evento[0]),
        "horario_fim": horario_atual,
        "arquivos": arquivos_do_evento.copy(),
        "serie": serie.copy(),
        "juntou": juntou
    })

    return eventos_unificados

caminho_raiz = r"C:\data"

subpastas = os.listdir(caminho_raiz)

todos_eventos = []

for subpasta in subpastas:
    caminho_subpasta = os.path.join(caminho_raiz, subpasta)

    if os.path.isdir(caminho_subpasta):
        eventos_subpasta = juntar_eventos_da_pasta(caminho_subpasta)

        for evento in eventos_subpasta:
            evento["classe"] = subpasta

        todos_eventos.extend(eventos_subpasta)

print(f"Total geral de eventos: {len(todos_eventos)}")


X = []
Y = []
for i, evento in enumerate(todos_eventos):
    a = extrair_atributos(evento["serie"], evento["juntou"])
    X.append(a)
    Y.append(evento["classe"])
