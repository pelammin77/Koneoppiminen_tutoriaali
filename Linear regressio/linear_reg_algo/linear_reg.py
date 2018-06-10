"""
Tiedosto: linear_tutorial.py
Tekijä: Petri Lamminaho
Kuvaus: Yksinkertainen lineaarinen regressio algoritmi
        Toteuteutettu keväällä 2018 osana Youtubessa pitämääni
        koneoppimis-tutoriaalia
#######################################################################
Ominaisuudet:
            - Laskee mallin m ja b arvot
            -Pirtää datapisteet Mathplotlib-kirjaston avulla
            -Piirtää best fit linen
            -Ennustaa y:n arvon annetun x:n avulla
            -Piirtää ennusteviivan
            -Laskee mallin error arvon
            -Laskee mallin luottettavuuden
            -Generoi isomman testidatan random-kirjaston avulla
            - Osaa lukea datan csv-tiedostosta käyttäen Pandas-kirjastoa
"""
##################################################################
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import random
import pandas as pd

#reg-linen kaava
# f(x)= mx+b

# testi data
#########################################################
Xs = np.array( [0,1, 2, 3, 4, 5,6,7 ], dtype=np.float64)
Ys = np.array( [3,5, 4, 6, 5, 6,8, 9  ], dtype=np.float64)
#Xs = np.array( [1, 2, 3, 4, 5 ], dtype=np.float64)
#Ys = np.array( [5, 4, 6, 5, 6 ], dtype=np.float64)
#########################################################
print(plt.style.available)# tulostaa käytettävissä olevat tyylit
plt.style.use('fivethirtyeight') # ottaa tyylin käyttöön
#################################################
def laske_errorin_nelio(ys_viiva, ys_pisteet):
    """
    Laskee funkion virheen summan neliön (SE-arvon) ja palauttaa sen
    :param ys_viiva:
    :param ys_pisteet:
    :return: sum((ys_viiva-ys_pisteet) ** 2)
    """
    return sum((ys_viiva-ys_pisteet) ** 2)
########################################################################
def luo_testi_data(num_of_data_points, variance, step=2, correlation=False):
    """
    Luo isomman testidatan randomilla
    Voidaan määrittää pisteiden määrä, datan varianssi,  steppi sekä korrelaatio-arvo
    Palauttaa pisteet numpy-taulukossa
    :param num_of_data_points:
    :param variance:
    :param step:
    :param correlation:
    :return: np.array(xs,dtype=np.float64), np.array(ys, dtype=np.float64)
    """
    #xs = []
    ys = []
    val = 1
    for i in range(num_of_data_points):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == "pos":
            val+= step
        elif correlation and correlation =="neg":
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs,dtype=np.float64), np.array(ys, dtype=np.float64)
#############################################################################
def laske_mallin_luotettavuus(ys_viiva, ys_pisteet):
    """
    Laskee mallin luotettavuuden
    :param ys_viiva:
    :param ys_pisteet:
    :return: 1 - (best_fit_line_SE / mean_line_SE)
    """
    ys_mean_line = [mean(ys_pisteet) for _ in ys_pisteet]
    best_fit_line_SE = laske_errorin_nelio(ys_viiva, ys_pisteet)
    mean_line_SE = laske_errorin_nelio(ys_mean_line, ys_pisteet)
    return 1 - (best_fit_line_SE / mean_line_SE)
########################################################################
def palauta_datan_suurin_piste(data):
    """
    apufunkio joka palautaa taulukon suurimman alkion
    :param data:
    :return:  data[np.argsort(data)[-1]]
    """
    return data[np.argsort(data)[-1]]# järjestää taulukon suuruusjärjestykseen
                                     # ja palauttaa viimeisen(suurimman) alkion
########################################################################
def laske_m_ja_b(xs, ys):
    """
    Laskee mallin b(y:n leikkauspiste) ja m( viivan kulmakerroin)
    lopukksi palauttaa arvot
    :param xs:
    :param ys:
    :return: m, b
    """
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
         ((mean(xs) **2) - mean(xs **2)))
    b = mean(ys) - m * mean(xs)
    return m, b
#####################################################
if __name__ == '__main__':
    dataframe = pd.read_csv('data.csv', sep=',')
    print(dataframe.head())
    Xs = np.array(dataframe['x'])
    Ys = np.array(dataframe['y'])
    suurin_piste_x = palauta_datan_suurin_piste(Xs)
    m, b = laske_m_ja_b(Xs, Ys)
    print("m on ", round(m, 2))
    print("b on",round(b, 2))
    reg_line = []
    line=[(m*x)+b for x in Xs]
    enuste_x = 50
    enuste_y = (m * enuste_x) + b
    ennuste_line = [(m*x) + b for x in range(int(suurin_piste_x), int(enuste_x + 2))]
    print()
    rel = laske_mallin_luotettavuus(line, Ys)
    rel = round(rel,2)
    print("Ennusteen luotettavuus on", rel*100, "%")
    print("y:n arvo pisteessä", enuste_x,"on",round(enuste_y, 2))
    plt.scatter(Xs,Ys)# piirtää pisteet
    plt.plot(Xs, line)# piirtää reg-linen
    plt.scatter(enuste_x, enuste_y)#piirtää ennustepisteen
    plt.plot(range(int(suurin_piste_x), int(enuste_x + 2)), ennuste_line, c="b")#piirtää ennusteviivan
    plt.show()#näyttää plt-kaavion
