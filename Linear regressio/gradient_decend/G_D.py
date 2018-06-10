"""Tiedosto: G_D.py
Tekijä: Petri Lamminaho
Kuvaus: Yksinkertainen Gradiend decend algoritmi joka säätää best fit linea gradientin avulla ja
        laskee error-arvon
        Lineaari datalle Apuna käytetään vain Numpy-kirjastoa
        Toteuteutettu keväällä 2018 osana Youtubessa pitämääni koneoppimis-tutoriaalia
"""
import numpy as np
#######################################################
def calculate_error(b, m, data_points):
    """
    Laskee algoritmin virheen neliön ja palauttaa keskiarvo errorin

    :param b:
    :param m:
    :param data_points:
    :return: err / float(len(data_points))
    """
    err = 0
    for i in range(len(data_points)):
        x = data_points[i, 0]
        y = data_points[i,1]
        err += (y - (m *x + b)) **2 #summafunktio laskee errorin neliön
    return err / float(len(data_points))# palauttaa keskimääräisen errorin
#################################################################
def gradiend_step(b_current, m_current, data_points, LR):
    """
    Funktio laskee gradientin b:le ja m:lle osittais derivaatalla
    Säätää best fit linea
    :param b_current:
    :param m_current:
    :param data_points:
    :param LR:
    :return: [updated_b, updated_m]
    """
    grad_b = 0
    grad_m = 0
    N = float(len(data_points))
    for i in range(len(data_points)):
        x = data_points[i, 0]
        y = data_points[i, 1]
        grad_b += -(2/N) * (y - ((m_current * x) + b_current))# laskee gradient b osittais derivaatalla
        grad_m += -(2/N) * x * (y - ((m_current * x) + b_current)) # laskee gradient m osittais derivaatalla
    updated_b = b_current - (LR * grad_b) #vännetään nykyisestä b:stä  gradientti b
                                          #saadaan pos tai neg arvo(suuntima mihin päin mennään
                                        # korjataan tulosta kertomalla LR:llä

    updated_m = m_current - (LR * grad_m) #vännetään nykyisestä m:stä  gradientti m
                                          #saadaan pos tai neg arvo(suuntima mihin päin mennään
                                          # korjataan tulosta kertomalla LR:llä
    return [updated_b, updated_m] # palautetaan päivitetyt b ja m
#################################################################################
def run_gradient_decent(data_points, starting_b, starting_m, LR, num_iteration):
    """
    kutsuu gradiend_step funktiota jokaisella iteraatiolla
    palauttaa laketut b ja m
    :param data_points:
    :param starting_b:
    :param starting_m:
    :param LR:
    :param num_iteration:
    :return: [b, m ]
    """
    b = starting_b
    m = starting_m
    for i in range(num_iteration):
        b, m = gradiend_step(b,m, np.array(data_points), LR)
    return [b, m]
########################################################################
def run():
     ##Tekee alkuvalmistelut
     ## Kutsuu run_gradient_decent funkiota
     ##  piirtää graafit
   data = np.genfromtxt("data.csv",delimiter=',') # luetaan data
   learning_rate =  0.0001 # kuinka nopeasti algoritmi oppii
   starting_b = 0 # alku b
   starting_m = 0 # alku m
   num_iter = 1000 #Montako kertaa treenataan
   print("Aloitetaan....")
   print("b on:", starting_b)
   print("m on:", starting_m)
   print("Error on:", calculate_error(starting_b, starting_m, data))
   print(num_iter, "askeleen jälkeen")
   b, m  =run_gradient_decent(data,starting_b, starting_m, learning_rate,num_iter)
   print("b:",b)
   print("m:",m)
   print("Error:",calculate_error(b,m,data))
   ## Piirretään best fit line##
   import matplotlib.pyplot as plt
   xs = []
   ys =[]
   for i in range(len(data)):
       x = data[i, 0]
       y = data[i, 1]
       xs.append(x)
       ys.append(y)
   plt.scatter(xs, ys)
   reg_line = [(m * x) + b for x in xs]
   plt.plot(xs, reg_line)
   enuste_x = 50
   enuste_y = (m * enuste_x) + b
   print("y:n arvo pisteessä", enuste_x, "on", round(enuste_y, 2))
   plt.scatter(enuste_x, enuste_y)
   plt.show()
#########################################################################################
if __name__ == '__main__':
    """
    Pääohjelma kutsuu run-funktiota 
    """
    run()
 ###################################################################################