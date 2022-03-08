import numpy as np

def one_component_model(w1, w2):
    def model(t, n, triplet_frac, triplet_time, t_d):
        triplet_term = 1 + triplet_frac/(1-triplet_frac)*(np.exp(-t/triplet_time))
        term1 = 1/n
        term2 = 1/((1+(t/t_d))*np.sqrt((1+t/t_d*(w1/w2)**2)))
        return 1+triplet_term*term1*term2
    return model

def two_component_model(w1, w2):
    def model(t, triplet_frac, triplet_time, n, f1, t_d1, t_d2):
        triplet_term = 1 + triplet_frac/(1-triplet_frac)*(np.exp(-t/triplet_time))
        term1 = 1/n
        term2 = f1/((1+(t/t_d1))*np.sqrt((1+t/t_d1*(w1/w2)**2)))
        term3 = (1-f1)/((1+(t/t_d2))*np.sqrt((1+t/t_d2*(w1/w2)**2)))
        return 1+triplet_term*term1*(term2+term3)
    return model

def add_second_component(n, t_d1, w1, w2):
    def two_component_model(t, triplet_frac, triplet_time, f1, t_d2):
        triplet_term = 1 + triplet_frac/(1-triplet_frac)*(np.exp(-t/triplet_time))
        term1 = 1/n
        term2 = f1/((1+(t/t_d1))*np.sqrt((1+t/t_d1*(w1/w2)**2)))
        term3 = (1-f1)/((1+(t/t_d2))*np.sqrt((1+t/t_d2*(w1/w2)**2)))
        return 1+triplet_term*term1*(term2+term3)
    return two_component_model