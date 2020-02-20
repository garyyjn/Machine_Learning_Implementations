import pandas as pd
import numpy as np

rock = pd.read_csv('vectorizations/rock_vectorized.csv')
country = pd.read_csv('vectorizations/vectorized_country.csv')
folk = pd.read_csv('vectorizations/vectorized_folk.csv')
hip_pop = pd.read_csv('vectorizations/hip-hop_vectorized.csv')
pop = pd.read_csv('vectorizations/pop_vectorized.csv')

hip_pop = hip_pop.to_numpy()
pop = pop.to_numpy()
rock = rock.to_numpy()
country = country.to_numpy()
folk = folk.to_numpy()

def word_vector_to_apperence(matrix):
    numbers = matrix[:,4:]
    return (numbers > 0).astype(int)

hip_pop_bin = word_vector_to_apperence(hip_pop)
np.savetxt("appearance_vectors/hip_pop_bin.csv", hip_pop_bin, delimiter=",")