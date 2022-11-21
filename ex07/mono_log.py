import sys
import numpy as np
import pandas as pd


def get_arg():
    if len(sys.argv) != 2:
        raise Exception("wrong number of arguments")
    s = sys.argv[1]
    param = s.split("=")
    if param[0] != "-zipcode":
        raise Exception("wrong arg name")
    zipcode = int(param[1])
    if (zipcode < 0 or zipcode > 3):
        raise Exception("wrong arg value")
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))
    return zipcode

def load_datasets():
    content = pd.read_csv("solar_system_census.csv")
    X = np.array(content[["weight", "height", "bone_density"]])
    if X.shape[1] !=  3:
        raise Exception("Datas are missing in solar_system_census.csv")        
    content = pd.read_csv("solar_system_census_planets.csv")
    Y = np.array(content[["Origin"]])
    if Y.shape[1] !=  1:
        raise Exception("Datas are missing in solar_system_census_planets.csv")   
    return X, Y

def main():
    try:
        zipcode = get_arg()
    except Exception as e:
        print(f"{e}, use -zipcode=X with X being 0, 1, 2 or 3 to start")
        return
    try:
        X, Y = load_datasets()
    except Exception as e:
            print("Error in datas loading :", e)
            return
    print(X, Y)
    
    
    

if __name__ == "__main__":
    main()