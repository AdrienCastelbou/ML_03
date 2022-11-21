import sys
import numpy as np

def main():
    try:
        if len(sys.argv) != 2:
            raise Exception("wrong number of arguments")
        s = sys.argv[1]
        param = s.split("=")
        if param[0] != "-zipcode":
            raise Exception("wrong param name")
        zipcode = int(param[1])
        if (zipcode < 0 or zipcode > 3):
            raise Exception("wrong param value")
        print('Number of arguments:', len(sys.argv), 'arguments.')
        print('Argument List:', str(sys.argv))
    except Exception as e:
        print(f"{e}, use -zipcode=X with X being 0, 1, 2 or 3 to start")

if __name__ == "__main__":
    main()