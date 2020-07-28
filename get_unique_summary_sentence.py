import os

def get_unique_sentence(filename):
    with open(filename) as f: 
        lines = f.readlines() 
    data = [] 

    for line in lines:
        line = line.strip("\n")
        line = line.split(",")
        if len(line) > 2:
            data.append(line[2])
    
    return set(data)


def main():
    filename = "/home/anuj/Downloads/videoset_data(1)/videoset_data/annotations/dy/dy01.csv"
    unique_sentences = get_unique_sentence(filename)
    print ("Total unique 5 sec clips", len(unique_sentences))
    print ("Duration of summamry reuired", len(unique_sentences) * 5 )


if __name__ == "__main__":
    main()

