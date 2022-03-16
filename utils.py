import matplotlib.pyplot as plt 

def gen_graph(items, name, run):
    plt.plot(range(len(items)), items)
    plt.xlabel("Epoch")
    plt.ylabel(name)
    plt.savefig(f"./images/{name}/{run}")
    plt.clf()