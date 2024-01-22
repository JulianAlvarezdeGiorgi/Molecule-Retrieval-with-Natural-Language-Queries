"""
Copy each molecule and add 0 at the end of its CID
Do node dropping augmentation on each molecule and add 1 at the end of its CID

TODO : do the same with subgraph (random walk)
"""

import os
import shutil
import random

source_data = "/home/admpc/Documents/MVA/S1/altegrad/projet/Public/data/"
target_data = "/home/admpc/Documents/MVA/S1/altegrad/projet/Public/data_enhanced/"

def node_drop(f_source,f_target,p=0.8) :
    """
    from a source file compute node drop in target file
    keep the proportion p of nodes
    """
    lines=f_source.readlines()
    index=lines.index("idx to identifier:\n")
    nb_nodes=len(lines)-index-1
    #construct list of node to keep
    keep=list(range(nb_nodes))
    while len(keep)>int(p*nb_nodes) :
        keep.pop(random.randint(0,len(keep)-1))
    # construct mapping from old idx to new idx
    mapping={}
    k=0
    for idx in range(nb_nodes) :
        if k<int(p*nb_nodes) and idx==keep[k] :
            mapping[str(idx)]=str(k)
            k+=1
        else :
            mapping[str(idx)]=None
    #write edges
    f_target.write("edgelist:\n")
    for line in lines[1:index-1] :
        x1,x2=line[:-1].split(" ")
        y1,y2=mapping[x1],mapping[x2]
        if y1 is not None and y2 is not None :
            f_target.write(y1+' '+y2+'\n')

    f_target.write("\n")
    #write nodes
    f_target.write("idx to identifier:\n")
    for line in lines[index+1:] :
        x,identifier=line[:-1].split(" ")
        y=mapping[x]
        if y is not None:
            f_target.write(y+' '+identifier+'\n')


#copy each original molecule and add 0 at the end of the ID
#do node dropping augmentation and add 1 at the end of the ID

try : os.mkdir(target_data)
except : pass
try : os.mkdir(target_data+"raw/")
except : pass

print("\nCopy single files")
shutil.copyfile(source_data+"test_text.txt", target_data+"test_text.txt")
shutil.copyfile(source_data+"token_embedding_dict.npy", target_data+"token_embedding_dict.npy")

with open(source_data+"test_cids.txt", 'r') as f_source :
    with open(target_data+"test_cids.txt", 'w+') as f_target :
        for l in f_source.readlines() :
            f_target.write(l[:-1]+"0\n")

with open(source_data+"train.tsv", 'r') as f_source :
    with open(target_data+"train.tsv", 'w+') as f_target :
        for l in f_source.readlines() :
            cid,text=l.split('\t')
            f_target.write(cid+"0\t"+text)
            f_target.write(cid+"1\t"+text)

with open(source_data+"val.tsv", 'r') as f_source :
    with open(target_data+"val.tsv", 'w+') as f_target :
        for l in f_source.readlines() :
            cid,text=l.split('\t')
            f_target.write(cid+"0\t"+text)

print("\nCopy raw files")
for f in os.listdir(source_data+"raw/") :
    cid=f.split('.')[0]
    shutil.copyfile(source_data+"raw/"+f, target_data+"raw/"+cid+"0.graph")

print("\nNode dropping")
for f in os.listdir(source_data+"raw/") :
    if f!= ".DS_Store" :
        cid=f.split('.')[0]
        with open(source_data+"raw/"+f, 'r') as f_source :
            with open(target_data+"raw/"+cid+"1.graph", 'w+') as f_target :
                node_drop(f_source,f_target)
