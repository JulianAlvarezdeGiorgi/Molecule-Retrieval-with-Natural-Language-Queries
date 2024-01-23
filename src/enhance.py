"""
Copy each molecule and add 0 at the end of its CID
Do node dropping augmentation on each molecule and add 1 at the end of its CID
Do subgraph augmentation on each molecule and add 2 at the end of its CID

Copy all the other files.
The train set is enhanced with those two augmentations,
there are three times more element in the train set

The validation set and test set remains untouched
"""

import os
import shutil
import random

source_data = "/home/admpc/Documents/MVA/S1/altegrad/projet/Public/data/"
target_data = "/home/admpc/Documents/MVA/S1/altegrad/projet/Public/data_enhanced/"

def node_drop(lines,p=0.8) :
    """
    from a CID file lines and a proportion
    keep p% of the nodes by random selection

    Return a sorted array of the nodes to keep
    """
    index=lines.index("idx to identifier:\n")
    nb_nodes=len(lines)-index-1

    #construct list of node to keep
    keep=list(range(nb_nodes))
    while len(keep)>int(p*nb_nodes) :
        keep.pop(random.randint(0,len(keep)-1))
    return keep

def subgraph(lines,p=0.8) :
    """
    from a CID file lines and a proportion
    keep p% of the nodes by random walk

    Return a sorted array of the nodes to keep
    """
    index=lines.index("idx to identifier:\n")
    nb_nodes=len(lines)-index-1

    #construct adjacency list
    adj={k:set() for k in range(nb_nodes)}
    for line in lines[1:index-1] :
        v1,v2=line[:-1].split(" ")
        v1,v2=int(v1),int(v2)
        adj[v1].add(v2)

    #construct list of node to keep
    v=random.randint(0,nb_nodes-1)
    keep={v}
    neigh=adj[v].copy()
    while len(keep)<p*nb_nodes :
        v=random.choice(list(neigh))
        if v not in keep :
            keep.add(v)
            neigh=neigh.union(adj[v])
    keep=list(keep)
    keep.sort()
    return keep

def write_keep(lines,keep,f_target) :
    """
    From a CID file lines, a sorted list of node to keep and a file

    Write the subgraph with reindexing the nodes
    """

    # construct mapping from old idx to new idx
    index=lines.index("idx to identifier:\n")
    nb_nodes=len(lines)-index-1

    mapping={}
    k=0
    for idx in range(nb_nodes) :
        if k<len(keep) and idx==keep[k] :
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

try : os.mkdir(target_data)
except : pass
try : os.mkdir(target_data+"raw/")
except : pass

print("Copy single files")
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
            f_target.write(cid+"2\t"+text)

with open(source_data+"val.tsv", 'r') as f_source :
    with open(target_data+"val.tsv", 'w+') as f_target :
        for l in f_source.readlines() :
            cid,text=l.split('\t')
            f_target.write(cid+"0\t"+text)

print("Copy raw files")
for f in os.listdir(source_data+"raw/") :
    cid=f.split('.')[0]
    shutil.copyfile(source_data+"raw/"+f, target_data+"raw/"+cid+"0.graph")

print("Node dropping")
for f in os.listdir(source_data+"raw/") :
    if f!= ".DS_Store" :
        cid=f.split('.')[0]
        with open(source_data+"raw/"+f, 'r') as f_source :
            lines=f_source.readlines()
        keep=node_drop(lines)
        with open(target_data+"raw/"+cid+"1.graph", 'w+') as f_target :
            write_keep(lines,keep,f_target)

print("Sub graph")
for f in os.listdir(source_data+"raw/") :
    if f!= ".DS_Store" :
        cid=f.split('.')[0]
        with open(source_data+"raw/"+f, 'r') as f_source :
            lines=f_source.readlines()
        keep=subgraph(lines)
        with open(target_data+"raw/"+cid+"2.graph", 'w+') as f_target :
            write_keep(lines,keep,f_target)
