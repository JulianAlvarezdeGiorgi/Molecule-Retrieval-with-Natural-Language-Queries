# Reorder the files
import os
import shutil
import sys
import argparse
import re


# All files that end in 1 are going to be moved to the drop_dir
# All files that end in 2 are going to be moved to the subgraph_dir

source_dir = "/home/infres/jalvarez-22/ALTEGRAD/Challenge/Public/data_enhanced/raw"
origin_dir = "/home/infres/jalvarez-22/ALTEGRAD/Challenge/Public/data_enhanced/raw/origin"
drop_dir = "/home/infres/jalvarez-22/ALTEGRAD/Challenge/Public/data_enhanced/raw/drop"
subgraph_dir = "/home/infres/jalvarez-22/ALTEGRAD/Challenge/Public/data_enhanced/raw/subgraph"

def main():
    # all files 
    files = os.listdir(source_dir)
    print("Number of files: ", len(files))
    for file in files:
        if file.endswith("1.graph"):
            shutil.move(source_dir + "/" + file, drop_dir + "/" + file)
        elif file.endswith("2.graph"):
            shutil.move(source_dir + "/" + file, subgraph_dir + "/" + file)
        elif file.endswith("0.graph"):
            shutil.move(source_dir + "/" + file, origin_dir + "/" + file)
        else:
            print("File does not end in 1 or 2")
            exit(1)

if __name__ == "__main__":
    main()

