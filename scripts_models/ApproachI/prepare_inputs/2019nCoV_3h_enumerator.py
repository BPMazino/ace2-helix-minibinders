#!/usr/bin/python

import argparse
import sys
import os
import random
import itertools

def parse_arguments( argv ):
    argv_tmp = sys.argv
    sys.argv = argv
    description = 'blueprint file generator for helical bundles. longxing@uw.edu 2020-02-12'
    parser = argparse.ArgumentParser( description = description )
    parser.add_argument('-maximum_length', type=int, default=88, help='maximum length of the protein')
    parser.add_argument('-nter_extension', action='store_true', default=False, help='n-terminal extension')
    parser.add_argument('-nc_extension', action='store_true', default=False, help='n-ter c-ter extention extension')
    parser.add_argument('-cter_extension', action='store_true', default=False, help='c-terminal extension')
    args = parser.parse_args()
    sys.argv = argv_tmp
    
    return args

loops = ['B', 'E', 'G',
         'GB', 'BB','GE', # GE??
         'GBB', 'BAB', 'BBB', 'BBG', 'BBE', 'GGB',
         'GABB', 'BBBB', 'GBBB', 'BAAB', 'BBAB', 'GBAB', 'BGBB',
         'GBBBB', 'BAABB', 'BBAAB', 'GABAB', 'BBGBB', 'BABBB'] 

loops = ['GB', 'BB',
         'GBB', 'BAB', 'BBB', 'BBG', 'BBE', 'GGB',
         'GABB', 'BBBB', 'GBBB', 'BAAB', 'BBAB', 'GBAB', 'BGBB',
         'GBBBB', 'BAABB', 'BBAAB', 'GABAB', 'BBGBB', 'BABBB'] 

motif_blueprint = '''1   V    HA    .
2   Q    HA    .
3   V    HA    .
4   V    HA    .
5   T    HA    .
6   V    HA    .
7   V    HA    .
8   V    HA    .
9   K    HA    .
10  V    HA    .
11  V    HA    .
12  H    HA    .
13  V    HA    .
14  V    HA    .
15  V    HA    .
16  D    HA    .
17  V    HA    .
18  V    HA    .
19  Y    HA    .
20  Q    HA    .
21  V    HA    .
22  V    HA    .
23  L    HA    .
24  V    HA    .'''

def write_loop(f, abego):
    for i in range(len(abego)):
        f.write('0   G    L%s    R\n' % abego[i])
def write_helix(f, num):
    for i in range(num):
        f.write('0   V    HA    R\n')

def write_loop_ss(f, abego):
    for i in range(len(abego)):
        f.write('0   G    D    R\n')
def write_helix_ss(f, num):
    for i in range(num):
        f.write('0   V    D    R\n')

def Nter_extension(maximum_length):
    motif_length = len( motif_blueprint.split('\n') ) - 1
    helix1 = [0,1,2,3,4]
    loop1 = loops
    helix2 = [motif_length-2, motif_length-1, motif_length, motif_length+1, motif_length+2]
    loop2 = loops
    for h1 in helix1:
        for l1 in loop1:
            for h2 in helix2:
                for l2 in loop2:
                    h3 = maximum_length - (motif_length + h1 + len(l1) + h2 + len(l2))
                    h3 = h3 if h3 < h2 else h2
                    name = "M{}H{}{}H{}{}H.bp".format(h1,l1,h2,l2,h3)
                    with open(name, 'w') as f:
                        motif_residues = motif_blueprint.split('\n')
                        for i in range(motif_length-1):
                            f.write(motif_residues[i]+'\n')
                        write_helix(f, h1)
                        write_loop(f, l1)
                        write_helix(f, h2)
                        write_loop(f, l2)
                        write_helix(f, h3)

def Cter_extension(maximum_length):
    motif_length = len( motif_blueprint.split('\n') ) - 1
    loop1 = loops
    helix2 = [motif_length-2, motif_length-1, motif_length, motif_length+1, motif_length+2]
    loop2 = loops
    helix3 = [0,1,2,3,4]
    for l1 in loop1:
        for h2 in helix2:
            for l2 in loop2:
                for h3 in helix3:
                    h1 = maximum_length - (motif_length + h2 + len(l1) + h3 + len(l2))
                    h1 = h1 if h1 < h2 else h2
                    name = "{}H{}{}H{}{}HM.bp".format(h1,l1,h2,l2,h3)
                    with open(name, 'w') as f:
                        write_helix(f, h1)
                        write_loop(f, l1)
                        write_helix(f, h2)
                        write_loop(f, l2)
                        write_helix(f, h3)
                        motif_residues = motif_blueprint.split('\n')
                        for i in range(1, motif_length):
                            f.write(motif_residues[i]+'\n')

def Nter_Cter_extension(maximum_length):
    motif_length = len( motif_blueprint.split('\n') ) - 2
    loop1 = loops
    helix2_1 = [0,1,2,3,4]
    helix2_2 = [0,1,2,3,4]
    loop2 = loops
    for l1 in loop1:
        for h2_1 in helix2_1:
            for l2 in loop2:
                for h2_2 in helix2_2:
                    h13 = (maximum_length-(len(l1)+h2_1+motif_length+h2_2+len(l2)))//2
                    if h13 > h2_1+motif_length+h2_2:
                        h13 = h2_1+motif_length+h2_2
                    name = "{}H{}{}HM{}H{}{}H.bp".format(h13,l1,h2_1,h2_2,l2,h13)
                    with open(name, 'w') as f:
                        write_helix(f, h13)
                        write_loop(f, l1)
                        write_helix(f, h2_1)
                        motif_residues = motif_blueprint.split('\n')
                        for i in range(1, motif_length-1):
                            f.write(motif_residues[i]+'\n')
                        write_helix(f, h2_2)
                        write_loop(f, l2)
                        write_helix(f, h13)


def main( argv ):
    args = parse_arguments( argv )

    if args.nter_extension:
        Nter_extension(args.maximum_length)
    if args.nc_extension:
        Nter_Cter_extension(args.maximum_length)
    if args.cter_extension:
            Cter_extension(args.maximum_length)

if __name__ == '__main__':
    main( sys.argv )
