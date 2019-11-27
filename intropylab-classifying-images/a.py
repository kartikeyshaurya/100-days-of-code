"""

import argparse
if __name__ == "_main_":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
"""
'''
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-o','--open-file', help='Description', required=False)
parser.add_argument('-s','--save-file', help='Description', required=False)

args = parser.parse_args()

print(args.open_file)
print(args.save_file)
'''
'''
import sys

print('Arguments:', len(sys.argv))
print('List:', str(sys.argv))

if sys.argv < 1:
    print('To few arguments, please specify a filename')

filename = sys.argv[1]
print('Filename:', filename) 
'''
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-o','--open-file', help='Description', required=False)
parser.add_argument('-s','--save-file', help='Description', required=False)

args = parser.parse_args()

print(args.open_file)
print(args.save_file)

