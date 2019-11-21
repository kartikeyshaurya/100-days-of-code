#day2
#author kartikeyshaurya

import argparse
# Main program function defined below
# Creates Argument Parser object named parser
parser = argparse.ArgumentParser()

# Argument 1: that's a path to a folder
parser.add_argument('--he', type=str, default='this is kartikey shaurya ',
                        help='path to the folder my_folder')

# Argument 2: that's an integer
parser.add_argument('--num', type=int, default=1,help='Number (integer) input')

# Assigns variable in_args to parse_args()
in_args = parser.parse_args()

# Accesses values of Arguments 1 and 2 by printing them
print("Argument 1:", in_args.he, "  Argument 2:", in_args.num)

