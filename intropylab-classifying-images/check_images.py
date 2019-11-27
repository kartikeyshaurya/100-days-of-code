
# PURPOSE: Check images & report results: read them in, predict their
#          content (classifier), compare prediction to actual value labels
#          and output results



import argparse
from time import time, sleep
from os import listdir
from classifier import classifier # imoorting to use cnn images
# Imports print functions that check the lab
from print_functions_for_lab_checks import *


def main():
    # collecting start time
    start_time = time()

    # line arguments -- defined get_input_args() function to create and retrieve command
    in_arg = get_input_args()
    
    # TODO: 3. Define get_pet_labels() function to create pet image labels by
    # creating a dictionary with key=filename and value=file label to be used
    # to check the accuracy of the classifier function
    answers_dic   = get_pet_labels()

    # TODO: 4. Define classify_images() function to create the classifier 
    # labels with the classifier function using in_arg.arch, comparing the 
    # labels, and creating a dictionary of results (result_dic)
    result_dic = classify_images()
    
    # TODO: 5. Define adjust_results4_isadog() function to adjust the results
    # dictionary(result_dic) to determine if classifier correctly classified
    # images as 'a dog' or 'not a dog'. This demonstrates if the model can
    # correctly classify dog images as dogs (regardless of breed)
    adjust_results4_isadog()

    # TODO: 6. Define calculates_results_stats() function to calculate
    # results of run and puts statistics in a results statistics
    # dictionary (results_stats_dic)
    results_stats_dic = calculates_results_stats()

    # TODO: 7. Define print_results() function to print summary results, 
    # incorrect classifications of dogs and breeds if requested.
    print_results()

    end_time = time()

    tot_time = start_time - end_time
    print("\n** Total Elapsed Runtime:", tot_time)



# TODO: 2.-to-7. Define all the function below. Notice that the input

def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object. 
     3 command line arguments are created:
       dir - Path to the pet image files(default- 'pet_images/')
       arch - CNN model architecture to use for image classification(default-
              pick any of the following vgg, alexnet, resnet)
       dogfile - Text file that contains all labels associated to dogs(default-
                'dognames.txt'
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='pet_images/',
                        help='path to the folder of pet images')
    parser.add_argument('--arch', type=str, default='vgg', help='chosen model')
    parser.add_argument('--dogfile', type=str, default='dogname', help='dog u wanna search')
    # Assigns variable in_args to parse_args()

    # Accesses values of Argument 1 by printing it
    #print("Argument 1:", in_args.dir)
    return parser.parse_args()




def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels based upon the filenames of the image 
    files. Reads in pet filenames and extracts the pet image labels from the 
    filenames and returns these labels as petlabel_dic. This is used to check 
    the accuracy of the image classifier model.
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by pre trained CNN models (string)
    Returns:
     petlabels_dic - Dictionary storing image filename (as key) and Pet Image
                     Labels (as value)  
    """
    in_files = listdir(image_dir)

    petlabels_dic = dict()

    for idx in range(0, len(in_files), 1):
        if in_files[idx][0] != ".":
            pet_label =  ""

            if in_files[idx] not in petlabels_dic:
                petlabels_dic[in_files[idx]] = pet_label

            else:
                print("warning: Duplicate files are there ", in_files[idx])
        return(petlabels_dic)
def classify_images(images_dir, petlabel_dic, model):
    """
    Creates classifier labels with classifier function, compares labels, and 
    creates a dictionary containing both labels and comparison of them to be
    returned.
     PLEASE NOTE: This function uses the classifier() function defined in 
     classifier.py within this function. The proper use of this function is
     in test_classifier.py Please refer to this program prior to using the 
     classifier() function to classify images in this function. 
     Parameters: 
      images_dir - The (full) path to the folder of images that are to be
                   classified by pretrained CNN models (string)
      petlabel_dic - Dictionary that contains the pet image(true) labels
                     that classify what's in the image, where its key is the
                     pet image filename & its value is pet image label where
                     label is lowercase with space between each word in label 
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
     Returns:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    +idx 2 = 1/0 (int)   where 1 = match between pet image and 
                    classifer labels and 0 = no match between labels
    """
    results_dic = dict()
    for key in petlabel_dic:
        model_label = classifier(images_dir+key, model)
        model_label = model_label.lower()
        model_label = model_label.strip()

        truth = petlabel_dic[key]
        found = model_label.find(truth)

    if found >= 0:
        if ((found == 0 and len(truth) == len(model_label)) or
                (((found == 0) or (model_label[found - 1] == " ")) and
                 ((found + len(truth) == len(model_label)) or
                  (model_label[found + len(truth): found + len(truth) + 1] in
                   (",", " "))
                 )
                )
        ):
            # found label as stand-alone term (not within label)
            if key not in results_dic:
                results_dic[key] = [truth, model_label, 1]

        # found within a word/term not a label existing on its own
        else:
            if key not in results_dic:
                results_dic[key] = [truth, model_label, 0]

    # if not found set results dictionary with match=0(no)
    else:
        if key not in results_dic:
            results_dic[key] = [truth, model_label, 0]
    return results_dic


def adjust_results4_isadog(results_dic, dogsfile):
    """
    Adjusts the results dictionary to determine if classifier correctly 
    classified images 'as a dog' or 'not a dog' especially when not a match. 
    Demonstrates if model architecture correctly classifies dog images even if
    it gets dog breed wrong (not a match).
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    --- where idx 3 & idx 4 are added by this function ---
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
     dogsfile - A text file that contains names of all dogs from ImageNet 
                1000 labels (used by classifier model) and dog names from
                the pet image files. This file has one dog name per line.
                Dog names are all in lowercase with spaces separating the 
                distinct words of the dogname. This file should have been
                passed in as a command line argument. (string - indicates 
                text file's name)
    Returns:
           None - results_dic is mutable data type so no return needed.
    """           
    dognames_dic = dict()

    with open(dogsfile, "r") as infile:
        line = infile.readline()

        while line != "":
            line = line.rstrip()

            if line not in dognames_dic:
                dognames_dic[line] =  1
            else:
                print("warning: Duplicate dognames ", line )

            line = infile.readline()

    for key in results_dic:

        if results_dic[key][0] in dognames_dic:
            if results_dic[key][1] in dognames_dic:
                results_dic[key].extend((1,1))
            else:
                results_dic[key].extend((1,0))

        else:
            if results_dic[key][1] in dognames_dic:
                results_dic[key].extends((0,1))

            else:
                results_dic[key].extends((0,0))


def calculates_results_stats(results_dic):
    """
    Calculates statistics of the results of the run using classifier's model
    architecture on classifying images. Then puts the results statistics in a
    dictionary (results_stats) so that it's returned for printing as to help
    the user to determine the 'best' model for classifying images. Note that
    the statistics calculated as the results are either percentages or counts.
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
    Returns:
     results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value
    """
    # creates empty dictionary for results_stats
    results_stats = dict()

    # Sets all counters to initial values of zero so that they can
    # be incremented while processing through the images in results_dic
    results_stats['n_dogs_img'] = 0
    results_stats['n_match'] = 0
    results_stats['n_correct_dogs'] = 0
    results_stats['n_correct_notdogs'] = 0
    results_stats['n_correct_breed'] = 0

    # process through the results dictionary
    for key in results_dic:
        # Labels Match Exactly
        if results_dic[key][2] == 1:
            results_stats['n_match'] += 1

        # Pet Image Label is a Dog AND Labels match- counts Correct Breed
        if sum(results_dic[key][2:]) == 3:
            results_stats['n_correct_breed'] += 1

        # Pet Image Label is a Dog - counts number of dog images
        if results_dic[key][3] == 1:
            results_stats['n_dogs_img'] += 1

            # Classifier classifies image as Dog (& pet image is a dog)
            # counts number of correct dog classifications
            if results_dic[key][4] == 1:
                results_stats['n_correct_dogs'] += 1

        # Pet Image Label is NOT a Dog
        else:
            # Classifier classifies image as NOT a Dog(& pet image isn't a dog)
            # counts number of correct NOT dog clasifications.
            if results_dic[key][4] == 0:
                results_stats['n_correct_notdogs'] += 1

    # Calculates run statistics (counts & percentages) below that are calculated
    # using counters from above.

    # calculates number of total images
    results_stats['n_images'] = len(results_dic)

    # calculates number of not-a-dog images using - images & dog images counts
    results_stats['n_notdogs_img'] = (results_stats['n_images'] -
                                      results_stats['n_dogs_img'])

    # Calculates % correct for matches
    results_stats['pct_match'] = (results_stats['n_match'] /
                                  results_stats['n_images']) * 100.0

    # Calculates % correct dogs
    results_stats['pct_correct_dogs'] = (results_stats['n_correct_dogs'] /
                                         results_stats['n_dogs_img']) * 100.0

    # Calculates % correct breed of dog
    results_stats['pct_correct_breed'] = (results_stats['n_correct_breed'] /
                                          results_stats['n_dogs_img']) * 100.0

    # Calculates % correct not-a-dog images
    # Uses conditional statement for when no 'not a dog' images were submitted
    if results_stats['n_notdogs_img'] > 0:
        results_stats['pct_correct_notdogs'] = (results_stats['n_correct_notdogs'] /
                                                results_stats['n_notdogs_img']) * 100.0
    else:
        results_stats['pct_correct_notdogs'] = 0.0

    # returns results_stast dictionary
    return results_stats


def print_results(results_dic, results_stats, model,
                  print_incorrect_dogs=False, print_incorrect_breed=False):
    """
    Prints summary results on the classification and then prints incorrectly
    classified dogs and incorrectly classified dog breeds if user indicates
    they want those printouts (use non-default values)
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
      results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
      print_incorrect_dogs - True prints incorrectly classified dog images and
                             False doesn't print anything(default) (bool)
      print_incorrect_breed - True prints incorrectly classified dog breeds and
                              False doesn't print anything(default) (bool)
    Returns:
           None - simply printing results.
    """
    # Prints summary statistics over the run
    print("\n\n*** Results Summary for CNN Model Architecture", model.upper(),
          "***")
    print("%20s: %3d" % ('N Images', results_stats['n_images']))
    print("%20s: %3d" % ('N Dog Images', results_stats['n_dogs_img']))
    print("%20s: %3d" % ('N Not-Dog Images', results_stats['n_notdogs_img']))

    # Prints summary statistics (percentages) on Model Run
    print(" ")
    for key in results_stats:
        if key[0] == "p":
            print("%20s: %5.1f" % (key, results_stats[key]))

    # IF print_incorrect_dogs == True AND there were images incorrectly
    # classified as dogs or vice versa - print out these cases
    if (print_incorrect_dogs and
            ((results_stats['n_correct_dogs'] + results_stats['n_correct_notdogs'])
             != results_stats['n_images'])
    ):
        print("\nINCORRECT Dog/NOT Dog Assignments:")

        # process through results dict, printing incorrectly classified dogs
        for key in results_dic:

            # Pet Image Label is a Dog - Classified as NOT-A-DOG -OR-
            # Pet Image Label is NOT-a-Dog - Classified as a-DOG
            if sum(results_dic[key][3:]) == 1:
                print("Real: %-26s   Classifier: %-30s" % (results_dic[key][0],
                                                           results_dic[key][1]))

    # IF print_incorrect_breed == True AND there were dogs whose breeds
    # were incorrectly classified - print out these cases
    if (print_incorrect_breed and
            (results_stats['n_correct_dogs'] != results_stats['n_correct_breed'])
    ):
        print("\nINCORRECT Dog Breed Assignment:")

        # process through results dict, printing incorrectly classified breeds
        for key in results_dic:

            # Pet Image Label is-a-Dog, classified as-a-dog but is WRONG breed
            if (sum(results_dic[key][3:]) == 2 and
                    results_dic[key][2] == 0):
                print("Real: %-26s   Classifier: %-30s" % (results_dic[key][0],
                                                           results_dic[key][1]))


# Call to main function to run the program
if __name__ == "__main__":
    main()
