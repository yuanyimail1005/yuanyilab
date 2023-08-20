import sys
import os
import os.path

# loop through all files in the dictionary
def remove_space_dictionary(dict_path):
    for file in os.listdir(dict_path):
        # get the full file path
        file_path = os.path.join(dict_path, file)
        read_file_and_remove_tailing_space(file_path)

def read_file_and_remove_tailing_space(file_path):
    # open the file in read mode
    with open(file_path, "r") as f:
        # read the file content as a list of lines
        lines = f.readlines()
    # open the file in write mode
    with open(file_path, "w") as f:
        # loop through each line
        for line in lines:
            # remove the tailing space and write the line to the file
            f.write(line.rstrip() + "\n")

def help():
    print("./remove_trailing_space.py dictionary_or_file_name")

def main():
    # This is the main body of the program.
    input_str = sys.argv[1]

    if os.path.isdir(input_str):
        print("remove all tailing space from files under ", input_str)
        remove_space_dictionary(input_str)
    elif os.path.isfile(input_str):
        print("remove all tailing space from file ", input_str)
        read_file_and_remove_tailing_space(input_str)
    else:
        print(input_str)
        help()

if __name__ == "__main__":
    main()