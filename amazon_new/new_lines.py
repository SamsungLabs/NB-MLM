import sys

file_path = sys.argv[1]
input_file = open(file_path, 'r')
output_file = open(file_path.replace('.texts', '.newlined.texts'), 'w')

for line in input_file:
    output_file.write(line)
    output_file.write('\n')
output_file.close()
input_file.close()
