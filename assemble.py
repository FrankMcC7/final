import os

def join_files(output_file, part_files):
    with open(output_file, 'wb') as outfile:
        for part_file in part_files:
            with open(part_file, 'rb') as infile:
                outfile.write(infile.read())

part_files = sorted([f for f in os.listdir('.') if f.startswith('pytorch_model_part_')])
join_files('pytorch_model.bin', part_files)
