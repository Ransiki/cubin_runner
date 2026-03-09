import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--dir', required=True, help="Path to CUDA toolkit extension")
args = parser.parse_args()

backup_file_name = ''
new_props_file_name = ''
patched = False
for f in os.listdir(args.dir):
    if f.endswith('.props'):
        print('found props file: ',f)
        if not (f.startswith('CUDA 10.2') or f.startswith('CUDA 11.4')):
            break
        new_props_file_name = f
        with open(Path(args.dir)/f, 'r') as props_file:
            lines = props_file.readlines()
            for line in lines:
                if line.find('DriverApiCommandLineTemplate') != -1 and line.find('[CompileOut]') != -1:
                    if line.find('[Defines]') != -1:
                        patched = True

        if patched:
            break

        backup_file_name = f + '.backup'
        # move file
        os.rename(Path(args.dir)/f, Path(args.dir)/backup_file_name)

        with open(Path(args.dir)/new_props_file_name, 'w') as new_file:
            for line in lines:
                if line.find('DriverApiCommandLineTemplate') != -1 and line.find('[CompileOut]') != -1:
                    if line.find('[Defines]') == -1:
                        print('BEFORE: ', line)
                        line = line.replace('[CompileOut]', '[Defines] [CompileOut]') 
                        print('AFTER: ', line)
                        new_file.write(line)
                        patched = True
                else:
                    new_file.write(line)

if patched:
    print('patching complete')

