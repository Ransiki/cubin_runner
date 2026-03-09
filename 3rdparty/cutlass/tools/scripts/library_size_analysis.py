
import sys
import os
import subprocess
import csv

#
# For each library,
#   1. list functions via cuobjdump -res-usage <path> and search for 'Function '
#   2. extract .fatbin via cuobjdump --extract-fatbin and observe sizes
#   3. record:
#      - .so name
#      - kernel count
#      - total bytes of .so
#      - total bytes of .fatbin(s)
#      - compressed code
#      - uncompressed code
#
# Usage:
#   $ cd build
#   $ python ../tools/scripts/library_size_analysis.py tools/library
#   $ ls results.csv

#
class Fatbin:
  def __init__(self, fatbin_path):
    self.fatbin_path = fatbin_path
    self.fatbin_name = os.path.basename(fatbin_path)
    self.kernels = []
    self.total_size = 0
    self.compressed_size = 0
    self.uncompressed_size = 0

    self.process_()

  # Gets size
  def process_(self):

    # Get size
    self.total_size = os.path.getsize(self.fatbin_path)

    # Get function count
    cmd = "cuobjdump -res-usage %s" % self.fatbin_path
    results = subprocess.run(cmd, shell=True, text=True, capture_output=True)

    if results.returncode != 0:
      print("Command `%s` exited with error %s" % (cmd, results.returncode))
      exit(results.returncode)

    function_prefix = ' Function '

    for line in results.stdout.split('\n'):
      if line.startswith(function_prefix):
        kernel_name = line[len(function_prefix):]
        self.kernels.append(kernel_name)

  #
  def kernel_count(self):
    return len(self.kernels)


#
class Library:
  def __init__(self, library_path):
    self.library_path = library_path
    self.library_directory, self.library_name = os.path.split(library_path)
    self.total_size   = os.path.getsize(library_path)
    self.fatbin_contents = []
    self.populate_fatbins_()

  #
  def populate_fatbins_(self):

    # extract fatbins
    cmd = "cuobjdump --extract-fatbin %s" % str(self.library_path)

    results = subprocess.run(cmd, shell=True, text=True, capture_output=True)

    if results.returncode != 0:
      print("Command `%s` exited with error %s" % (cmd, results.returncode))
      exit(results.returncode)

    omit = 'extracting fatbin to '

    for line in results.stdout.split('\n'):
      if len(line) >= len(omit):
        fatbin_path = line[len(omit):]
        if len(fatbin_path) and os.path.exists(fatbin_path):
          self.fatbin_contents.append(Fatbin(fatbin_path))

  #
  def kernel_count(self):
    count = 0
    for fatbin in self.fatbin_contents:
      count += fatbin.kernel_count()
    return count

  #
  def fatbin_size(self):
    count = 0
    for fatbin in self.fatbin_contents:
      count += fatbin.total_size
    return count

  #
  def avg_size_per_kernel(self):
    return self.total_size / self.kernel_count()

  #
  def avg_fatbin_size_per_kernel(self):
    return self.fatbin_size() / self.kernel_count()

  #
  def host_size(self):
    return self.total_size - self.fatbin_size()

  #
  def avg_host_size_per_kernel(self):
    return self.host_size() / self.kernel_count()

  #
  @staticmethod
  def columns():
    return [
      'library',
      'total_size',
      'kernels',
      'fatbin_size',
      'avg_size_per_kernel',
      'avg_fatbin_per_kernel',
      'avg_host_per_kernel',
    ]

  #
  def record(self):
    record = {
      'library': self.library_name,
      'total_size': str(self.total_size),
      'kernels': str(self.kernel_count()),
      'fatbin_size': str(self.fatbin_size()),
      'avg_size_per_kernel': str(self.avg_size_per_kernel()),
      'avg_fatbin_per_kernel': str(self.avg_fatbin_size_per_kernel()),
      'avg_host_per_kernel': str(self.avg_host_size_per_kernel()),
    }
    return record

  #
  def print(self):
    print("Library: %s" % (self.library_name))
    print("   total_size: %d" % (self.total_size))
    print("      kernels: %d" % (self.kernel_count()))
    print("  fatbin size: %d" % (self.fatbin_size()))

    print("    avg size per kernel: %f" % self.avg_size_per_kernel())
    print("  avg fatbin per kernel: %f" % self.avg_fatbin_size_per_kernel())
    print("    avg host per kernel: %f" % self.avg_host_size_per_kernel())

#
def Process(library_directory):

  libraries = []

  for item in os.listdir(library_directory):
    if item.startswith('libcutlass_') and item.endswith('.so') and item != 'libcutlass_library.so':
      library_path = os.path.join(library_directory, item)
      library = Library(library_path)
      libraries.append(library)

  return libraries

#
def Aggregate(libraries):

  total_size   = 0
  kernel_count = 0
  fatbin_total_size = 0
  host_total_size = 0

  for library in libraries:
    total_size += library.total_size
    kernel_count += library.kernel_count()
    fatbin_total_size += library.fatbin_size()
    host_total_size += library.host_size()

  avg_size_per_kernel        = (total_size / kernel_count if kernel_count else 0)
  avg_fatbin_size_per_kernel = (fatbin_total_size / kernel_count if kernel_count else 0)
  avg_host_size_per_kernel   = (host_total_size / kernel_count if kernel_count else 0)

  print("Summary:")
  print("    total size: %d" % total_size)
  print("  kernel count: %d" % kernel_count)
  print("   fatbin size: %d" % fatbin_total_size)
  print("     host size: %d" % host_total_size)
  print("")
  print("    avg size per kernel: %f" % avg_size_per_kernel)
  print("  avg fatbin per kernel: %f" % avg_fatbin_size_per_kernel)
  print("    avg host per kernel: %f" % avg_host_size_per_kernel)


#
def Main():

  library_directory = 'tools/library'
  if len(sys.argv) > 1:
    library_directory = sys.argv[1]

  libraries = Process(library_directory)

  print("")

  with open("results.csv", 'w', newline='') as results_file:
    writer = csv.DictWriter(results_file, fieldnames = Library.columns())

    writer.writeheader()
    for library in libraries:
      writer.writerow(library.record())

  print("")

  Aggregate(libraries)

#
if __name__ == '__main__':
  sys.exit(Main())
