#!/usr/bin/python
#
###################################################################################################
#
# Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
###################################################################################################
#
#  {$nv-release-never file} - guard to avoid releasing the release script
#
# Script to construct a CUTLASS release from a Git repository.

import os
import sys
import shlex
import shutil
import argparse
import subprocess

###################################################################################################

#
def str_to_bool(x):
    """ Returns false if the string is 0, false, or False. Otherwise, returns True."""
    return x is not None and x not in ('0', 'false', 'False')

#
class Options:
  def __init__(self):

    self.dst_path = '.'
    self.src_path = '../cutlass_public_rc'
    self.dry_run = False
    self.verbose = True

    self.parser = argparse.ArgumentParser(description='CUTLASS release automation')
    self.parser.add_argument('--dst_path', dest='dst_path', metavar='<str>', type=str,
      default=self.dst_path, help='Path to directory in which release directory is created.')
    self.parser.add_argument('--src_path', dest='src_path', metavar='<str>', type=str,
      default=self.src_path, help='Path to CUTLASS project directory.')
    self.parser.add_argument('--dry_run', dest='dry_run', metavar='<bool>', type=str,
      default=str(self.dry_run), help="If true, no copies, creations, or git modifications are made.")
    self.parser.add_argument('--verbose', dest='verbose', metavar='<bool>', type=str,
      default=str(self.verbose), help="Prints verbose logging of filtering operations.")
    self.parser.add_argument('--include-ci', dest='include_ci', metavar='<bool>', type=str,
      default="false", help="If true, CI scripts are included")

    self.args = self.parser.parse_args()

    self.dst_path = self.args.dst_path
    self.src_path = self.args.src_path
    self.dry_run = str_to_bool(self.args.dry_run)
    self.verbose = str_to_bool(self.args.verbose)
    self.include_ci = str_to_bool(self.args.include_ci)

###################################################################################################

#
class CutlassSource:
  #
  def __init__(self, dir_root, blacklist = [
      '.git',
      '.gitignore',
      '.gitmodules',
      'docs',
      'python/docs'
    ]):

    self.dir_root = dir_root

    # Paths to avoid modifying in destination or recursing through in source
    self.blacklist = blacklist

    self.keys = set()
    self.listing = []
    self._visit_item('.')

  #
  def __iter__(self):
    return self.listing.__iter__()

  #
  def __contains__(self, item):
    return item in self.keys

  #
  def _normalize(self, item):
    normalized = os.path.normpath(item)
    if normalized.startswith('./') or normalized.startswith('.\\'):
      normalized = normalized[2:]
    return normalized

  #
  def _append_item(self, item):
    item = self._normalize(item)
    self.listing.append(item)
    self.keys.add(item)

  #
  def _visit_dir(self, path):
    self._append_item(path)
    dir_path = os.path.join(self.dir_root, path)
    for item in os.listdir(dir_path):
      if item != '.' and item != '..':
        self._visit_item(os.path.join(path, item))

  #
  def _visit_file(self, path):
    self._append_item(path)

  #
  def _visit_item(self, path):
    if self._normalize(path) in self.blacklist:
      return

    item_path = os.path.join(self.dir_root, path)
    if os.path.isfile(item_path):
      self._visit_file(path)
    elif os.path.isdir(item_path):
      self._visit_dir(path)

###################################################################################################

#
class Mirror:
  ''' Class to mirror a repository'''

  #
  def __init__(self, options = Options()):
    self.options = options

    blacklist = [
      '.git',
      '.gitignore',
      '.gitmodules',
      'docs',
      'python/docs'
    ]

    if not self.options.include_ci:
      blacklist += ['Jenkinsfile', 'bloom', 'tools/scripts', 'compiler_testlists']

    self.dst_listing = CutlassSource(self.options.dst_path, blacklist)
    self.src_listing = CutlassSource(self.options.src_path, blacklist)

  #
  def _cmd(self, cmd, always = False):
    if self.options.verbose:
      print(cmd)
    if always or not self.options.dry_run:
      args = shlex.split(cmd)
      return subprocess.run(args)
    return None

  #
  def _remove_item(self, path):
    ''' Performs `git rm` on a directory or file. '''
    cmd = "git rm -rf \"%s\"" % path
    return self._cmd(cmd)

  #
  def _add_item(self, path):
    ''' Performs `git add` on a directory or file. '''
    cmd = "git add \"%s\"" % path
    return self._cmd(cmd)

  #
  def _clear_destination(self):
    ''' Recursively enumerates destination paths and removes them if
        they do not exist in the source directory. '''
    if self.options.verbose:
      print('# Removing unwanted items in destination repository.')

    for item in self.dst_listing:
      if item not in self.src_listing:
        self._remove_item(os.path.join(self.options.dst_path, item))

  #
  def _mirror_source(self):
    ''' Recursively enumerates paths in source directory and copies them
        to destination directory. Runs git-add on them. '''

    if self.options.verbose:
      print('# Adding modified items from source directory.')

    for item in self.src_listing:
      is_new = item not in self.dst_listing

      src_path = os.path.join(self.options.src_path, item)
      dst_path = os.path.join(self.options.dst_path, item)

      if os.path.isfile(src_path):
        if self.options.verbose:
          print('# copying %s' % item)
        if not self.options.dry_run:

          # copy file
          shutil.copyfile(src_path, dst_path)

          # replicate privileges
          if os.path.isfile(dst_path):
            os.chmod(dst_path, os.stat(src_path).st_mode)

      elif os.path.isdir(src_path) and is_new:
        if self.options.verbose:
          print('# mkdir %s' % item)
        if not self.options.dry_run:
          os.mkdir(dst_path)

      if is_new:
        self._add_item(dst_path)

  #
  def is_cutlass_repository(self, path):
    ''' Sanity check to make sure the indicated path contains several basic
        CUTLASS project compoents.  '''
    expected = [
      ('include/cutlass', os.path.isdir),
      ('tools', os.path.isdir),
      ('CMakeLists.txt', os.path.isfile),
    ]
    for item, func in expected:
      item_path = os.path.join(path, item)
      if not func(item_path):
        print('Error - could not find %s' % item_path)
        return False
    return True

  #
  def is_git_repository(self):
    ''' Sanity checking to make sure the destination is a git repository'''
    completed = self._cmd('git status', True)
    return completed.returncode == 0

  #
  def is_sane(self):
    if not self.is_git_repository():
      print('This script must be run from within the destination git repository.')
      return False

    repositories = [
      (self.options.dst_path, 'Destination'),
      (self.options.src_path, 'Source')
    ]

    for path, name in repositories:
      if not self.is_cutlass_repository(path):
        print('Error - %s path "%s" must be a CUTLASS repository.' % (name, path))
        return False

    return True

  #
  def run(self):
    self._clear_destination()
    self._mirror_source()
    return 0

###################################################################################################

#
def Main():
  mirror = Mirror()
  if not mirror.is_sane():
    return -1
  return mirror.run()

###################################################################################################

#
if __name__ == "__main__":
  sys.exit(Main())

###################################################################################################
