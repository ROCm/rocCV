# ##############################################################################
# Copyright (c)  - 2024 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ##############################################################################

import os
import sys
import argparse
import platform
import traceback
if sys.version_info[0] < 3:
    import commands
else:
    import subprocess

libraryName = "rocCV"

__copyright__ = f"Copyright (c) 2024, AMD ROCm {libraryName}"
__version__ = "0.1.0"
__email__ = "mivisionx.support@amd.com"
__status__ = "Shipping"

# ANSI Escape codes for info messages
TEXT_WARNING = "\033[93m\033[1m"
TEXT_ERROR = "\033[91m\033[1m"
TEXT_INFO = "\033[1m"
TEXT_DEFAULT = "\033[0m"

def info(msg):
    print(f"{TEXT_INFO}INFO:{TEXT_DEFAULT} {msg}")

def warn(msg):
    print(f"{TEXT_WARNING}WARNING:{TEXT_DEFAULT} {msg}")

def error(msg):
    print(f"{TEXT_ERROR}ERROR:{TEXT_DEFAULT} {msg}")

# error check for calls
def ERROR_CHECK(waitval):
    if(waitval != 0): # return code and signal flags
        error('ERROR_CHECK failed with status:'+str(waitval))
        traceback.print_stack()
        status = ((waitval >> 8) | waitval) & 255 # combine exit code and wait flags into single non-zero byte
        exit(status)

def install_packages(linuxFlag, linuxSystemInstall, linuxSystemInstall_check, package_list):
    cmd_str = 'sudo ' + linuxFlag + ' ' + linuxSystemInstall + \
        ' ' + linuxSystemInstall_check+' install '
    for i in range(len(package_list)):
        cmd_str += package_list[i] + " "
    ERROR_CHECK(os.system(cmd_str))

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--rocm_path', 	type=str, default='/opt/rocm',
                    help='ROCm Installation Path - optional (default:/opt/rocm) - ROCm Installation Required')

args = parser.parse_args()
ROCM_PATH = args.rocm_path

if "ROCM_PATH" in os.environ:
    ROCM_PATH = os.environ.get('ROCM_PATH')
info("ROCm PATH set to -- " + ROCM_PATH)

# check ROCm installation
if os.path.exists(ROCM_PATH):
    info("ROCm Installation Found -- "+ROCM_PATH)
    # os.system('echo ROCm Info -- && '+ROCM_PATH+'/bin/rocminfo')
else:
    warn(
        "If ROCm installed, set ROCm Path with \"--rocm_path\" option for full installation [Default:/opt/rocm]")
    error(f"{libraryName} Setup requires ROCm install")
    exit(-1)

# get platform info
platformInfo = platform.platform()

# sudo requirement check
sudoLocation = ''
userName = ''
if sys.version_info[0] < 3:
    status, sudoLocation = commands.getstatusoutput("which sudo")
    if sudoLocation != '/usr/bin/sudo':
        status, userName = commands.getstatusoutput("whoami")
else:
    status, sudoLocation = subprocess.getstatusoutput("which sudo")
    if sudoLocation != '/usr/bin/sudo':
        status, userName = subprocess.getstatusoutput("whoami")

# check os version
os_info_data = 'NOT Supported'
if os.path.exists('/etc/os-release'):
    with open('/etc/os-release', 'r') as os_file:
        os_info_data = os_file.read().replace('\n', ' ')
        os_info_data = os_info_data.replace('"', '')

# setup for Linux
linuxSystemInstall = ''
linuxCMake = 'cmake'
linuxSystemInstall_check = ''
linuxFlag = ''
sudoValidate = 'sudo -v'
osUpdate = ''
if "centos" in os_info_data or "redhat" in os_info_data:
    linuxSystemInstall = 'yum -y'
    linuxSystemInstall_check = '--nogpgcheck'
    osUpdate = 'makecache'
    if "VERSION_ID=8" in os_info_data:
        platformInfo = platformInfo+'-centos-8-based'
    elif "VERSION_ID=9" in os_info_data:
        platformInfo = platformInfo+'-centos-9-based'
    else:
        platformInfo = platformInfo+'-centos-undefined-version'
elif "Ubuntu" in os_info_data:
    linuxSystemInstall = 'apt-get -y'
    linuxSystemInstall_check = '--allow-unauthenticated'
    osUpdate = 'update'
    linuxFlag = '-S'
    if "VERSION_ID=22" in os_info_data:
        platformInfo = platformInfo+'-ubuntu-22'
    elif "VERSION_ID=24" in os_info_data:
        platformInfo = platformInfo+'-ubuntu-24'
    else:
        platformInfo = platformInfo+'-ubuntu-undefined-version'
elif "SLES" in os_info_data:
    linuxSystemInstall = 'zypper -n'
    linuxSystemInstall_check = '--no-gpg-checks'
    osUpdate = 'refresh'
    platformInfo = platformInfo+'-sles'
elif "Mariner" in os_info_data:
    linuxSystemInstall = 'tdnf -y'
    linuxSystemInstall_check = '--nogpgcheck'
    platformInfo = platformInfo+'-mariner'
    osUpdate = 'makecache'
else:
    print("\rocCV Setup on "+platformInfo+" is unsupported\n")
    print("\nrocCV Setup Supported on: Ubuntu 20/22, RedHat 8/9, & SLES 15\n")
    exit(-1)

# rocCV Setup
info(f"{libraryName} Setup on: "+platformInfo)
info(f"{libraryName} Dependencies Installation with roccv-setup.py V-"+__version__)

if userName == 'root':
    ERROR_CHECK(os.system(linuxSystemInstall+' '+osUpdate))
    ERROR_CHECK(os.system(linuxSystemInstall+' install sudo'))

# debian packages
coreDebianPackages = [
    'libdlpack-dev',
    'python3-dev',
    'python3-pip',
    'python3-opencv'
]

# rpm packages
coreRpmPackages = [
    'python3-devel'
]

# common packages
coreCommonPackages = [
    'cmake',
    'python3-pip',
    'python3-pytest',
    'python3-numpy'
]

# rocm dependencies
rocmPackages = [
    'rocm-hip-runtime-dev'
]

# pip3 packages
pip3Packages = [
    'pybind11~=2.12',
    'wheel~=0.30'
]

pip3RpmPackages = [
    'opencv-python~=4.10'
]

info(f"{libraryName} Dependencies Installation with roccv-setup.py V-"+__version__)

# update
ERROR_CHECK(os.system('sudo '+linuxFlag+' '+linuxSystemInstall +' '+linuxSystemInstall_check+' '+osUpdate))

# rocCV Core - Requirements
ERROR_CHECK(os.system('sudo '+sudoValidate))
if "ubuntu" in platformInfo:
    install_packages(linuxFlag, linuxSystemInstall, linuxSystemInstall_check, coreDebianPackages)
else:
    install_packages(linuxFlag, linuxSystemInstall, linuxSystemInstall_check, coreRpmPackages)

install_packages(linuxFlag, linuxSystemInstall, linuxSystemInstall_check, coreCommonPackages)

# rocCV - ROCm packages
install_packages(linuxFlag, linuxSystemInstall, linuxSystemInstall_check, rocmPackages)

for i in range(len(pip3Packages)):
    ERROR_CHECK(os.system('pip3 install '+ pip3Packages[i]))
if not("ubuntu" in platformInfo):
    for i in range(len(pip3RpmPackages)):
        ERROR_CHECK(os.system('pip3 install '+ pip3RpmPackages[i]))

info(f"{libraryName} Dependencies Installed with roccv-setup.py V-"+__version__)
