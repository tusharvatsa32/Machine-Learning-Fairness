import subprocess
import os,stat,sys
import shutil
from shutil import copyfile
from sys import exit

subprocess.check_call([sys.executable,'-m','pip','install','seaborn'])
subprocess.check_call([sys.executable,'-m','pip','install','plotly'])