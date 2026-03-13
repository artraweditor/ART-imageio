import subprocess, sys

subprocess.run(['dnglab', 'convert', sys.argv[1], sys.argv[2]], check=True)
