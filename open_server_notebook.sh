#!/bin/bash

fr() {
	ssh iamjli@fraenkel-node${1:-9}.csbi.mit.edu
}

node='18'
tmuxname='GSD_jupyter'
serverport='8889'
localport='8890'

# SSH into specified node via node17
fr 17
fr $node

# Open tmux session
tmux new-session -A -s $tmuxname

# cd to mirrored directory
cd /nfs/latdata/iamjli/projects/GSD

# Spin up virtual environment
source venv/bin/activate

# Fires-up a Jupyter notebook by supplying a specific port
jupyter notebook --no-browser --port=$serverport

# Back to local
exit
exit

# Forwards port $1 into port $2 and listens to it
ssh -N -f -L localhost:${localport}:localhost:${serverport} iamjli@fraenkel-node${node}.csbi.mit.edu

echo localhost:${localport}






fr 17

tmux new -s GSD_jupyter

# tmux a -t GSD_jupyter

cd /nfs/latdata/iamjli/projects/GSD
source venv/bin/activate
jupyter notebook --no-browser --port=8889
exit


ssh -N -f -L localhost:8890:localhost:8889 iamjli@fraenkel-node17.csbi.mit.edu


# kill port listening
lsof -n -i4TCP:8890 | grep LISTEN | awk '{ print $2 }' | xargs kill