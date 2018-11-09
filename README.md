# GSD

Graph-Structured Decomposition is a form of matrix factorization where the components in the signals matrix are structured with respect to a prior knowledge graph. 



# Setup Jupyter notebook on server

## Initialization

SSH into server.

```
ssh iamjli@fraenkel-node17.csbi.mit.edu
```

Spin up tmux session. 

```
tmux new -s GSD_jupyter
```

Start notebook server without opening web browswer.

```
cd /nfs/latdata/iamjli/projects/GSD
source venv/bin/activate
jupyter notebook --no-browser --port=8889
```

Copy token if first time logging in, then detach from tmux session and close server connection.  

```
exit
```

Listen to server port via SSH tunnel.

```
ssh -N -f -L localhost:8890:localhost:8889 iamjli@fraenkel-node17.csbi.mit.edu
```

Jupyter notebook can now be accessed at http://localhost:8890. SSH tunnel will close occasionally if network connection is lost, but the notebook can still be accessed without having to go through the entire setup by running the last command again. 


## Handy commands

Access session

```
tmux a -t GSD_jupyter
```

Kill tmux session

```
tmux kill-session -t GSD_jupyter
```

Close port tunnel

```
lsof -n -i4TCP:8890 | grep LISTEN | awk '{ print $2 }' | xargs kill
```