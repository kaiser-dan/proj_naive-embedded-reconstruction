#!/usr/bin/env bash

# >>> Setup >>>
# Set up log file
echo "Setting up logfile..."
LOGFILE=log_create_LFR_caches.log
touch $LOGFILE
echo "Last run: $(date)" | tee -a $LOGFILE

# Set up additional files
SC=cache_LFRs.py

# >>> Preface >>>
echo "Creating synthetic networks and caching various embeddings..." | tee -a $LOGFILE

# >>> Computations >>>
# LE embedding
echo "LE embedding, 1000 nodes, PC false" | tee -a $LOGFILE
python $SC LE 128 | tee -a $LOGFILE

echo "LE embedding, 1000 nodes, PC true" | tee -a $LOGFILE
python $SC LE 128 --percomponent | tee -a $LOGFILE

# N2V embedding
echo "N2V embedding, 1000 nodes, PC false" | tee -a $LOGFILE
python $SC N2V 128 | tee -a $LOGFILE

echo "N2V embedding, 1000 nodes, PC true" | tee -a $LOGFILE
python $SC N2V 128 --percomponent | tee -a $LOGFILE

# >>> Postface >>>
echo "Completed default caching: $(date)" | tee -a $LOGFILE
echo "Have a nice day!"
