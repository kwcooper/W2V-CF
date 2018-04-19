#!/bin/bash
#PBS -l nodes=1:ppn=16 
#PBS -l walltime=00:60:00
cd /N/u/kc42/Karst/W2V-CI_HPSC 
python3 runBiasedModel_HPSC_V2.py  
