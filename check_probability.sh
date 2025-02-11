#!/bin/bash

# Nombre d'itérations maximum
N=10

success_count_me=0
success_count_sklearn=0
# Boucle d'exécution
for ((i=1; i<=N; i++)); do
    echo "Iteration $i/$N"

    # Exécute Python et capture la sortie
    python src/main.py split > /dev/null

    result_me=$( (python ./src/main.py train --disable-plot --disable_pbar && python ./src/main.py predict 2>/dev/null) | grep -Eo 'loss: [0-9]+\.[0-9]+' | sed 's/.*loss: //' | tail -1)
    result_sklearn=$( (python ./src/main.py sklearn 2>/dev/null) | grep -Eo '[0-9]+\.[0-9]+' | tail -1)
    echo "My result is $result_me"
    echo "Sklearn result is $result_sklearn"
#         Vérifie si le nombre est inférieur à 0.08
    if (( $(echo "$result_me <= 0.08" | bc -l) )); then
      echo "(me) + 1"
      success_count_me=$((success_count_me+1))
    fi
    if (( $(echo "$result_sklearn <= 0.08" | bc -l) )); then
      echo "(sklearn) + 1"
      success_count_sklearn=$((success_count_sklearn+1))
    fi
done

echo "Success rate (me): $success_count_me/$N"
echo "Success rate (sklearn): $success_count_sklearn/$N"
