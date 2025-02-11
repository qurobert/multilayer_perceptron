#!/bin/bash

# Nombre d'itérations
N=50

# Compteur de succès (valeur < 0.08)
success_count=0

# Boucle d'exécution
for ((i=1; i<=N; i++)); do
    echo "Iteration $i/$N"

    # Exécute Python et capture la sortie
    result=$(python3 src/main.py sklearn 2>/dev/null | grep -Eo '[0-9]+\.[0-9]+' | tail -1)

    # Vérifie si la sortie est bien un nombre
    if [[ $result =~ ^[0-9]+\.[0-9]+$ ]]; then
        # Vérifie si le nombre est inférieur à 0.08
        if (( $(echo "$result < 0.08" | bc -l) )); then
          echo "Succès : $result"
            ((success_count++))
        fi
    fi
done

# Affiche le résultat final
echo "Nombre de succès (< 0.08) : $success_count sur $N essais"
