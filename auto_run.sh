#!/bin/bash

# Function to generate a random number
generate_random_seed() {
    echo $((RANDOM))
}

# Main function
main() {
    echo "Running" > log.txt
    while true; do
        seed=140
        echo "Executing python main_pvic.py --seed $seed" >> log.txt
        python main_pvic.py --pretrained data/detr-r50-hicodet.pth --output-dir data/pvic-detr-r50-hicodet --world-size 4 --batch-size 4 --seed $seed

        # Check if command execution was successful
        if [ $? -eq 0 ]; then
            echo "Command execution successful!" >> log.txt
            break
        else
            echo "Command execution failed. Retrying with a random seed." >> log.txt
            seed=$(generate_random_seed)
        fi
    done
}

# Call the main function
main
