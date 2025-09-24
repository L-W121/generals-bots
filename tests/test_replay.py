from datasets import load_dataset

# 1. Load the dataset directly from the Hugging Face Hub
#    You may need to log in first: huggingface-cli login
print("Loading dataset...")
dataset = load_dataset("strakammm/generals_io_replays")

# The dataset is loaded into a DatasetDict, we'll use the 'train' split
train_dataset = dataset['train']

# 2. Print the total number of replays in the dataset
print(f"\nTotal number of replays: {len(train_dataset)}")
print(train_dataset[6])
# 3. Iterate over the first few replays to showcase the data
'''
print("\n--- Showcasing first 5 replays ---")
for i in range(5):
    replay = train_dataset[i]
    
    # Get the players and the number of moves
    players = replay['usernames']
    num_moves = len(replay['moves'])
    
    print(f"\nReplay {i+1}:")
    print(f"  Players: {players[0]} vs {players[1]}")
    print(f"  Total moves: {num_moves}")

print("\n------------------------------------")
'''