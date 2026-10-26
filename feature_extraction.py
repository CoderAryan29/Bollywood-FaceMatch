'''
import os
import pickle

base_path = 'data/facematcher data/Bollywood_celeb_face_localized'

filenames = []

# All three dataset folders
dataset_folders = [
    'bollywood_celeb_faces2',
    'bollywood_celeb_faces_0',
    'bollywood_celeb_faces_1'
]

# Collect image paths from every celebrity folder
for dataset in dataset_folders:
    dataset_path = os.path.join(base_path, dataset)

    for actor in os.listdir(dataset_path):
        actor_path = os.path.join(dataset_path, actor)

        if not os.path.isdir(actor_path):
            continue

        for file in os.listdir(actor_path):
            file_path = os.path.join(actor_path, file)

            if (
                os.path.isfile(file_path)
                and file.lower().endswith(('.jpg', '.jpeg', '.png'))
            ):
                filenames.append(file_path)

print("Total images collected:", len(filenames))

# Save the paths into a pickle file
with open('filenames.pkl', 'wb') as file:
    pickle.dump(filenames, file)

print("All filenames saved to filenames.pkl")
'''










