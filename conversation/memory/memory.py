"""
Memory management module for user interactions in a conversational AI system.
This module provides functions to retrieve and update user memory in a SQLite database.
It includes functions to retrieve a user's memory, update it with new features, and retrieve or create a user based on face encodings.
==========
Functions:
- memory_retriever: Retrieve the memory of a user from the database.
- update_memory: Update the user's memory in the database with new memory objects.
- user_retriever: Retrieve or create a user based on face encodings.
"""

#--------------------------------------- Imports ---------------------------------------#

import json
import sqlite3

import torch

import numpy as np

from config.utils import suppress_stdout

from config.settings import SIMILARITY_THERESHOLD

#--------------------------------------- Functions ---------------------------------------#

def memory_retriever(user_id, conn):
    """  
    Retrieve the memory of a user from the database.
    This function fetches the user's memory as a list of dictionaries, each containing
    the name and value of each memory object.
    -----
    Args:
        user_id (str): The unique identifier for the user.
        conn (sqlite3.Connection): The database connection object.
    Returns:
        list: A list of dictionaries representing the user's memory.
    """
    if conn is None:
        raise ValueError("A database connection is required.")
    
    # Retrieve the user's features from its table in the database
    cursor = conn.cursor()
    cursor.execute(f"SELECT type, name, description, value FROM {user_id}")
    rows = cursor.fetchall()

    # Format the retrieved data into a list of dictionaries
    memory = []
    for row in rows:
        type, name, description, value = row
        memory.append({
            "type": type,
            "name": name,
            "description": description,
            "value": value
        })
    
    return memory


def update_memory(new_memory_object, current_user, conn, database=None):
    """
    Update the user's memory in the database with new memory objects.
    This function checks if the feature already exists and updates it if it does,
    or inserts it if it does not.
    -----
    Args:
        new_memory_object (list): A list of memory objects to be added or updated.
        current_user (str): The unique identifier for the user.
        conn (sqlite3.Connection): The database connection object.
    Returns:
        list: The updated memory of the user.   
    """
    new_conn = False
    if conn is None:
        conn = sqlite3.connect(database)
        new_conn = True

    primary_memory = new_memory_object.primary_features
    contextual_memory = new_memory_object.features

    cursor = conn.cursor()

    # Update primary features
    cursor.execute(f"SELECT name FROM {current_user} WHERE type = 'primary'")
    existing_primary_features = cursor.fetchall()
    existing_primary_features = [feature[0] for feature in existing_primary_features]
    # check if all the primary features are already in the database
    # If not, insert them
    primary_features = {
        "nom": "Les prénoms et noms de l'utilisateur.",
        "age": "L'âge de l'utilisateur.",
        "genre": "Le genre de l'utilisateur (masculin, féminin, non-binaire, etc.).",
        "preference_dialogue": "Les préférences de dialogue de l'utilisateur (formel, informel, humoristique, etc.)."
    }
    for feature, description in primary_features.items():
        if feature not in existing_primary_features:
            cursor.execute(
                f"INSERT INTO {current_user} (type, name, description, value, embeddings) VALUES (?, ?, ?, ?, ?)",
                ("primary", feature, description, '', None)
            )
            conn.commit()

    new_primary_features = [item for item in primary_memory if item.type == "primary"]

    for item in new_primary_features:
        if item.value:
            cursor.execute(
                f"UPDATE {current_user} SET value = ? WHERE name = ?",
                ((";").join(item.value) if isinstance(item.value, (list, tuple)) else str(item.value), item.name)
            )
            conn.commit()


    # Update contextual features
    cursor.execute(f"SELECT name FROM {current_user} WHERE type = 'contextual'")
    existing_features = cursor.fetchall()
    new_features = [item for item in contextual_memory if item.type == "contextual"]

    for item in new_features:
        if item.value:
            # Check if the feature already exists based on the name
            if item.name not in [feature[0] for feature in existing_features]:  # Feature does not exist
                # Insert new feature into the user's memory
                cursor.execute(
                    f"INSERT INTO {current_user} (type, name, description, value, embeddings) VALUES (?, ?, ?, ?, ?)",
                    (
                        item.type, 
                        item.name, 
                        item.description, 
                        (";").join(item.value), 
                        json.dumps(item.embeddings) if item.embeddings else None
                    )
                )
                conn.commit()
            else:  # Feature exists, update it
                cursor.execute(
                    f"UPDATE {current_user} SET value = ?, embeddings = ? WHERE name = ?",
                    (
                        (";").join(item.value) if isinstance(item.value, (list, tuple)) else str(item.value),
                        json.dumps(item.embeddings) if item.embeddings else None,
                        item.name
                    )
                )
                conn.commit()
    if new_conn:
        conn.close()
    return memory_retriever(current_user, conn)

 
def user_retriever(embedding, conn, database=None):
    """
    Retrieve or create a user based on face encodings.
    This function checks if the user already exists in the database based on the provided encodings.
    If a user is found, it returns the user ID and their memory.
    If no user is found, it creates a new user with the provided encodings and the associated table, and returns the new user ID and an empty memory.
    -----
    Args:
        embedding: Face embedding for the user (np.ndarray or torch.Tensor).
        conn (sqlite3.Connection): The database connection object.
    Returns:
        tuple: A tuple containing the user ID and their memory.
    """
    new_conn = False
    if conn is None:
        conn = sqlite3.connect(database)
        new_conn = True

    if conn is None:
        raise ValueError("A database connection is required.")
    cursor = conn.cursor()

    # Normalize embedding to a torch.Tensor, since it may come in as a
    # np.ndarray (e.g. InsightFace's face.embedding) or already as a tensor
    if isinstance(embedding, np.ndarray):
        embedding = torch.tensor(embedding, dtype=torch.float32)

    cursor.execute("SELECT user_id, embeddings FROM user_embeddings WHERE embeddings IS NOT NULL")
    embeddings_dict = {user_id: np.frombuffer(blob, dtype=np.float32) for user_id, blob in cursor.fetchall()}

    known_user_embeddings = list(embeddings_dict.values())
    if known_user_embeddings:
        device = embedding.device
        known_user_embeddings = torch.stack([
            torch.tensor(e, device=device, dtype=torch.float32)
            for e in known_user_embeddings
        ])

        cos = torch.nn.CosineSimilarity(dim=-1)
        similarities = cos(embedding.unsqueeze(0), known_user_embeddings)

        best_score, best_idx = torch.max(similarities, dim=0)
        print(f"Best similarity: {best_score.item():.3f}")

        if best_score.item() > 0.5:  # TODO: config variable, validate threshold empirically
            user_id = list(embeddings_dict.keys())[best_idx.item()]
            try:
                user_memory = memory_retriever(user_id, conn)
            except sqlite3.OperationalError as e:
                print(f"[!] Could not retrieve memory for {user_id}: {e}")
                user_memory = ""
            return user_id, user_memory

    # if no user found, create a new user
    cursor.execute("SELECT COUNT(*) FROM user_embeddings")
    user_number = cursor.fetchone()[0] + 1
    new_user_id = f"user{user_number}"

    embedding_blob = embedding.detach().cpu().numpy().astype(np.float32).tobytes()
    cursor.execute(
        "INSERT INTO user_embeddings (user_id, embeddings) VALUES (?, ?)",
        (new_user_id, embedding_blob)
    )

    # NOTE: one table per user won't scale well as the user base grows.
    # Consider a single shared table (user_id, type, name, description,
    # value, embeddings) with an index on user_id. Not fixing now.
    cursor.execute(
        # safe: new_user_id is internally generated, never from user input
        f"CREATE TABLE IF NOT EXISTS {new_user_id} (type TEXT, name TEXT, description TEXT, value TEXT, embeddings BLOB)"
    )

    primary_features = {
        "nom": "Les prénoms et noms de l'utilisateur.",
        "age": "L'âge de l'utilisateur.",
        "genre": "Le genre de l'utilisateur (masculin, féminin, non-binaire, etc.).",
        "preference_dialogue": "Les préférences de dialogue de l'utilisateur (formel, informel, humoristique, etc.)."
    }
    for feature, description in primary_features.items():
        cursor.execute(
            # safe: new_user_id is internally generated, never from user input
            f"INSERT INTO {new_user_id} (type, name, description, value, embeddings) VALUES (?, ?, ?, ?, ?)",
            ("primary", feature, description, '', None),
        )

    user_memory = memory_retriever(new_user_id, conn)

    conn.commit()
    if new_conn:
        conn.close()
    return new_user_id, user_memory
