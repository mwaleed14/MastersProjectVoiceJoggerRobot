from typing import Optional
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
from misc import Command
import numpy as np


class TextClassifier:

    def __init__(self) -> None:

        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        commands_sentences = [member.name.lower().replace('_', ' ') for member in Command]

        self.command_embeddings = self.model.encode(commands_sentences, convert_to_tensor=False)

    def find_match(self, input_sentence: str, threshold: float) -> Optional[Command]:
        print(" sentence is in match " , input_sentence)
        inp_embedding = self.model.encode(input_sentence, convert_to_tensor=False).reshape(1, -1)

        sim = (1 - distance.cdist(inp_embedding, self.command_embeddings, 'cosine')).flatten()

        maxidx = np.argmax(sim)
        if sim[maxidx] < threshold:
            return None
        return Command(maxidx)
