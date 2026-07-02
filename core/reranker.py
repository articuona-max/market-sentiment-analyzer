import numpy as np
from datetime import datetime, timezone

class TimeDecayReranker:
    def __init__(self, decay_rate: float = 0.35):
        self.decay_rate = decay_rate

    def rerank(self, documents: list) -> list:
        """
        Applies exponential decay: S_final = S_semantic * e^(-lambda * t)
        t = age in hours
        """
        current_time = datetime.now(timezone.utc)
        processed_docs = []
        
        for doc in documents:
            doc_time = datetime.fromisoformat(doc['timestamp'])
            # Delta T in hours
            age_hours = max(0.0, (current_time - doc_time).total_seconds() / 3600.0)
            
            decay_penalty = np.exp(-self.decay_rate * age_hours)
            doc['final_score'] = float(round(doc['semantic_score'] * decay_penalty, 4))
            processed_docs.append(doc)
            
        # Sort descending by the new score matrix
        processed_docs.sort(key=lambda x: x['final_score'], reverse=True)
        return processed_docs