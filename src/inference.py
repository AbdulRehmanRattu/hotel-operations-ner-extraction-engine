import os
import spacy
from sentence_transformers import SentenceTransformer, util

class HotelNERAndMapper:
    """
    Two-stage NLP Engine:
    Stage 1: Custom spaCy Transition-Based NER for boundary token extraction (LOCATION, ITEM).
    Stage 2: SentenceTransformer Dense Embedding Cosine Similarity to map raw entities to Standard ERP Catalog.
    """
    def __init__(self, model_path=None, embedding_model='paraphrase-mpnet-base-v2'):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'custom_ner_model_md')
        
        print(f"Loading custom spaCy NER model from: {model_path}")
        self.nlp = spacy.load(model_path)
        
        print(f"Loading SentenceTransformer embedding model: {embedding_model}")
        self.mapper = SentenceTransformer(embedding_model)
        
        # Standardized Hotel Inventory & Facility Taxonomies
        self.standard_items = [
            "Extra Bath Towels", "Hypoallergenic Pillow", "Bed Linen Set",
            "Box of Artisanal Chocolates", "Mineral Water Bottle", "Plumbing Leak Repair",
            "Luggage Valet Cart", "Room Service Breakfast Tray", "Dental Hygiene Kit",
            "Iron and Ironing Board", "Air Conditioning Maintenance", "Baby Crib"
        ]
        
        self.standard_locations = [
            "Guest Room 101", "Guest Room 102", "Guest Room 201", "Guest Room 202",
            "Penthouse Suite A", "Presidential Suite B", "Lobby Lounge",
            "Poolside Bar & Grill", "Fitness Center", "Executive Conference Hall",
            "Basement Parking Bay", "Rooftop Terrace"
        ]
        
        # Pre-compute catalog embeddings
        self.item_embeddings = self.mapper.encode(self.standard_items, convert_to_tensor=True)
        self.location_embeddings = self.mapper.encode(self.standard_locations, convert_to_tensor=True)

    def extract_entities(self, text: str):
        doc = self.nlp(text)
        entities = {"LOCATION": [], "ITEM": []}
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
        return entities

    def map_entity(self, raw_text: str, category: str):
        query_emb = self.mapper.encode(raw_text, convert_to_tensor=True)
        if category == "ITEM":
            scores = util.cos_sim(query_emb, self.item_embeddings)[0]
            best_idx = int(scores.argmax())
            return {
                "raw_entity": raw_text,
                "mapped_standard": self.standard_items[best_idx],
                "confidence": float(scores[best_idx])
            }
        else:
            scores = util.cos_sim(query_emb, self.location_embeddings)[0]
            best_idx = int(scores.argmax())
            return {
                "raw_entity": raw_text,
                "mapped_standard": self.standard_locations[best_idx],
                "confidence": float(scores[best_idx])
            }

    def process_request(self, text: str):
        extracted = self.extract_entities(text)
        mapped_results = {"raw_input": text, "extracted_entities": extracted, "standardized_dispatch": []}
        
        for item in extracted["ITEM"]:
            mapped_results["standardized_dispatch"].append(self.map_entity(item, "ITEM"))
            
        for loc in extracted["LOCATION"]:
            mapped_results["standardized_dispatch"].append(self.map_entity(loc, "LOCATION"))
            
        return mapped_results

if __name__ == "__main__":
    engine = HotelNERAndMapper()
    sample_text = "Please send two extra pillows and a box of chocolates to room 101 immediately."
    result = engine.process_request(sample_text)
    print("\nSample Dispatch Result:")
    import json
    print(json.dumps(result, indent=2))
