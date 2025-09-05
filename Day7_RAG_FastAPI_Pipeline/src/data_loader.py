"""
Data loading module for Wikipedia corpus.
Handles fetching, processing, and saving Wikipedia articles for RAG pipeline.
"""

import os
import json
import pandas as pd
import wikipedia
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """
    Data class for storing document information.
    
    This is our core data structure - each Wikipedia article becomes a Document.
    Using @dataclass automatically creates __init__, __repr__, and comparison methods.
    """
    id: str          # Unique identifier (e.g., "AI_0", "AI_1")
    title: str       # Wikipedia article title
    content: str     # Full article text content
    url: str         # Wikipedia URL for reference
    
    def to_dict(self) -> Dict:
        """Convert Document to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "title": self.title, 
            "content": self.content,
            "url": self.url
        }


class WikipediaDataLoader:
    """
    Handles loading and processing Wikipedia articles for the RAG system.
    
    This class manages:
    - Data directory setup
    - Document storage and retrieval
    - Wikipedia API interactions
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the Wikipedia data loader.
        
        Args:
            data_dir: Directory path where data files will be saved/loaded
        """
        self.data_dir = data_dir
        self.documents: List[Document] = []  # Will store our fetched documents
        
        # Ensure the data directory exists (create if it doesn't)
        os.makedirs(data_dir, exist_ok=True)
        logger.info(f"WikipediaDataLoader initialized with data_dir: {data_dir}")
    
    def get_default_topics(self) -> List[str]:
        """
        Get a default set of diverse topics for demonstration.
        
        These topics are chosen to:
        - Cover different domains (tech, science, environment, space)
        - Have rich, well-written Wikipedia articles
        - Be interesting for semantic search demonstrations
        
        Returns:
            List of Wikipedia topic strings
        """
        topics = [
            'Artificial Intelligence',
            'Machine Learning',
            'Deep Learning',
            'Neural Networks',
            'Generative AI',
            'Computer Vision',
            'Large Language Model',
            'Retrieval-augmented generation',
            'Object Detection',
            'Face Recognition',
            'Natural Language Processing',
            'Image Processing',
            'Data Science',
            'Data Mining',
            'Big Data',
            'Data Analytics',
            'Predictive Analytics',
            'Statistical Modeling',
            'Data Visualization',
            'Exploratory Data Analysis',
            'Data Cleaning',
            'ETL (Extract Transform Load)',
            'Business Intelligence',
            'Data Warehousing',
            'Feature Engineering',
            'Time Series Analysis',
            'Reinforcement Learning',
            'Anomaly Detection',
            'Data Governance',
            'Data Ethics',
            'Cloud Computing for Data Science',
            "Quantum Computing",
            "Blockchain Technology"
        ]
        return topics
    
    def fetch_wikipedia_articles(self, 
                                topics: List[str], 
                                max_articles_per_topic: int = 3,
                                max_content_length: int = 2000) -> List[Document]:
        """
        Fetch Wikipedia articles for given topics.
        
        This method:
        1. Searches Wikipedia for each topic
        2. Retrieves article content
        3. Handles errors gracefully (disambiguation, missing pages)
        4. Creates Document objects with structured data
        
        Args:
            topics: List of Wikipedia topics to search
            max_articles_per_topic: Maximum articles to fetch per topic
            max_content_length: Maximum character length for content
            
        Returns:
            List of Document objects containing fetched articles
        """
        documents = []
        
        for topic in topics:
            logger.info(f"Fetching articles for topic: {topic}")
            
            try:
                # Search for articles related to the topic
                search_results = wikipedia.search(topic, results=max_articles_per_topic)
                
                for i, title in enumerate(search_results[:max_articles_per_topic]):
                    try:
                        # Get the Wikipedia page
                        page = wikipedia.page(title)
                        
                        # Truncate content if too long
                        content = page.content
                        if len(content) > max_content_length:
                            content = content[:max_content_length] + "..."
                        
                        # Create document with unique ID
                        doc = Document(
                            id=f"{topic.replace(' ', '_')}_{i}",
                            title=page.title,
                            content=content,
                            url=page.url
                        )
                        
                        documents.append(doc)
                        logger.info(f"✓ Fetched: {page.title}")
                        
                    except wikipedia.exceptions.DisambiguationError as e:
                        # Handle disambiguation - try the first suggestion
                        try:
                            page = wikipedia.page(e.options[0])
                            content = page.content
                            if len(content) > max_content_length:
                                content = content[:max_content_length] + "..."
                            
                            doc = Document(
                                id=f"{topic.replace(' ', '_')}_{i}",
                                title=page.title,
                                content=content,
                                url=page.url
                            )
                            documents.append(doc)
                            logger.info(f"✓ Fetched (disambiguated): {page.title}")
                            
                        except Exception as inner_e:
                            logger.warning(f"Failed to fetch disambiguated page: {inner_e}")
                            
                    except Exception as e:
                        logger.warning(f"Failed to fetch article '{title}': {e}")
                        
            except Exception as e:
                logger.error(f"Failed to search for topic '{topic}': {e}")
        
        # Store the documents in the class
        self.documents = documents
        logger.info(f"Successfully fetched {len(documents)} articles")
        return documents
    
    def save_corpus(self, filename: str = "wikipedia_corpus.json") -> str:
        """
        Save the fetched corpus to a JSON file.
        
        This method:
        1. Validates that documents exist
        2. Creates metadata about the corpus
        3. Saves in a structured JSON format for easy loading
        
        Args:
            filename: Output filename for the corpus
            
        Returns:
            Path to the saved file
            
        Raises:
            ValueError: If no documents have been fetched yet
        """
        if not self.documents:
            raise ValueError("No documents to save. Call fetch_wikipedia_articles() first.")
        
        filepath = os.path.join(self.data_dir, filename)
        
        # Create structured corpus data with metadata
        corpus_data = {
            "metadata": {
                "total_documents": len(self.documents),
                "topics_covered": list(set(doc.id.split('_')[0] for doc in self.documents)),
                "generation_info": "Created by WikipediaDataLoader"
            },
            "documents": [doc.to_dict() for doc in self.documents]
        }
        
        # Save to JSON with proper encoding
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(corpus_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Corpus saved to: {filepath}")
        logger.info(f"Saved {len(self.documents)} documents covering {len(corpus_data['metadata']['topics_covered'])} unique topics")
        return filepath
    
    def load_corpus(self, filename: str = "wikipedia_corpus.json") -> List[Document]:
        """
        Load corpus from a previously saved JSON file.
        
        This method:
        1. Validates that the file exists
        2. Loads and parses the JSON data
        3. Reconstructs Document objects from the data
        4. Updates the class's document storage
        
        Args:
            filename: Input filename to load from
            
        Returns:
            List of Document objects loaded from file
            
        Raises:
            FileNotFoundError: If the corpus file doesn't exist
            json.JSONDecodeError: If the file is not valid JSON
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Corpus file not found: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract documents from the saved structure
            documents = []
            for doc_dict in data["documents"]:
                doc = Document(
                    id=doc_dict["id"],
                    title=doc_dict["title"],
                    content=doc_dict["content"],
                    url=doc_dict["url"]
                )
                documents.append(doc)
            
            # Store in class and log success
            self.documents = documents
            
            # Log metadata if available
            if "metadata" in data:
                metadata = data["metadata"]
                logger.info(f"Loaded {metadata.get('total_documents', len(documents))} documents")
                logger.info(f"Topics covered: {len(metadata.get('topics_covered', []))}")
            else:
                logger.info(f"Loaded {len(documents)} documents from: {filepath}")
            
            return documents
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in {filepath}: {e}")
            raise
        except KeyError as e:
            logger.error(f"Missing required field in corpus file: {e}")
            raise ValueError(f"Corpus file format error - missing field: {e}")


def create_sample_corpus(data_dir: str = "data") -> List[Document]:
    """
    Convenience function to create a sample Wikipedia corpus.
    
    This function:
    1. Checks if a corpus already exists (avoids re-fetching)
    2. If not, creates a new corpus with default topics
    3. Saves the corpus for future use
    4. Returns the documents for immediate use
    
    Perfect for getting started quickly or for demonstrations.
    
    Args:
        data_dir: Directory to save/load data (defaults to "data")
        
    Returns:
        List of Document objects (either loaded or newly fetched)
        
    Example:
        >>> documents = create_sample_corpus()
        >>> print(f"Ready to use {len(documents)} documents!")
    """
    loader = WikipediaDataLoader(data_dir)
    
    # Check if corpus already exists
    corpus_path = os.path.join(data_dir, "wikipedia_corpus.json")
    
    if os.path.exists(corpus_path):
        logger.info("Found existing corpus - loading...")
        return loader.load_corpus()
    else:
        logger.info("No existing corpus found - creating new one...")
        
        # Get default topics and fetch articles
        topics = loader.get_default_topics()
        logger.info(f"Fetching articles for {len(topics)} topics...")
        
        # Fetch with reasonable defaults for a sample corpus
        documents = loader.fetch_wikipedia_articles(
            topics, 
            max_articles_per_topic=2,  # Keep it manageable
            max_content_length=1500    # Good balance for embeddings
        )
        
        # Save for future use
        loader.save_corpus()
        
        return documents
