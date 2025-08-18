#!/usr/bin/env python3
"""
Script to ingest the movie dataset into Neo4j.
This script reads the Cypher queries from movies_dataset.txt and executes them against a Neo4j database.
"""

import os
import sys
from neo4j import GraphDatabase
from typing import List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Neo4jIngester:
    def __init__(self, uri: str, user: str, password: str):
        """Initialize Neo4j connection."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()

    def test_connection(self) -> bool:
        """Test the connection to Neo4j."""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                value = result.single()["test"]
                logger.info(f"Connection test successful: {value}")
                return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def clear_database(self) -> bool:
        """Clear all nodes and relationships from the database."""
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (n) DETACH DELETE n")
                logger.info("Database cleared successfully")
                return True
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            return False

    def execute_cypher_queries(self, queries: List[str]) -> bool:
        """Execute a list of Cypher queries."""
        try:
            with self.driver.session() as session:
                for i, query in enumerate(queries, 1):
                    if query.strip():  # Skip empty lines
                        logger.info(f"Executing query {i}/{len(queries)}")
                        logger.debug(f"Query: {query[:100]}...")

                        result = session.run(query)
                        # Consume the result to ensure the query completes
                        list(result)

                        logger.info(f"Query {i} executed successfully")

            logger.info("All queries executed successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to execute queries: {e}")
            return False

    def verify_ingestion(self) -> dict:
        """Verify that the data was ingested correctly."""
        try:
            with self.driver.session() as session:
                # Count nodes by type
                node_counts = session.run(
                    """
                    MATCH (n)
                    RETURN labels(n)[0] as node_type, count(n) as count
                    ORDER BY count DESC
                """
                )

                # Count relationships by type
                rel_counts = session.run(
                    """
                    MATCH ()-[r]->()
                    RETURN type(r) as rel_type, count(r) as count
                    ORDER BY count DESC
                """
                )

                # Get some sample data
                sample_movies = session.run(
                    """
                    MATCH (m:Movie)
                    RETURN m.title as title, m.released as year
                    LIMIT 5
                """
                )

                sample_actors = session.run(
                    """
                    MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
                    RETURN p.name as actor, m.title as movie
                    LIMIT 5
                """
                )

                return {
                    "node_counts": [dict(record) for record in node_counts],
                    "rel_counts": [dict(record) for record in rel_counts],
                    "sample_movies": [dict(record) for record in sample_movies],
                    "sample_actors": [dict(record) for record in sample_actors],
                }

        except Exception as e:
            logger.error(f"Failed to verify ingestion: {e}")
            return {}


def read_cypher_file(file_path: str) -> List[str]:
    """Read Cypher queries from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Split by semicolon and filter out empty lines
        queries = [query.strip() for query in content.split(";") if query.strip()]

        logger.info(f"Read {len(queries)} Cypher queries from {file_path}")
        return queries

    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        return []


def main():
    """Main function to run the ingestion process."""
    # Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "aisac_kg")
    DATASET_FILE = "agentic_graph_rag/movies_dataset.txt"

    logger.info("Starting Neo4j movie dataset ingestion...")
    logger.info(f"Neo4j URI: {NEO4J_URI}")
    logger.info(f"Dataset file: {DATASET_FILE}")

    # Check if dataset file exists
    if not os.path.exists(DATASET_FILE):
        logger.error(f"Dataset file not found: {DATASET_FILE}")
        sys.exit(1)

    # Initialize Neo4j connection
    ingester = Neo4jIngester(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    try:
        # Test connection
        if not ingester.test_connection():
            logger.error(
                "Cannot connect to Neo4j. Please check your connection settings."
            )
            sys.exit(1)

        # Clear existing data (optional - comment out if you want to keep existing data)
        logger.info("Clearing existing database...")
        if not ingester.clear_database():
            logger.warning("Failed to clear database. Continuing anyway...")

        # Read Cypher queries
        queries = read_cypher_file(DATASET_FILE)
        if not queries:
            logger.error("No queries found in dataset file.")
            sys.exit(1)

        # Execute queries
        logger.info("Executing Cypher queries...")
        if not ingester.execute_cypher_queries(queries):
            logger.error("Failed to execute queries.")
            sys.exit(1)

        # Verify ingestion
        logger.info("Verifying ingestion...")
        verification = ingester.verify_ingestion()

        if verification:
            logger.info("=== INGESTION VERIFICATION ===")
            logger.info("Node counts:")
            for node in verification["node_counts"]:
                logger.info(f"  {node['node_type']}: {node['count']}")

            logger.info("Relationship counts:")
            for rel in verification["rel_counts"]:
                logger.info(f"  {rel['rel_type']}: {rel['count']}")

            logger.info("Sample movies:")
            for movie in verification["sample_movies"]:
                logger.info(f"  {movie['title']} ({movie['year']})")

            logger.info("Sample actor-movie relationships:")
            for actor in verification["sample_actors"]:
                logger.info(f"  {actor['actor']} -> {actor['movie']}")

        logger.info("Movie dataset ingestion completed successfully!")

    except KeyboardInterrupt:
        logger.info("Ingestion interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during ingestion: {e}")
        sys.exit(1)
    finally:
        ingester.close()


if __name__ == "__main__":
    main()
