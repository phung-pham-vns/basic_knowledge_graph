import_nodes_query = """
MERGE (b:Book {id: $book_id})
MERGE (b)-[:HAS_CHUNK]->(c:__Chunk__ {id: $chunk_id})
SET c.text = $text
WITH c
UNWIND $data AS row
MERGE (n:__Entity__ {name: row.entity_name})
SET n:$(row.entity_type),
    n.description = coalesce(n.description, []) + [row.entity_description]
MERGE (n)<-[:MENTIONS]-(c)
"""

import_relationships_query = """
UNWIND $data AS row
MERGE (s:__Entity__ {name: row.source_entity})
MERGE (t:__Entity__ {name: row.target_entity})
CREATE (s)-[r:RELATIONSHIP {description: row.relationship_description, strength: row.relationship_strength}]->(t)
"""

import_community_query = """
UNWIND $data AS row
MERGE (c:__Community__ {communityId: row.communityId})
SET c.title = row.community.title,
    c.summary = row.community.summary,
    c.rating = row.community.rating,
    c.rating_explanation = row.community.rating_explanation
WITH c, row
UNWIND row.nodes AS node
MERGE (n:__Entity__ {name: node})
MERGE (n)-[:IN_COMMUNITY]->(c)
"""

community_info_query = """MATCH (e:__Entity__)
WHERE e.louvain IS NOT NULL
WITH e.louvain AS louvain, collect(e) AS nodes
WHERE size(nodes) > 1
CALL apoc.path.subgraphAll(nodes[0], {
	whitelistNodes:nodes
})
YIELD relationships
RETURN louvain AS communityId,
       [n in nodes | {id: n.name, description: n.summary, type: [el in labels(n) WHERE el <> '__Entity__'][0]}] AS nodes,
       [r in relationships | {start: startNode(r).name, type: type(r), end: endNode(r).name, description: r.description}] AS rels"""


def import_entity_summary(neo4j_driver, entity_information):
    neo4j_driver.execute_query(
        """
    UNWIND $data AS row
    MATCH (e:__Entity__ {name: row.entity})
    SET e.summary = row.summary
    """,
        data=entity_information,
    )

    # If there was only 1 description use that
    neo4j_driver.execute_query(
        """
    MATCH (e:__Entity__)
    WHERE size(e.description) = 1
    SET e.summary = e.description[0]
    """
    )


def import_rels_summary(neo4j_driver, rel_summaries):
    neo4j_driver.execute_query(
        """
    UNWIND $data AS row
    MATCH (s:__Entity__ {name: row.source}), (t:__Entity__ {name: row.target})
    MERGE (s)-[r:SUMMARIZED_RELATIONSHIP]-(t)
    SET r.summary = row.summary
    """,
        data=rel_summaries,
    )

    # If there was only 1 description use that
    neo4j_driver.execute_query(
        """
    MATCH (s:__Entity__)-[e:RELATIONSHIP]-(t:__Entity__)
    WHERE NOT (s)-[:SUMMARIZED_RELATIONSHIP]-(t)
    MERGE (s)-[r:SUMMARIZED_RELATIONSHIP]-(t)
    SET r.summary = e.description
    """
    )


def calculate_communities(neo4j_driver):
    # Drop graph if exist
    try:
        neo4j_driver.execute_query(
            """
       CALL gds.graph.drop('entity')
      """
        )
    except:
        pass
    neo4j_driver.execute_query(
        """
    MATCH (source:__Entity__)-[r:RELATIONSHIP]->(target:__Entity__)
    WITH gds.graph.project('entity', source, target, {}, {undirectedRelationshipTypes: ['*']}) AS g
    RETURN
      g.graphName AS graph, g.nodeCount AS nodes, g.relationshipCount AS rels
    """
    )

    records, _, _ = neo4j_driver.execute_query(
        """
    CALL gds.louvain.write("entity", {writeProperty:"louvain"})
    """
    )
    return [el.data() for el in records][0]
