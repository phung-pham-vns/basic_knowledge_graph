from neo4j import GraphDatabase

uri = "bolt://127.0.0.1:7687"  # force IPv4
auth = ("neo4j", "aisac_kg")

with GraphDatabase.driver(uri, auth=auth) as driver:

    def ping(tx):
        return tx.run("RETURN 1 AS ok").single()["ok"]

    with driver.session() as session:
        print(session.execute_read(ping))
