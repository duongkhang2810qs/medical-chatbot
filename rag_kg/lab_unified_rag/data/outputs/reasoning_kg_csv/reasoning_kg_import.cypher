// ==============================================
// IMPORT REASONING KG
// Files required in Neo4j import folder:
// - reasoning_nodes.csv
// - reasoning_edges.csv
//
// This graph is book/rule reasoning knowledge:
// FindingConcept -> Pattern -> Condition -> Evidence -> Source
// It is intentionally separate from the old case-centric import.
// ==============================================

CREATE CONSTRAINT kg_node_id IF NOT EXISTS
FOR (n:KGNode) REQUIRE n.id IS UNIQUE;

// ------------------------------
// Nodes by type
// ------------------------------

LOAD CSV WITH HEADERS FROM 'file:///reasoning_nodes.csv' AS row
WITH row WHERE row.node_type = 'Panel'
MERGE (n:KGNode:Panel {id: row.id})
SET n += row;

LOAD CSV WITH HEADERS FROM 'file:///reasoning_nodes.csv' AS row
WITH row WHERE row.node_type = 'Status'
MERGE (n:KGNode:Status {id: row.id})
SET n += row;

LOAD CSV WITH HEADERS FROM 'file:///reasoning_nodes.csv' AS row
WITH row WHERE row.node_type = 'Test'
MERGE (n:KGNode:Test {id: row.id})
SET n += row;

LOAD CSV WITH HEADERS FROM 'file:///reasoning_nodes.csv' AS row
WITH row WHERE row.node_type = 'FindingConcept'
MERGE (n:KGNode:FindingConcept {id: row.id})
SET n += row;

LOAD CSV WITH HEADERS FROM 'file:///reasoning_nodes.csv' AS row
WITH row WHERE row.node_type = 'Pattern'
MERGE (n:KGNode:Pattern {id: row.id})
SET n += row;

LOAD CSV WITH HEADERS FROM 'file:///reasoning_nodes.csv' AS row
WITH row WHERE row.node_type = 'Condition'
MERGE (n:KGNode:Condition {id: row.id})
SET n += row;

LOAD CSV WITH HEADERS FROM 'file:///reasoning_nodes.csv' AS row
WITH row WHERE row.node_type = 'Evidence'
MERGE (n:KGNode:Evidence {id: row.id})
SET n += row,
    n.page = CASE WHEN row.page = '' THEN null ELSE toInteger(row.page) END,
    n.score = CASE WHEN row.score = '' THEN null ELSE toFloat(row.score) END,
    n.trust = CASE WHEN row.trust = '' THEN null ELSE toFloat(row.trust) END;

LOAD CSV WITH HEADERS FROM 'file:///reasoning_nodes.csv' AS row
WITH row WHERE row.node_type = 'Source'
MERGE (n:KGNode:Source {id: row.id})
SET n += row;

// ------------------------------
// Edges
// ------------------------------

LOAD CSV WITH HEADERS FROM 'file:///reasoning_edges.csv' AS row
MATCH (s:KGNode {id: row.source_id})
MATCH (t:KGNode {id: row.target_id})
MERGE (s)-[r:KG_REL {id: row.id}]->(t)
SET r.type = row.relation,
    r.confidence = CASE WHEN row.confidence = '' THEN null ELSE toFloat(row.confidence) END,
    r.provenance = row.provenance;

// ------------------------------
// Demo queries
// ------------------------------

// 1. Node counts
MATCH (n:KGNode)
RETURN n.node_type AS node_type, count(n) AS cnt
ORDER BY cnt DESC;

// 2. Relationship counts
MATCH (:KGNode)-[r:KG_REL]->(:KGNode)
RETURN r.type AS relation_type, count(r) AS cnt
ORDER BY cnt DESC;

// 3. Example: WBC high -> Pattern -> Condition
MATCH path = (f:FindingConcept {id:'finding_concept:cbc:wbc:high'})
  -[r1:KG_REL]->(p:Pattern)
  -[r2:KG_REL]->(c:Condition)
WHERE r1.type IN ['REQUIRED_FOR_PATTERN', 'OPTIONAL_FOR_PATTERN']
  AND r2.type = 'SUGGESTS_CONDITION'
RETURN path;

// 4. Example: WBC high -> Evidence -> Source
MATCH path = (f:FindingConcept {id:'finding_concept:cbc:wbc:high'})
  -[r1:KG_REL]->(e:Evidence)
  -[r2:KG_REL]->(s:Source)
WHERE r1.type = 'SUPPORTED_BY_EVIDENCE'
  AND r2.type = 'FROM_SOURCE'
RETURN path
LIMIT 20;

// 5. Example: co-occurring findings for reasoning
MATCH path = (f:FindingConcept)-[r:KG_REL]->(g:FindingConcept)
WHERE r.type = 'CO_OCCURS_WITH'
RETURN path
LIMIT 30;
