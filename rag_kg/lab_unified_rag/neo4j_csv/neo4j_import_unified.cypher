// ==============================================
// CLEAN OLD GRAPH OPTIONAL
// Chỉ chạy dòng dưới nếu muốn xóa graph cũ:
// MATCH (n) DETACH DELETE n;
// ==============================================

// ==============================================
// CONSTRAINTS
// ==============================================
CREATE CONSTRAINT panel_id IF NOT EXISTS FOR (n:Panel) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT test_id IF NOT EXISTS FOR (n:Test) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT status_id IF NOT EXISTS FOR (n:Status) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT condition_id IF NOT EXISTS FOR (n:Condition) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT source_id IF NOT EXISTS FOR (n:Source) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (n:Topic) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT case_id IF NOT EXISTS FOR (n:Case) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT finding_id IF NOT EXISTS FOR (n:Finding) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT pattern_id IF NOT EXISTS FOR (n:Pattern) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT evidence_id IF NOT EXISTS FOR (n:Evidence) REQUIRE n.id IS UNIQUE;

// ==============================================
// IMPORT NODES
// ==============================================
LOAD CSV WITH HEADERS FROM 'file:///nodes_panel.csv' AS row
MERGE (n:Panel {id: row.id})
SET n.name = row.name,
    n.display_name = row.display_name;

LOAD CSV WITH HEADERS FROM 'file:///nodes_test.csv' AS row
MERGE (n:Test {id: row.id})
SET n.test_code = row.test_code,
    n.name = row.name,
    n.panel = row.panel;

LOAD CSV WITH HEADERS FROM 'file:///nodes_status.csv' AS row
MERGE (n:Status {id: row.id})
SET n.name = row.name,
    n.display_name = row.display_name;

LOAD CSV WITH HEADERS FROM 'file:///nodes_condition.csv' AS row
MERGE (n:Condition {id: row.id})
SET n.name = row.name,
    n.canonical_name = row.canonical_name;

LOAD CSV WITH HEADERS FROM 'file:///nodes_source.csv' AS row
MERGE (n:Source {id: row.id})
SET n.name = row.name,
    n.kind = row.kind;

LOAD CSV WITH HEADERS FROM 'file:///nodes_topic.csv' AS row
MERGE (n:Topic {id: row.id})
SET n.name = row.name;

LOAD CSV WITH HEADERS FROM 'file:///nodes_case.csv' AS row
MERGE (n:Case {id: row.id})
SET n.case_id = row.case_id,
    n.summary = row.summary,
    n.combined_interpretation = row.combined_interpretation,
    n.warning = row.warning,
    n.confidence = row.confidence,
    n.panel = row.panel;

LOAD CSV WITH HEADERS FROM 'file:///nodes_finding.csv' AS row
MERGE (n:Finding {id: row.id})
SET n.test_code = row.test_code,
    n.direction = row.direction,
    n.value = row.value,
    n.ref = row.ref,
    n.raw_status = row.raw_status,
    n.name = row.name,
    n.panel = row.panel,
    n.unit = row.unit;

LOAD CSV WITH HEADERS FROM 'file:///nodes_pattern.csv' AS row
MERGE (n:Pattern {id: row.id})
SET n.name = row.name,
    n.interpretation = row.interpretation,
    n.confidence_label = row.confidence_label,
    n.match_score = toFloat(coalesce(row.match_score,'0')),
    n.source = row.source,
    n.panel = row.panel;

LOAD CSV WITH HEADERS FROM 'file:///nodes_evidence.csv' AS row
MERGE (n:Evidence {id: row.id})
SET n.text = row.text,
    n.record_type = row.record_type,
    n.type = row.type,
    n.page = row.page,
    n.trust = toFloat(coalesce(row.trust, '0')),
    n.score = toFloat(coalesce(row.score, '0')),
    n.panel = row.panel,
    n.source = row.source;

// ==============================================
// IMPORT EDGES
// Dùng 1 relationship type chung RELATED_TO
// property r.type giữ relation thật.
// ==============================================
LOAD CSV WITH HEADERS FROM 'file:///edges_all.csv' AS row
MATCH (s {id: row.source_id})
MATCH (t {id: row.target_id})
MERGE (s)-[r:RELATED_TO {id: row.id}]->(t)
SET r.type = row.relation,
    r.confidence = toFloat(coalesce(row.confidence,'0')),
    r.provenance = row.provenance,
    r.source_type = row.source_type,
    r.target_type = row.target_type;

// ==============================================
// DEMO QUERIES
// ==============================================

// 1. Thống kê node
MATCH (n)
RETURN labels(n) AS label, count(n) AS cnt
ORDER BY cnt DESC;

// 2. Thống kê relation
MATCH ()-[r:RELATED_TO]->()
RETURN r.type AS relation_type, count(*) AS cnt
ORDER BY cnt DESC;

// 3. Case -> Finding -> Test -> Status
MATCH (c:Case)-[r1:RELATED_TO]->(f:Finding),
      (f)-[r2:RELATED_TO]->(t:Test),
      (f)-[r3:RELATED_TO]->(s:Status)
WHERE r1.type = 'HAS_FINDING'
  AND r2.type = 'OF_TEST'
  AND r3.type = 'HAS_STATUS'
RETURN c, r1, f, r2, t, r3, s
LIMIT 30;

// 4. Case -> Finding -> Pattern -> Condition
MATCH (c:Case)-[r1:RELATED_TO]->(f:Finding),
      (f)-[r2:RELATED_TO]->(p:Pattern),
      (p)-[r3:RELATED_TO]->(cond:Condition)
WHERE r1.type = 'HAS_FINDING'
  AND r2.type = 'PART_OF_PATTERN'
  AND r3.type = 'SUGGESTS'
RETURN c, r1, f, r2, p, r3, cond
LIMIT 30;

// 5. Reasoning đầy đủ + evidence text
MATCH (c:Case)-[r1:RELATED_TO]->(f:Finding),
      (f)-[r2:RELATED_TO]->(p:Pattern),
      (p)-[r3:RELATED_TO]->(cond:Condition),
      (e:Evidence)-[r4:RELATED_TO]->(cond)
WHERE r1.type = 'HAS_FINDING'
  AND r2.type = 'PART_OF_PATTERN'
  AND r3.type = 'SUGGESTS'
  AND r4.type = 'SUPPORTS'
RETURN c.case_id AS case_id,
       f.name AS finding,
       p.name AS pattern,
       cond.name AS condition,
       left(e.text, 220) AS evidence_text,
       e.source AS source,
       e.page AS page,
       e.score AS score,
       e.trust AS trust
LIMIT 30;

// 6. Cross-panel cases
MATCH (c:Case)
WHERE c.panel CONTAINS 'CBC' AND c.panel CONTAINS 'BIOCHEM'
RETURN c
LIMIT 20;

// 7. Evidence theo panel
MATCH (e:Evidence)
RETURN e.panel AS panel, count(e) AS evidence_count
ORDER BY evidence_count DESC;