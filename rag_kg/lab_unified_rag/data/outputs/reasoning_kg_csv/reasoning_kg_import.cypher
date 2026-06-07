// ==============================================
// STEP 0: XÓA GRAPH CŨ (bỏ comment dòng dưới nếu muốn reset)
// MATCH (n) DETACH DELETE n;
// ==============================================

// ==============================================
// CONSTRAINTS
// ==============================================
CREATE CONSTRAINT panel_id     IF NOT EXISTS FOR (n:Panel)     REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT test_id      IF NOT EXISTS FOR (n:Test)      REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT status_id    IF NOT EXISTS FOR (n:Status)    REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT condition_id IF NOT EXISTS FOR (n:Condition) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT source_id    IF NOT EXISTS FOR (n:Source)    REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT topic_id     IF NOT EXISTS FOR (n:Topic)     REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT case_id      IF NOT EXISTS FOR (n:Case)      REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT finding_id   IF NOT EXISTS FOR (n:Finding)   REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT pattern_id   IF NOT EXISTS FOR (n:Pattern)   REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT evidence_id  IF NOT EXISTS FOR (n:Evidence)  REQUIRE n.id IS UNIQUE;

// ==============================================
// IMPORT NODES
// ==============================================
LOAD CSV WITH HEADERS FROM 'file:///nodes_panel.csv' AS row
MERGE (n:Panel {id: row.id})
SET n.name = row.name, n.display_name = row.display_name;

LOAD CSV WITH HEADERS FROM 'file:///nodes_test.csv' AS row
MERGE (n:Test {id: row.id})
SET n.test_code = row.test_code, n.name = row.name, n.panel = row.panel;

LOAD CSV WITH HEADERS FROM 'file:///nodes_status.csv' AS row
MERGE (n:Status {id: row.id})
SET n.name = row.name, n.display_name = row.display_name;

LOAD CSV WITH HEADERS FROM 'file:///nodes_condition.csv' AS row
MERGE (n:Condition {id: row.id})
SET n.name = row.name, n.canonical_name = row.canonical_name;

LOAD CSV WITH HEADERS FROM 'file:///nodes_source.csv' AS row
MERGE (n:Source {id: row.id})
SET n.name = row.name, n.kind = row.kind;

LOAD CSV WITH HEADERS FROM 'file:///nodes_topic.csv' AS row
MERGE (n:Topic {id: row.id})
SET n.name = row.name;

LOAD CSV WITH HEADERS FROM 'file:///nodes_case.csv' AS row
MERGE (n:Case {id: row.id})
SET n.case_id            = row.case_id,
    n.summary            = row.summary,
    n.combined_interpretation = row.combined_interpretation,
    n.warning            = row.warning,
    n.confidence         = row.confidence,
    n.panel              = row.panel;

LOAD CSV WITH HEADERS FROM 'file:///nodes_finding.csv' AS row
MERGE (n:Finding {id: row.id})
SET n.test_code  = row.test_code,
    n.direction  = row.direction,
    n.value      = row.value,
    n.ref        = row.ref,
    n.raw_status = row.raw_status,
    n.name       = row.name,
    n.panel      = row.panel,
    n.unit       = row.unit;

LOAD CSV WITH HEADERS FROM 'file:///nodes_pattern.csv' AS row
MERGE (n:Pattern {id: row.id})
SET n.name             = row.name,
    n.interpretation   = row.interpretation,
    n.confidence_label = row.confidence_label,
    n.match_score      = toFloat(coalesce(row.match_score, '0')),
    n.source           = row.source,
    n.panel            = row.panel;

LOAD CSV WITH HEADERS FROM 'file:///nodes_evidence.csv' AS row
MERGE (n:Evidence {id: row.id})
SET n.text        = row.text,
    n.record_type = row.record_type,
    n.type        = row.type,
    n.page        = row.page,
    n.trust       = toFloat(coalesce(row.trust,  '0')),
    n.score       = toFloat(coalesce(row.score,  '0')),
    n.panel       = row.panel,
    n.source      = row.source;

// ==============================================
// IMPORT EDGES — named relationship types
// Mỗi loại quan hệ được import riêng để Neo4j
// hiển thị đúng label trên Browser.
// ==============================================

LOAD CSV WITH HEADERS FROM 'file:///edges_all.csv' AS row
WITH row WHERE row.relation = 'FROM_SOURCE'
MATCH (s {id: row.source_id})
MATCH (t {id: row.target_id})
MERGE (s)-[r:FROM_SOURCE]->(t)
SET r.confidence = toFloat(coalesce(row.confidence, '0')),
    r.provenance = row.provenance;

LOAD CSV WITH HEADERS FROM 'file:///edges_all.csv' AS row
WITH row WHERE row.relation = 'BELONGS_TO_PANEL'
MATCH (s {id: row.source_id})
MATCH (t {id: row.target_id})
MERGE (s)-[r:BELONGS_TO_PANEL]->(t)
SET r.confidence = toFloat(coalesce(row.confidence, '0')),
    r.provenance = row.provenance;

LOAD CSV WITH HEADERS FROM 'file:///edges_all.csv' AS row
WITH row WHERE row.relation = 'MENTIONS_TEST'
MATCH (s {id: row.source_id})
MATCH (t {id: row.target_id})
MERGE (s)-[r:MENTIONS_TEST]->(t)
SET r.confidence = toFloat(coalesce(row.confidence, '0')),
    r.provenance = row.provenance;

LOAD CSV WITH HEADERS FROM 'file:///edges_all.csv' AS row
WITH row WHERE row.relation = 'MENTIONS_TOPIC'
MATCH (s {id: row.source_id})
MATCH (t {id: row.target_id})
MERGE (s)-[r:MENTIONS_TOPIC]->(t)
SET r.confidence = toFloat(coalesce(row.confidence, '0')),
    r.provenance = row.provenance;

LOAD CSV WITH HEADERS FROM 'file:///edges_all.csv' AS row
WITH row WHERE row.relation = 'SUPPORTS'
MATCH (s {id: row.source_id})
MATCH (t {id: row.target_id})
MERGE (s)-[r:SUPPORTS]->(t)
SET r.confidence = toFloat(coalesce(row.confidence, '0')),
    r.provenance = row.provenance;

LOAD CSV WITH HEADERS FROM 'file:///edges_all.csv' AS row
WITH row WHERE row.relation = 'HAS_FINDING'
MATCH (s {id: row.source_id})
MATCH (t {id: row.target_id})
MERGE (s)-[r:HAS_FINDING]->(t)
SET r.confidence = toFloat(coalesce(row.confidence, '0')),
    r.provenance = row.provenance;

LOAD CSV WITH HEADERS FROM 'file:///edges_all.csv' AS row
WITH row WHERE row.relation = 'OF_TEST'
MATCH (s {id: row.source_id})
MATCH (t {id: row.target_id})
MERGE (s)-[r:OF_TEST]->(t)
SET r.confidence = toFloat(coalesce(row.confidence, '0')),
    r.provenance = row.provenance;

LOAD CSV WITH HEADERS FROM 'file:///edges_all.csv' AS row
WITH row WHERE row.relation = 'HAS_STATUS'
MATCH (s {id: row.source_id})
MATCH (t {id: row.target_id})
MERGE (s)-[r:HAS_STATUS]->(t)
SET r.confidence = toFloat(coalesce(row.confidence, '0')),
    r.provenance = row.provenance;

LOAD CSV WITH HEADERS FROM 'file:///edges_all.csv' AS row
WITH row WHERE row.relation = 'PART_OF_PATTERN'
MATCH (s {id: row.source_id})
MATCH (t {id: row.target_id})
MERGE (s)-[r:PART_OF_PATTERN]->(t)
SET r.confidence = toFloat(coalesce(row.confidence, '0')),
    r.provenance = row.provenance;

LOAD CSV WITH HEADERS FROM 'file:///edges_all.csv' AS row
WITH row WHERE row.relation = 'MATCHES_PATTERN'
MATCH (s {id: row.source_id})
MATCH (t {id: row.target_id})
MERGE (s)-[r:MATCHES_PATTERN]->(t)
SET r.confidence = toFloat(coalesce(row.confidence, '0')),
    r.provenance = row.provenance;

LOAD CSV WITH HEADERS FROM 'file:///edges_all.csv' AS row
WITH row WHERE row.relation = 'SUGGESTS'
MATCH (s {id: row.source_id})
MATCH (t {id: row.target_id})
MERGE (s)-[r:SUGGESTS]->(t)
SET r.confidence = toFloat(coalesce(row.confidence, '0')),
    r.provenance = row.provenance;

// ==============================================
// VERIFY — chạy sau khi import xong để kiểm tra
// ==============================================

// 1. Thống kê node
MATCH (n)
RETURN labels(n) AS label, count(n) AS cnt
ORDER BY cnt DESC;

// 2. Thống kê relationship
MATCH ()-[r]->()
RETURN type(r) AS rel_type, count(r) AS cnt
ORDER BY cnt DESC;

// 3. Full reasoning chain: Finding → Pattern → Condition ← Evidence → Source
MATCH (c:Case)-[:HAS_FINDING]->(f:Finding)
      -[:PART_OF_PATTERN]->(p:Pattern)
      -[:SUGGESTS]->(cond:Condition)
      <-[:SUPPORTS]-(e:Evidence)
      -[:FROM_SOURCE]->(src:Source)
RETURN c.case_id          AS case_id,
       f.name             AS finding,
       p.name             AS pattern,
       p.match_score      AS pattern_score,
       cond.name          AS condition,
       left(e.text, 200)  AS evidence_snippet,
       src.name           AS source,
       e.page             AS page
ORDER BY pattern_score DESC
LIMIT 20;

// 4. Truy xuất evidence theo chỉ số bất thường (dùng trong code)
// Thay 'WBC' bằng test_code cần tìm
MATCH (t:Test {test_code: 'WBC'})
      <-[:MENTIONS_TEST]-(e:Evidence)
      -[:FROM_SOURCE]->(src:Source)
RETURN e.id, left(e.text, 200) AS snippet, src.name, e.page, e.type, e.trust
ORDER BY e.trust DESC, e.score DESC
LIMIT 10;

// 5. Truy xuất evidence theo condition (dùng trong code)
MATCH (cond:Condition)
      <-[:SUPPORTS]-(e:Evidence)
      -[:FROM_SOURCE]->(src:Source)
WHERE cond.name IN ['bacterial_infection', 'infection']
RETURN e.id, left(e.text, 200) AS snippet, src.name, e.page, e.type
ORDER BY e.trust DESC
LIMIT 10;
