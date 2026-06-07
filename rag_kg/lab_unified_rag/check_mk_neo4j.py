from neo4j import GraphDatabase

for pwd in ["25251325", "260803", "26082003", "admin"]:
    try:
        d = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", pwd))
        d.verify_connectivity()
        print(f"✓ PASSWORD ĐÚNG: '{pwd}'")
        d.close()
        break
    except Exception as e:
        print(f"✗ '{pwd}' sai: {e}")