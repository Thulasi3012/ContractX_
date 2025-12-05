"""
Knowledge Graph Builder for ContractX
Converts document analysis into Neo4j graph structure
Uses database UUID for document identification
"""

from neo4j import GraphDatabase
import os
from typing import Dict, Any, List
from datetime import datetime

class KnowledgeGraphBuilder:
    """Build and manage document knowledge graph in Neo4j"""
    
    def __init__(self):
        # Neo4j connection
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("[OK] Neo4j connected successfully")
            print(f"  - URI: {self.uri}")
        except Exception as e:
            print(f"[ERROR] Neo4j connection failed: {e}")
            self.driver = None
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
    
    def build_graph(self, document_data: Dict[str, Any], db_document_id: str) -> Dict[str, Any]:
        """
        Build complete knowledge graph from document analysis
        
        Args:
            document_data: Extraction results
            db_document_id: UUID from database (CRITICAL)
            
        Structure:
        - Document (root node with UUID from DB)
          ├─> Page nodes
          │    ├─> Section nodes
          │    │    ├─> Clause nodes
          │    │    │    └─> SubClause nodes
          │    ├─> Table nodes
          │    │    ├─> TableRow nodes
          │    │    └─> TableCell nodes
          │    └─> Visual nodes
          └─> Entity nodes (shared across pages)
               ├─> Buyer
               ├─> Seller
               ├─> Date nodes
               ├─> Deadline nodes
               └─> Alert nodes
        """
        if not self.driver:
            return {
                "status": "failed",
                "error": "Neo4j not connected",
                "document_id": db_document_id
            }
        
        print("\n" + "=" * 80)
        print("Building Knowledge Graph")
        print(f"Document ID: {db_document_id}")
        print("=" * 80)
        
        try:
            with self.driver.session() as session:
                # Create Document root node with database UUID
                doc_id = self._create_document_node(session, document_data, db_document_id)
                print(f"[OK] Created Document node: {doc_id}")
                
                # Create Entity nodes (shared across document)
                summary = document_data.get('summary', {})
                entity_ids = self._create_entity_nodes(session, doc_id, summary.get('entities', {}))
                print(f"[OK] Created {len(entity_ids)} entity nodes")
                
                # Process each page
                total_nodes = 0
                total_relationships = 0
                
                for page_data in document_data.get('pages', []):
                    page_num = page_data['page_number']
                    print(f"\n[Page {page_num}] Processing...")
                    
                    # Create Page node
                    page_id = self._create_page_node(session, doc_id, page_data)
                    
                    # Create text structure (sections, clauses)
                    text_stats = self._create_text_structure(session, page_id, page_data.get('text_analysis', {}))
                    print(f"  - Text: {text_stats['sections']} sections, {text_stats['clauses']} clauses")
                    
                    # Create table nodes
                    table_stats = self._create_table_structure(session, page_id, page_data.get('tables', []))
                    print(f"  - Tables: {table_stats['tables']} tables, {table_stats['rows']} rows")
                    
                    # Create visual nodes
                    visual_stats = self._create_visual_structure(session, page_id, page_data.get('visuals', []))
                    print(f"  - Visuals: {visual_stats['visuals']} visuals")
                    
                    # Link page entities to global entities
                    text_analysis = page_data.get('text_analysis', {})
                    self._link_page_entities(session, page_id, text_analysis.get('entities', {}), entity_ids)
                    
                    total_nodes += text_stats['total_nodes'] + table_stats['total_nodes'] + visual_stats['total_nodes']
                    total_relationships += text_stats['relationships'] + table_stats['relationships'] + visual_stats['relationships']
                
                print("\n" + "=" * 80)
                print("Knowledge Graph Complete")
                print(f"  Document ID: {db_document_id}")
                print(f"  Total Nodes: {total_nodes}")
                print(f"  Total Relationships: {total_relationships}")
                print("=" * 80)
                
                return {
                    "status": "success",
                    "document_id": db_document_id,
                    "total_nodes": total_nodes,
                    "total_relationships": total_relationships
                }
        except Exception as e:
            print(f"[ERROR] Knowledge graph build failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "document_id": db_document_id
            }
    
    def _create_document_node(self, session, doc_data: Dict[str, Any], db_document_id: str) -> str:
        """Create root Document node using database UUID"""
        query = """
        CREATE (d:Document {
            id: $doc_id,
            filename: $filename,
            total_pages: $total_pages,
            total_sections: $total_sections,
            total_tables: $total_tables,
            total_visuals: $total_visuals,
            created_at: $timestamp
        })
        RETURN d.id as doc_id
        """
        
        summary = doc_data.get('summary', {})
        
        result = session.run(query, {
            "doc_id": db_document_id,  # Use database UUID
            "filename": doc_data.get('filename', 'unknown'),
            "total_pages": doc_data.get('total_pages', 0),
            "total_sections": summary.get('total_sections', 0),
            "total_tables": summary.get('total_tables', 0),
            "total_visuals": summary.get('total_visuals', 0),
            "timestamp": doc_data.get('timestamp', datetime.now().isoformat())
        })
        
        return result.single()['doc_id']
    
    def _create_entity_nodes(self, session, doc_id: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Create global entity nodes (Buyer, Seller, Dates, etc.)"""
        entity_ids = {}
        
        # Buyer node
        if entities.get('buyer_name'):
            query = """
            CREATE (e:Entity:Buyer {
                id: $entity_id,
                name: $name,
                type: 'buyer'
            })
            WITH e
            MATCH (d:Document {id: $doc_id})
            CREATE (d)-[:HAS_BUYER]->(e)
            RETURN e.id as entity_id
            """
            result = session.run(query, {
                "entity_id": f"{doc_id}_buyer",
                "name": entities['buyer_name'],
                "doc_id": doc_id
            })
            entity_ids['buyer'] = result.single()['entity_id']
        
        # Seller node
        if entities.get('seller_name'):
            query = """
            CREATE (e:Entity:Seller {
                id: $entity_id,
                name: $name,
                type: 'seller'
            })
            WITH e
            MATCH (d:Document {id: $doc_id})
            CREATE (d)-[:HAS_SELLER]->(e)
            RETURN e.id as entity_id
            """
            result = session.run(query, {
                "entity_id": f"{doc_id}_seller",
                "name": entities['seller_name'],
                "doc_id": doc_id
            })
            entity_ids['seller'] = result.single()['entity_id']
        
        # Date nodes
        entity_ids['dates'] = []
        for idx, date_str in enumerate(entities.get('dates', []), 1):
            query = """
            CREATE (e:Entity:Date {
                id: $entity_id,
                value: $date_value,
                type: 'date'
            })
            WITH e
            MATCH (d:Document {id: $doc_id})
            CREATE (d)-[:HAS_DATE]->(e)
            RETURN e.id as entity_id
            """
            result = session.run(query, {
                "entity_id": f"{doc_id}_date_{idx}",
                "date_value": date_str,
                "doc_id": doc_id
            })
            entity_ids['dates'].append(result.single()['entity_id'])
        
        # Deadline nodes
        entity_ids['deadlines'] = []
        for idx, deadline_str in enumerate(entities.get('deadlines', []), 1):
            query = """
            CREATE (e:Entity:Deadline {
                id: $entity_id,
                value: $deadline_value,
                type: 'deadline'
            })
            WITH e
            MATCH (d:Document {id: $doc_id})
            CREATE (d)-[:HAS_DEADLINE]->(e)
            RETURN e.id as entity_id
            """
            result = session.run(query, {
                "entity_id": f"{doc_id}_deadline_{idx}",
                "deadline_value": deadline_str,
                "doc_id": doc_id
            })
            entity_ids['deadlines'].append(result.single()['entity_id'])
        
        # Alert nodes
        entity_ids['alerts'] = []
        for idx, alert_str in enumerate(entities.get('alerts', []), 1):
            query = """
            CREATE (e:Entity:Alert {
                id: $entity_id,
                value: $alert_value,
                type: 'alert',
                severity: 'high'
            })
            WITH e
            MATCH (d:Document {id: $doc_id})
            CREATE (d)-[:HAS_ALERT]->(e)
            RETURN e.id as entity_id
            """
            result = session.run(query, {
                "entity_id": f"{doc_id}_alert_{idx}",
                "alert_value": alert_str,
                "doc_id": doc_id
            })
            entity_ids['alerts'].append(result.single()['entity_id'])
        
        return entity_ids
    
    def _create_page_node(self, session, doc_id: str, page_data: Dict[str, Any]) -> str:
        """Create Page node"""
        query = """
        CREATE (p:Page {
            id: $page_id,
            page_number: $page_number,
            sections_count: $sections_count,
            tables_count: $tables_count,
            visuals_count: $visuals_count
        })
        WITH p
        MATCH (d:Document {id: $doc_id})
        CREATE (d)-[:HAS_PAGE]->(p)
        RETURN p.id as page_id
        """
        
        page_id = f"{doc_id}_page_{page_data['page_number']}"
        text_analysis = page_data.get('text_analysis', {})
        
        result = session.run(query, {
            "page_id": page_id,
            "page_number": page_data['page_number'],
            "sections_count": text_analysis.get('sections_count', 0),
            "tables_count": len(page_data.get('tables', [])),
            "visuals_count": len(page_data.get('visuals', [])),
            "doc_id": doc_id
        })
        
        return result.single()['page_id']
    
    def _create_text_structure(self, session, page_id: str, text_analysis: Dict[str, Any]) -> Dict[str, int]:
        """Create Section → Clause → SubClause hierarchy"""
        stats = {"sections": 0, "clauses": 0, "sub_clauses": 0, "total_nodes": 0, "relationships": 0}
        
        for section in text_analysis.get('sections', []):
            # Create Section node
            section_id = f"{page_id}_section_{section.get('heading_id', 'unknown')}"
            query = """
            CREATE (s:Section {
                id: $section_id,
                heading: $heading,
                heading_id: $heading_id
            })
            WITH s
            MATCH (p:Page {id: $page_id})
            CREATE (p)-[:HAS_SECTION]->(s)
            RETURN s.id as section_id
            """
            session.run(query, {
                "section_id": section_id,
                "heading": section.get('heading', ''),
                "heading_id": section.get('heading_id', 'unknown'),
                "page_id": page_id
            })
            stats['sections'] += 1
            stats['total_nodes'] += 1
            stats['relationships'] += 1
            
            # Create Clauses under Section
            for clause in section.get('clauses', []):
                clause_id = f"{section_id}_clause_{clause.get('clause_id', 'unknown')}"
                query = """
                CREATE (c:Clause {
                    id: $clause_id,
                    text: $text,
                    clause_id: $clause_id_val
                })
                WITH c
                MATCH (s:Section {id: $section_id})
                CREATE (s)-[:HAS_CLAUSE]->(c)
                RETURN c.id as clause_id
                """
                session.run(query, {
                    "clause_id": clause_id,
                    "text": clause.get('clause', ''),
                    "clause_id_val": clause.get('clause_id', 'unknown'),
                    "section_id": section_id
                })
                stats['clauses'] += 1
                stats['total_nodes'] += 1
                stats['relationships'] += 1
                
                # Create SubClauses
                for sub_clause in clause.get('sub_clauses', []):
                    sub_clause_id = f"{clause_id}_subclause_{sub_clause.get('sub_clause_id', 'unknown')}"
                    query = """
                    CREATE (sc:SubClause {
                        id: $sub_clause_id,
                        text: $text,
                        sub_clause_id: $sub_clause_id_val
                    })
                    WITH sc
                    MATCH (c:Clause {id: $clause_id})
                    CREATE (c)-[:HAS_SUBCLAUSE]->(sc)
                    """
                    session.run(query, {
                        "sub_clause_id": sub_clause_id,
                        "text": sub_clause.get('sub_clause', ''),
                        "sub_clause_id_val": sub_clause.get('sub_clause_id', 'unknown'),
                        "clause_id": clause_id
                    })
                    stats['sub_clauses'] += 1
                    stats['total_nodes'] += 1
                    stats['relationships'] += 1
        
        return stats
    
    def _create_table_structure(self, session, page_id: str, tables: List[Dict[str, Any]]) -> Dict[str, int]:
        """Create Table → Row → Cell structure"""
        stats = {"tables": 0, "rows": 0, "cells": 0, "total_nodes": 0, "relationships": 0}
        
        for table in tables:
            # Create Table node
            table_id = f"{page_id}_{table.get('table_id', 'unknown')}"
            query = """
            CREATE (t:Table {
                id: $table_id,
                table_id: $table_id_val,
                title: $title,
                table_type: $table_type,
                rows: $rows,
                columns: $columns,
                has_merged_cells: $has_merged_cells
            })
            WITH t
            MATCH (p:Page {id: $page_id})
            CREATE (p)-[:HAS_TABLE]->(t)
            RETURN t.id as table_id
            """
            session.run(query, {
                "table_id": table_id,
                "table_id_val": table.get('table_id', 'unknown'),
                "title": table.get('table_title', ''),
                "table_type": table.get('table_type', 'unknown'),
                "rows": table.get('total_rows', 0),
                "columns": table.get('total_columns', 0),
                "has_merged_cells": table.get('has_merged_cells', False),
                "page_id": page_id
            })
            stats['tables'] += 1
            stats['total_nodes'] += 1
            stats['relationships'] += 1
            
            # Create Header row
            if table.get('headers'):
                header_id = f"{table_id}_header"
                query = """
                CREATE (h:TableRow:Header {
                    id: $header_id,
                    row_index: 0,
                    values: $values
                })
                WITH h
                MATCH (t:Table {id: $table_id})
                CREATE (t)-[:HAS_HEADER]->(h)
                """
                session.run(query, {
                    "header_id": header_id,
                    "values": table['headers'],
                    "table_id": table_id
                })
                stats['rows'] += 1
                stats['total_nodes'] += 1
                stats['relationships'] += 1
            
            # Create Data rows
            for row_idx, row_data in enumerate(table.get('rows', []), 1):
                row_id = f"{table_id}_row_{row_idx}"
                query = """
                CREATE (r:TableRow {
                    id: $row_id,
                    row_index: $row_index,
                    values: $values
                })
                WITH r
                MATCH (t:Table {id: $table_id})
                CREATE (t)-[:HAS_ROW]->(r)
                """
                session.run(query, {
                    "row_id": row_id,
                    "row_index": row_idx,
                    "values": row_data,
                    "table_id": table_id
                })
                stats['rows'] += 1
                stats['total_nodes'] += 1
                stats['relationships'] += 1
        
        return stats
    
    def _create_visual_structure(self, session, page_id: str, visuals: List[Dict[str, Any]]) -> Dict[str, int]:
        """Create Visual nodes"""
        stats = {"visuals": 0, "total_nodes": 0, "relationships": 0}
        
        for visual in visuals:
            visual_id = f"{page_id}_{visual.get('visual_id', 'unknown')}"
            query = """
            CREATE (v:Visual {
                id: $visual_id,
                visual_id: $visual_id_val,
                type: $type,
                width: $width,
                height: $height,
                area: $area
            })
            WITH v
            MATCH (p:Page {id: $page_id})
            CREATE (p)-[:HAS_VISUAL]->(v)
            RETURN v.id as visual_id
            """
            try:
                session.run(query, {
                    "visual_id": visual_id,
                    "visual_id_val": visual.get('visual_id', 'unknown'),
                    "type": visual.get('type', 'unknown'),
                    "width": visual.get('width', 0),
                    "height": visual.get('height', 0),
                    "area": visual.get('area', 0),
                    "page_id": page_id
                })
                stats['visuals'] += 1
                stats['total_nodes'] += 1
                stats['relationships'] += 1
            except Exception as e:
                print(f"  [!] Error creating visual node: {e}")
        
        return stats
    
    def _link_page_entities(self, session, page_id: str, page_entities: Dict[str, Any], global_entity_ids: Dict[str, Any]):
        """Link page to global entity nodes"""
        # Link to Buyer
        if page_entities.get('buyer_name') and 'buyer' in global_entity_ids:
            query = """
            MATCH (p:Page {id: $page_id})
            MATCH (e:Buyer {id: $entity_id})
            CREATE (p)-[:MENTIONS_BUYER]->(e)
            """
            session.run(query, {"page_id": page_id, "entity_id": global_entity_ids['buyer']})
        
        # Link to Seller
        if page_entities.get('seller_name') and 'seller' in global_entity_ids:
            query = """
            MATCH (p:Page {id: $page_id})
            MATCH (e:Seller {id: $entity_id})
            CREATE (p)-[:MENTIONS_SELLER]->(e)
            """
            session.run(query, {"page_id": page_id, "entity_id": global_entity_ids['seller']})
    
    def query_graph(self, cypher_query: str) -> List[Dict[str, Any]]:
        """Execute Cypher query on graph"""
        if not self.driver:
            return []
        
        with self.driver.session() as session:
            result = session.run(cypher_query)
            return [dict(record) for record in result]
    
    def clear_graph(self):
        """Clear entire graph (use with caution!)"""
        if not self.driver:
            return
        
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("[OK] Graph cleared")
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete entire document and all related nodes from Knowledge Graph
        
        Args:
            document_id: UUID of document to delete
            
        Returns:
            Success status
        """
        if not self.driver:
            print(f"[KG ERROR] Neo4j not connected")
            return False
        
        try:
            with self.driver.session() as session:
                # Query to find and delete all nodes related to this document
                query = """
                MATCH (d:Document {id: $doc_id})
                DETACH DELETE d
                RETURN COUNT(*) as deleted_count
                """
                
                result = session.run(query, {"doc_id": document_id})
                deleted = result.single()
                
                if deleted and deleted['deleted_count'] > 0:
                    print(f"[OK] Deleted document from KG: {document_id}")
                    
                    # Also delete orphaned entity nodes if they're not connected to other documents
                    cleanup_query = """
                    MATCH (e:Entity)
                    WHERE NOT (e)<--(:Document)
                    DETACH DELETE e
                    RETURN COUNT(*) as cleaned_count
                    """
                    
                    cleanup_result = session.run(cleanup_query)
                    cleanup_count = cleanup_result.single()['cleaned_count']
                    
                    if cleanup_count > 0:
                        print(f"[OK] Cleaned up {cleanup_count} orphaned entity nodes")
                    
                    return True
                else:
                    print(f"[WARNING] Document not found in KG: {document_id}")
                    return False
                    
        except Exception as e:
            print(f"[KG ERROR] Failed to delete document from KG: {e}")
            return False