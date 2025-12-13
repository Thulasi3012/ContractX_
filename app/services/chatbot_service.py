# app/services/chatbot_service.py
"""
Chatbot Service - Combines RAG and Knowledge Graph for intelligent Q&A
"""

import google.generativeai as genai
import os
from typing import Dict, Any, List
from app.services.rag_service import RAGService
from app.services.knowledge_graph_builder import KnowledgeGraphBuilder
from app.config.config import Config

class ChatbotService:
    """Intelligent chatbot combining RAG + Knowledge Graph"""
    
    def __init__(self):
        # Configure Gemini
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(Config.GEMINI_MODEL)

        # Initialize services
        self.rag_service = RAGService()
        self.kg_builder = KnowledgeGraphBuilder()
        
        print("[OK] Chatbot Service initialized")
        print(f"  - LLM: {self.model}")
        print("  - RAG: Enabled")
        print("  - KG: Enabled")
    
    def chat(
        self, 
        document_id: str, 
        question: str,
        include_rag: bool = True,
        include_kg: bool = True,
        n_rag_results: int = 8
    ) -> Dict[str, Any]:
        """
        Answer question using RAG + KG context
        
        Args:
            document_id: UUID of the document
            question: User's question
            include_rag: Whether to include RAG context
            include_kg: Whether to include KG context
            n_rag_results: Number of RAG results to retrieve
        """
        print(f"\n[CHATBOT] Processing question for document: {document_id}")
        print(f"Question: {question}")
        
        # Step 1: Retrieve from RAG (vector store)
        rag_context = []
        if include_rag:
            print("\n[1/3] Retrieving from RAG...")
            rag_results = self.rag_service.search(
                query=question,
                document_id=document_id,
                n_results=n_rag_results
            )
            
            if rag_results:
                print(f"  - Found {len(rag_results)} relevant chunks")
                rag_context = self._format_rag_results(rag_results)
            else:
                print("  - No relevant chunks found")
        
        # Step 2: Retrieve from Knowledge Graph
        kg_context = []
        if include_kg:
            print("\n[2/3] Retrieving from Knowledge Graph...")
            kg_results = self._query_kg_for_context(document_id, question)
            
            if kg_results:
                print(f"  - Found {len(kg_results)} relevant nodes/relationships")
                kg_context = self._format_kg_results(kg_results)
            else:
                print("  - No relevant graph data found")
        
        # Step 3: Generate answer using LLM
        print("\n[3/3] Generating answer with Gemini...")
        
        # Build comprehensive context
        full_context = self._build_combined_context(rag_context, kg_context)
        
        # Generate answer
        answer = self._generate_answer(question, full_context, document_id)
        
        print("[OK] Answer generated")
        
        return {
            "document_id": document_id,
            "question": question,
            "answer": answer['text'],
            "sources": {
                "rag_sources": rag_context,
                "kg_sources": kg_context
            },
            "context_used": {
                "rag_chunks": len(rag_context),
                "kg_nodes": len(kg_context)
            },
            "confidence": answer.get('confidence', 'medium')
        }
    
    def _format_rag_results(self, rag_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format RAG results for context building"""
        formatted = []
        
        for result in rag_results:
            formatted.append({
                "content": result['content'],
                "type": result['metadata'].get('type', 'unknown'),
                "page": result['metadata'].get('page_number', 'unknown'),
                "section": result['metadata'].get('section_heading', ''),
                "source": "RAG"
            })
        
        return formatted
    
    def _format_kg_results(self, kg_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format Knowledge Graph results for context building"""
        formatted = []
        
        for result in kg_results:
            formatted_item = {
                "type": result.get('type', 'unknown'),
                "source": "KG"
            }
            
            # Format based on type
            if result['type'] == 'entity':
                formatted_item['entity_type'] = result.get('entity_type')
                formatted_item['name'] = result.get('name')
            elif result['type'] == 'deadline':
                formatted_item['value'] = result.get('value')
            elif result['type'] == 'alert':
                formatted_item['value'] = result.get('value')
                formatted_item['severity'] = result.get('severity', 'medium')
            elif result['type'] == 'section':
                formatted_item['heading'] = result.get('heading')
                formatted_item['page'] = result.get('page')
                formatted_item['clauses'] = result.get('clauses', [])
            elif result['type'] == 'table':
                formatted_item['title'] = result.get('title')
                formatted_item['page'] = result.get('page')
                formatted_item['headers'] = result.get('headers')
                formatted_item['sample_rows'] = result.get('sample_rows')
            elif result['type'] == 'visual':
                formatted_item['visual_type'] = result.get('visual_type')
                formatted_item['page'] = result.get('page')
            
            formatted.append(formatted_item)
        
        return formatted
    
    def _query_kg_for_context(self, document_id: str, question: str) -> List[Dict[str, Any]]:
        """
        Query Knowledge Graph for relevant context
        Uses keyword extraction and graph traversal
        """
        if not self.kg_builder.driver:
            return []
        
        # Extract keywords from question
        keywords = self._extract_keywords(question)
        
        results = []
        
        try:
            with self.kg_builder.driver.session() as session:
                # Query 1: Find document and its key entities
                query = """
                MATCH (d:Document {id: $doc_id})
                OPTIONAL MATCH (d)-[:HAS_BUYER]->(buyer:Buyer)
                OPTIONAL MATCH (d)-[:HAS_SELLER]->(seller:Seller)
                OPTIONAL MATCH (d)-[:HAS_DEADLINE]->(deadline:Deadline)
                OPTIONAL MATCH (d)-[:HAS_ALERT]->(alert:Alert)
                RETURN d, buyer, seller, collect(DISTINCT deadline) as deadlines, collect(DISTINCT alert) as alerts
                """
                result = session.run(query, {"doc_id": document_id})
                record = result.single()
                
                if record:
                    # Add document info
                    if record['buyer']:
                        results.append({
                            "type": "entity",
                            "entity_type": "buyer",
                            "name": record['buyer'].get('name', ''),
                            "source": "KG"
                        })
                    
                    if record['seller']:
                        results.append({
                            "type": "entity",
                            "entity_type": "seller",
                            "name": record['seller'].get('name', ''),
                            "source": "KG"
                        })
                    
                    for deadline in record['deadlines']:
                        if deadline:
                            results.append({
                                "type": "deadline",
                                "value": deadline.get('value', ''),
                                "source": "KG"
                            })
                    
                    for alert in record['alerts']:
                        if alert:
                            results.append({
                                "type": "alert",
                                "value": alert.get('value', ''),
                                "severity": alert.get('severity', 'medium'),
                                "source": "KG"
                            })
                
                # Query 2: Search for relevant sections/clauses using keywords
                if keywords:
                    keyword_pattern = "|".join(keywords)
                    query = """
                    MATCH (d:Document {id: $doc_id})-[:HAS_PAGE]->(p:Page)-[:HAS_SECTION]->(s:Section)
                    WHERE any(kw IN $keywords WHERE toLower(s.heading) CONTAINS toLower(kw))
                    OPTIONAL MATCH (s)-[:HAS_CLAUSE]->(c:Clause)
                    RETURN s.heading as section, s.heading_id as section_id, 
                           p.page_number as page, collect(c.text) as clauses
                    LIMIT 5
                    """
                    result = session.run(query, {"doc_id": document_id, "keywords": keywords})
                    
                    for record in result:
                        results.append({
                            "type": "section",
                            "heading": record['section'],
                            "page": record['page'],
                            "clauses": [c for c in record['clauses'] if c][:3],  # Limit to 3 clauses
                            "source": "KG"
                        })
                
                # Query 3: Search tables
                if any(kw in question.lower() for kw in ['table', 'payment', 'schedule', 'milestone', 'price', 'cost']):
                    query = """
                    MATCH (d:Document {id: $doc_id})-[:HAS_PAGE]->(p:Page)-[:HAS_TABLE]->(t:Table)
                    OPTIONAL MATCH (t)-[:HAS_HEADER]->(h:TableRow)
                    OPTIONAL MATCH (t)-[:HAS_ROW]->(r:TableRow)
                    RETURN t.title as title, t.table_id as table_id, p.page_number as page,
                           h.values as headers, collect(r.values)[0..3] as sample_rows
                    LIMIT 3
                    """
                    result = session.run(query, {"doc_id": document_id})
                    
                    for record in result:
                        results.append({
                            "type": "table",
                            "title": record['title'] or "Untitled Table",
                            "page": record['page'],
                            "headers": record['headers'],
                            "sample_rows": record['sample_rows'],
                            "source": "KG"
                        })
                
                # Query 4: Search visuals
                if any(kw in question.lower() for kw in ['chart', 'graph', 'diagram', 'image', 'figure']):
                    query = """
                    MATCH (d:Document {id: $doc_id})-[:HAS_PAGE]->(p:Page)-[:HAS_VISUAL]->(v:Visual)
                    RETURN v.type as visual_type, v.visual_id as visual_id, p.page_number as page
                    LIMIT 5
                    """
                    result = session.run(query, {"doc_id": document_id})
                    
                    for record in result:
                        results.append({
                            "type": "visual",
                            "visual_type": record['visual_type'],
                            "page": record['page'],
                            "source": "KG"
                        })
        
        except Exception as e:
            print(f"[ERROR] KG query failed: {e}")
        
        return results
    
    def _extract_keywords(self, question: str) -> List[str]:
        """Extract important keywords from question"""
        # Common stopwords to ignore
        stopwords = {'what', 'when', 'where', 'who', 'how', 'is', 'are', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or'}
        
        words = question.lower().split()
        keywords = [w.strip('?.,!') for w in words if w not in stopwords and len(w) > 2]
        
        return keywords[:5]  # Return top 5 keywords
    
    def _build_combined_context(
        self, 
        rag_context: List[Dict[str, Any]], 
        kg_context: List[Dict[str, Any]]
    ) -> str:
        """Build combined context from RAG + KG sources"""
        context_parts = []
        
        # Add RAG context
        if rag_context:
            context_parts.append("=== RELEVANT TEXT FROM DOCUMENT (RAG) ===\n")
            for idx, item in enumerate(rag_context, 1):
                context_parts.append(f"{idx}. [Page {item['page']}] [{item['type'].upper()}]")
                if item.get('section'):
                    context_parts.append(f"   Section: {item['section']}")
                context_parts.append(f"   {item['content']}\n")
        
        # Add KG context
        if kg_context:
            context_parts.append("\n=== STRUCTURED INFORMATION FROM KNOWLEDGE GRAPH ===\n")
            
            # Group by type
            entities = [x for x in kg_context if x['type'] == 'entity']
            sections = [x for x in kg_context if x['type'] == 'section']
            tables = [x for x in kg_context if x['type'] == 'table']
            deadlines = [x for x in kg_context if x['type'] == 'deadline']
            alerts = [x for x in kg_context if x['type'] == 'alert']
            visuals = [x for x in kg_context if x['type'] == 'visual']
            
            # Add entities
            if entities:
                context_parts.append("PARTIES:")
                for entity in entities:
                    context_parts.append(f"  - {entity['entity_type'].title()}: {entity['name']}")
                context_parts.append("")
            
            # Add deadlines
            if deadlines:
                context_parts.append("DEADLINES:")
                for deadline in deadlines:
                    context_parts.append(f"  - {deadline['value']}")
                context_parts.append("")
            
            # Add alerts
            if alerts:
                context_parts.append("ALERTS:")
                for alert in alerts:
                    context_parts.append(f"  - [{alert['severity'].upper()}] {alert['value']}")
                context_parts.append("")
            
            # Add sections
            if sections:
                context_parts.append("RELEVANT SECTIONS:")
                for section in sections:
                    context_parts.append(f"  - {section['heading']} (Page {section['page']})")
                    if section.get('clauses'):
                        for clause in section['clauses'][:2]:  # Limit to 2 clauses
                            context_parts.append(f"    • {clause}")
                context_parts.append("")
            
            # Add tables
            if tables:
                context_parts.append("RELEVANT TABLES:")
                for table in tables:
                    context_parts.append(f"  - {table['title']} (Page {table['page']})")
                    if table.get('headers'):
                        context_parts.append(f"    Headers: {', '.join(table['headers'])}")
                    if table.get('sample_rows'):
                        context_parts.append(f"    Sample data available: {len(table['sample_rows'])} rows")
                context_parts.append("")
            
            # Add visuals
            if visuals:
                context_parts.append("VISUALS IN DOCUMENT:")
                for visual in visuals:
                    context_parts.append(f"  - {visual['visual_type']} on Page {visual['page']}")
                context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _generate_answer(
        self, 
        question: str, 
        context: str, 
        document_id: str
    ) -> Dict[str, Any]:
        """Generate answer using Gemini LLM"""
        
        prompt = f"""You are a precise AI assistant analyzing technical documents. Your primary goal is to provide accurate, factual answers based strictly on the provided context.

DOCUMENT ID: {document_id}

CONTEXT FROM DOCUMENT:
{context}

USER QUESTION:
{question}

CRITICAL INSTRUCTIONS:

1. ACCURACY REQUIREMENTS:
   - Answer ONLY using information explicitly stated in the context above
   - For numerical values (zones, reference numbers, IDs, dates, amounts, quantities), extract the EXACT value from the context
   - Double-check all numbers, codes, and identifiers before responding
   - If the answer is not found in the context, state: "I cannot find this information in the document."

2. CRITICAL DATA HANDLING (Reference Numbers, IDs, Zones, Dates, Payments, Orders):
   - Extract exact values from tables and structured data
   - Verify the specific row/entry that matches ALL criteria in the question
   - For multi-part questions (e.g., "What is X for item Y in zone Z?"), ensure all conditions match
   - Cite the page number and table/section reference
   - Format: "For [specific item], the [requested field] is [exact value] (Page X, Table Y)"

3. RESPONSE LENGTH GUIDELINES:
   - For simple factual queries (numbers, codes, single values): Provide short, direct answers
   - For technical explanations or multi-part questions: Provide comprehensive details
   - For "how/why/explain" questions: Include context and reasoning
   - Always prioritize accuracy over brevity

4. STRUCTURED RESPONSES:
   - Use bullet points ONLY when answering multiple related items or listing components
   - For single values: Use clear, direct sentences
   - For tables: Explicitly state "According to [Table Name] on Page X..."
   - Include relevant metadata: page numbers, section titles, table references

5. HANDLING AMBIGUITY:
   - If multiple matches exist, list all and ask for clarification
   - If the question is unclear, provide the closest match and note any assumptions
   - For partial matches, state what was found and what is missing

6. SPECIAL ATTENTION TO:
   - Zone numbers and codes
   - ATA references
   - Part numbers and functional designations
   - Dates, deadlines, and time-sensitive information
   - Financial data (payments, costs, invoices)
   - Compliance and regulatory information

7.CRITICAL TABLE ANSWERING RULES:

    7.1. Never infer or rename column labels.
    7.2. Column names must match retrieved table headers exactly.
    7.3. Never guess the column name.
    7.4. If a question asks where a value is held, determine the column strictly from the retrieved table.
    7.5. If column certainty is less than 100%, respond using this exact template:

    "The value <VALUE> appears in the <COLUMN_NAME> column according to the table."

    7.6. Replace <VALUE> and <COLUMN_NAME> dynamically using retrieved data only.
    7.7. If no column can be confidently identified, explicitly state that the column cannot be determined from the table.

ANSWER:"""
        
        try:
            response = self.model.generate_content(prompt)
            answer_text = response.text
            
            # Determine confidence based on context availability
            confidence = "high" if context and len(context) > 100 else "low"
            
            return {
                "text": answer_text,
                "confidence": confidence
            }
        
        except Exception as e:
            print(f"[ERROR] Answer generation failed: {e}")
            return {
                "text": "I apologize, but I encountered an error while generating the answer. Please try again.",
                "confidence": "low"
            }
    
    def get_conversation_summary(self, document_id: str) -> Dict[str, Any]:
        """Get a summary of document for conversation context"""
        # Get RAG stats
        rag_stats = self.rag_service.get_document_stats(document_id)
        
        # Get KG stats
        kg_stats = {"available": False}
        if self.kg_builder.driver:
            try:
                with self.kg_builder.driver.session() as session:
                    query = """
                    MATCH (d:Document {id: $doc_id})
                    OPTIONAL MATCH (d)-[:HAS_BUYER]->(buyer:Buyer)
                    OPTIONAL MATCH (d)-[:HAS_SELLER]->(seller:Seller)
                    RETURN d.filename as filename, d.total_pages as pages,
                           buyer.name as buyer, seller.name as seller
                    """
                    result = session.run(query, {"doc_id": document_id})
                    record = result.single()
                    
                    if record:
                        kg_stats = {
                            "available": True,
                            "filename": record['filename'],
                            "pages": record['pages'],
                            "buyer": record['buyer'],
                            "seller": record['seller']
                        }
            except Exception as e:
                print(f"[ERROR] KG summary query failed: {e}")
        
        return {
            "document_id": document_id,
            "rag_stats": rag_stats,
            "kg_stats": kg_stats
        }